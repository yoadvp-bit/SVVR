#preprocess_volumes.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import os
import copy
import timeit
import pickle
import pydicom
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
from statistics import mode

import visualize_volumes
#dicom_numpy code is from https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py
#downloaded on September 19, 2019
#see dicom_numpy/LICENSE.txt for the dicom_numpy license 
from dicom_numpy import combine_slices

class CleanCTScans(object):
    def __init__(self, mode, old_log_file_path, identifiers_path, results_path,
                 dirty_path, clean_path):
        """Variables:
        <mode>:
            if 'testing' then run entire pipeline but only on a few images,
                and save visualizations on each image
            if 'run' then run entire pipeline on all images
        <old_log_file_path>:
            path to a CSV log file from a previous run of this module. If this
            path is not the empty string '' then the logging dataframe will be
            initialized from this old log file and the module will pick up
            where it left off.
        <identifiers_path>:
            path to a CSV with columns 'MRN', 'Accession', and 'Set_Assigned.'
            Example fake excerpt:         
                MRN,Accession,Set_Assigned
                FAKEMRN123,AAFAKE123,train
                FAKEMRN567,AAFAKE567,test
        <results_path>: path to a directory where you want the log files from
            this run to be saved.
        <dirty_path>: location of the 'dirty' CT scans that need to be cleaned.
            Each file in this directory represents one CT scan.
            The file is a pickled Python list. Each element of the list is a
            pydicom.dataset.FileDataset that represents a DICOM file and thus
            contains metadata as well as pixel data. Each
            pydicom.dataset.FileDataset corresponds to one slice of the CT scan.
            The slices are not necessarily 'in order' in this list.
        <clean_path>: location where the cleaned CT scans produced by this
            code will be saved as .npz files."""
        self.mode = mode
        assert self.mode == 'testing' or self.mode == 'run'
        self.old_log_file_path = old_log_file_path
        self.identifiers_path = identifiers_path
        
        #Location to save visualizations and other meta-output of this script:
        self.logdir = os.path.join(results_path,datetime.datetime.today().strftime('%Y-%m-%d')+'_volume_preprocessing')
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        
        #Locations of CT scans
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        
        #Set up input and output paths and necessary filenames
        self.set_up_identifiers()
        self.set_up_logdf()
        
        #Run
        self.clean_all_pace_data() 
           
    #############
    # PACE Data #---------------------------------------------------------------
    #############
    def set_up_identifiers(self):
        """Read in the identifiers file which defines the data splits as well
        as the fake filename translation. Also read in all of the available
        filenames."""
        #Read in identifiers file
        #identifiers file includes: ['MRN','Accession','Set_Assigned']
        ids_file = os.path.join(self.identifiers_path,'all_identifiers.csv')
        self.ids_df = pd.read_csv(ids_file, header=0, index_col = 'Accession')
        self.note_accessions = self.ids_df.index.values.tolist() #e.g. AA12345
        
        if self.mode == 'testing':
            #Option 1: run on 10 random volumes:
            #self.note_accessions = self.note_accessions[0:10]
            #Option 2: run on pre-specified volumes:
            self.note_accessions = [x[0] for x in pd.read_csv(os.path.join(self.identifiers_path,'testing_ids.csv'),header=0).values.tolist()]
        
        #The actual filenames of the downloaded volumes may be prefixed with
        #additional letters besides AA, will have an underscore followed by
        #a number at the end, and will have the .pkl file extension at the end.
        #For example RHAA12345_3.pkl. Thus they are not exact string matches
        #to the accessions in self.note_accessions.
        #Also until the data is completely downloaded, self.volume_accessions
        #will have fewer entries in it than self.note_accessions because
        #self.volume_accessions represents what's actually been downloaded
        #while self.note_accessions represents everything that COULD BE downloaded
        self.volume_accessions = os.listdir(self.dirty_path)
    
    def set_up_logdf(self):
        if self.old_log_file_path == '':
            #there is no preexisting log file; initialize from scratch
            self.logdf = copy.deepcopy(self.ids_df)
            for colname in ['status','status_reason','full_filename_pkl','full_filename_npz',
                            'orig_square','orig_numslices','orig_slope',
                            'orig_inter','orig_yxspacing',
                            'orig_zpositions_all','orig_zdiff_all','orig_zdiff_set','orig_zdiff','zdiffs_all_equal',
                            'orig_orientation','orig_gantry_tilt','final_square',
                            'final_numslices','final_spacing','transform']:
                self.logdf[colname]=''
        else:
            #there is a preexisting log file; start from this.
            self.logdf = pd.read_csv(self.old_log_file_path,header=0,index_col=0)
            self.update_note_accessions_from_existing()
        
        #Ensure correct dtypes
        for colname in ['transform','orig_zpositions_all','orig_zdiff_all','orig_zdiff_set']:
            self.logdf[colname]=self.logdf[colname].astype('object')
    
    def update_note_accessions_from_existing(self):
        """If there is a preexisting log file defined by self.old_log_file_path,
        then update self.note_accessions so that it includes only the note_accs
        which need to be processed in this run."""
        print('Initializing logging df from previous version at ',self.old_log_file_path)
        print('Total available note accessions:',len(self.note_accessions))
        volume_not_found = self.logdf[self.logdf['status_reason']=='volume_not_found'].index.values.tolist()
        print('number of volume_not_found',len(volume_not_found))
        wrong_submatch = self.logdf[self.logdf['status_reason']=='processed_wrong_submatch_accession'].index.values.tolist()
        print('number of processed_wrong_submatch_accession',len(wrong_submatch))
        self.note_accessions = volume_not_found + wrong_submatch
        print('Total note accessions that will be processed:',len(self.note_accessions))
        
    def clean_all_pace_data(self):
        """Clean each image and save it as a .npz compressed array. Save a log
        file documenting the cleaning process and metadata."""
        t0 = timeit.default_timer()
        count = 0
        for note_acc in self.note_accessions:
            if self.mode=='testing': print('Working on',note_acc)
            #note_acc is e.g. AA12345. full_filename_pkl is e.g. RHAA12345_3.pkl
            #full_filename_pkl is the full filename of the raw dirty CT.
            #full_filename_pkl will be 'fail' if not found.
            full_filename_pkl = self.find_full_filename_pkl(note_acc)
            self.logdf.at[note_acc,'full_filename_pkl'] = full_filename_pkl
            
            if full_filename_pkl == 'fail':
                self.logdf.at[note_acc,'status'] = 'fail'
                self.logdf.at[note_acc,'status_reason'] = 'volume_not_found'
                print('\tFailed on',note_acc,'because volume was not found')
            
            if full_filename_pkl != 'fail':
                try:
                    #Clean
                    ctvol = self.process_ctvol(full_filename_pkl, note_acc)
                    #Save
                    full_filename_npz = full_filename_pkl.replace('.pkl','.npz') #e.g. RHAA12345_3.npz
                    self.logdf.at[note_acc,'full_filename_npz'] = full_filename_npz
                    out_filename_path = os.path.join(self.clean_path, full_filename_npz)
                    #Save using lossless compression (zip algorithm) so that there
                    #will be enough space in local storage to fit the entire dataset
                    np.savez_compressed(out_filename_path,ct=ctvol)
                    self.logdf.at[note_acc,'status'] = 'success'
                    self.report_unequal_zdiffs(note_acc, ctvol)
                except Exception as e:
                    self.logdf.at[note_acc,'status'] = 'fail'
                    self.logdf.at[note_acc,'status_reason'] = str(e)
                    print('\tFailed on',note_acc,'due to',str(e))
            count+=1
            if count % 20 == 0: self.report_progress_and_save_logfile(count,t0,note_acc)
    
    def report_unequal_zdiffs(self, note_acc, ctvol):
        """Check whether the zdiffs are equal for the whole scan or not. If not,
        print a warning message and make gifs for that final volume to enable
        visual inspection later."""
        if self.logdf.at[note_acc,'zdiffs_all_equal'] is False:
            orig_zdiff_set = str(self.logdf.at[note_acc,'orig_zdiff_set'])
            zmode = str(self.logdf.at[note_acc,'orig_zdiff'])
            print('\tWarning: zdiffs not all equal for',note_acc,':',orig_zdiff_set,'. Mode chosen:',zmode)
            #Uncomment the next line to make a GIF as a sanity check:
            #visualize_volumes.make_gifs(ctvol,os.path.join(self.logdir,note_acc+'_zdiffs_unequal'), chosen_views=['coronal'])
    
    def report_progress_and_save_logfile(self,count,t0,note_acc):
        t1 = timeit.default_timer()
        percent = round(float(count)/len(self.note_accessions), 2)*100
        elapsed = round((t1 - t0)/60.0,2)
        print('Finished up to',note_acc,count,'=',percent,'percent. Elapsed time:',elapsed,'min')
        try:
            self.logdf.to_csv(os.path.join(self.logdir,'CT_Scan_Preprocessing_Log_File.csv'))
            print('Saved log file')
        except Exception as e:
            print('Could not save log file this time due to Exception',e)
    
    def find_full_filename_pkl(self, note_acc):
        """e.g. if <note_acc>==AA12345 return RHAA12345_3.pkl, which is the
        full filename corresponding to this accession number.
        Previously I was checking if note_acc in full_filename_pkl but that does
        not work because if note_acc = AA1234, then it could match with
        multiple possible full_filename_pkls including AA123456 and AA12345678.
        Therefore I have to split the full_filename_pkl and ensure an exact
        match with the first part of the name."""
        for full_filename_pkl in self.volume_accessions:
            full_filename_pkl_extract = full_filename_pkl.split('_')[0].replace('RH','').replace('B','') #e.g. RHAA1234_6.pkl --> AA1234
            if note_acc == full_filename_pkl_extract:
                return full_filename_pkl
        return 'fail'
    
    def process_ctvol(self, full_filename_pkl, note_acc):
        """Read in the pickled CT volume and return the CT volume as a numpy
        array. Save to the log file important characteristics. Includes sanity checks."""
        #Load volume. Format: python list. Each element of the list is a
        #pydicom.dataset.FileDataset that contains metadata as well as pixel
        #data. Each pydicom.dataset.FileDataset corresponds to one slice.
        raw = pickle.load(open(os.path.join(self.dirty_path, full_filename_pkl),'rb'))
        
        #Extract information from all the slices (function
        #includes for loop over slices) and save to self.logdf
        if self.mode=='testing': print('running extract_info()')
        self.logdf = CleanCTScans.extract_info(raw, note_acc, self.logdf)
        
        #Create the volume by stacking the slices in the right order:
        if self.mode=='testing': print('running create_volume()')
        ctvol, self.logdf = CleanCTScans.create_volume(raw, note_acc, self.logdf)
        #if self.mode == 'testing': self.visualize(ctvol,note_acc,'_1_raw_and_rescaled')
        
        #Resample
        if self.mode=='testing': print('running resample_volume()')
        orig_yxspacing = self.logdf.at[note_acc,'orig_yxspacing']
        orig_zspacing = self.logdf.at[note_acc,'orig_zdiff']
        #z first, then square (xy) for the sitk resampling function:
        original_spacing = [float(x) for x in [orig_zspacing, orig_yxspacing, orig_yxspacing]] 
        ctvol, self.logdf = CleanCTScans.resample_volume(ctvol, original_spacing, note_acc, self.logdf, self.mode)
        #if self.mode == 'testing': self.visualize(ctvol,note_acc,'_2_resampled')
        
        #Represent the volume more efficiently by casting to float16 and
        #clipping the Hounsfield Units:
        ctvol = CleanCTScans.represent_volume_efficiently_and_transpose(ctvol)
        if self.mode == 'testing': self.visualize(ctvol,note_acc,'_final')
        
        #Return the final volume. ctvol is a 3D numpy array with float16 elements
        return ctvol
    
    ##################
    # Helper Methods #----------------------------------------------------------
    ##################
    @staticmethod
    def extract_info(raw, note_acc, logdf):
        """Process the CT data contained in <raw> and save properties of the
        data in <logdf> under index <note_acc>.
        Variables:
        <raw> is a list of pydicom.dataset.FileDataset objects, one for
            each slice."""
        #Initialize empty lists for collecting info on each slice
        #These values should all be identical for each slice but we want to
        #double check that, because TRUST NOTHING
        gathered_slopes = []
        gathered_inters = []
        gathered_spacing = []
        gathered_orientation = []
        #I believe the gantry tilt should always be zero for chest CTs
        #but it makes sense to save it and record it just in case.
        gathered_gantry_tilt = []
        
        for index in range(len(raw)):
            oneslice = raw[index]
            
            #Gather the scaling slope and intercept: https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
            scl_slope = oneslice.data_element('RescaleSlope').value #example: "1"
            gathered_slopes.append(scl_slope)
            scl_inter = oneslice.data_element('RescaleIntercept').value #example: "-1024"
            gathered_inters.append(scl_inter)
            assert oneslice.data_element('RescaleType').value == 'HU', 'Error: RescaleType not equal to HU'
            
            #Gather the spacing: example: ['0.585938', '0.585938']; what we save is 0.585938
            #for DICOM, first vlaue is row spacing (vertical spacing) and
            #the second value is column spacing (horizontal spacing)
            yxspacing = oneslice.data_element('PixelSpacing').value
            assert float(yxspacing[0])==float(yxspacing[1]), 'Error: non-square pixels: yxspacing[0] not equal to yxspacing[1]' #i.e. verify that they are square pixels
            gathered_spacing.append(yxspacing[0])
            
            #Check the orientation
            #example: ['1.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000']
            orient = [float(x) for x in oneslice.data_element('ImageOrientationPatient').value]
            assert orient==[1.0,0.0,0.0,0.0,1.0,0.0], 'Error: nonstandard ImageOrientationPatient'
            
            #Save the gantry tilt. example: "0.000000"
            gathered_gantry_tilt.append(oneslice.data_element('GantryDetectorTilt').value)
        
        #Make sure the values for all the slices are the same
        #Slopes and intercepts:
        assert len(set(gathered_slopes)) == 1, 'Error: more than one slope'
        assert len(set(gathered_inters)) == 1, 'Error: more than one intercept'
        logdf.at[note_acc,'orig_slope'] = list(set(gathered_slopes))[0]
        logdf.at[note_acc,'orig_inter'] = list(set(gathered_inters))[0]
        
        #yxspacing
        assert len(set(gathered_spacing)) == 1, 'Error: more than one yxspacing'
        logdf.at[note_acc,'orig_yxspacing'] = list(set(gathered_spacing))[0]
        
        #orientations
        logdf.at[note_acc,'orig_orientation'] = '1,0,0,0,1,0'       
        
        #gantry tilt
        assert len(set(gathered_gantry_tilt))==1, 'Error: more than one gantry tilt'
        gtilt = list(set(gathered_gantry_tilt))[0]
        if float(gtilt)!=0: print('gantry tilt nonzero:',gtilt)
        logdf.at[note_acc,'orig_gantry_tilt'] = gtilt
        
        return logdf
    
    @staticmethod
    def create_volume(raw, note_acc, logdf):
        """Concatenate the slices in the correct order and return a 3D numpy
        array. Also rescale using the slope and intercept."""
        #According to this website https://itk.org/pipermail/insight-users/2003-September/004762.html
        #the only reliable way to order slices is to use ImageOrientationPatient
        #and ImagePositionPatient. You can't even trust Image Number because
        #for some scanners it doesn't work.
        #Just for reference, you access ImagePositionPatient like this:
        #positionpatient = oneslice.data_element('ImagePositionPatient').value
        #and the values are e.g. ['-172.100', '-177.800', '-38.940'] (strings
        #of floats).
        #We can't use this for ordering the volume:
        #slice_number = int(oneslice.data_element('InstanceNumber').value)
        #because on some CT scanners the InstanceNumber is unreliable.
        #We will use a concatenation implementation from combine_slices.py:
        #ctvol has shape [num_columns, num_rows, num_slices] i.e. square, square, slices
        
        #Fields in the logdf that we fill out:
        #'orig_zpositions_all': a list of the raw z position values in order
        #'orig_zdiff_all': a list of the z diffs in order
        #'orig_zdiff_set': a set of the unique z diff values. Should have 1 member
        #'orig_zdiff': the final zdiff value. If 'orig_zdiff_set' has one
        #    member then that is the final zdiff value. If 'orig_zdiff_set'
        #    has more than one member then the final zdiff value is the mode.
        #'zdiffs_all_equal': True if the orig_zdiff_set has only one member.
        #    False otherwise.
        ctvol, slice_positions, transform = combine_slices.combine_slices_func(raw,rescale=True)
        assert slice_positions == sorted(slice_positions,reverse=True), 'Error: combine_slices did not sort slice_positions correctly'
        logdf.at[note_acc,'orig_zpositions_all'] = [round(x,4) for x in slice_positions]
        
        #figure out the z spacing by taking the difference in the z positions
        #of adjacent slices. Z spacing should be consistent throughout the
        #entire volume:
        zdiffs = [abs(round(x,4)) for x in np.ediff1d(slice_positions)] #round so you don't get floating point arithmetic problems. abs because distances can't be negative.
        logdf.at[note_acc,'orig_zdiff_all'] = zdiffs
        logdf.at[note_acc,'orig_zdiff_set'] = list(set(zdiffs))
        if len(list(set(zdiffs))) == 1:
            logdf.at[note_acc,'orig_zdiff'] = list(set(zdiffs))[0]
            logdf.at[note_acc,'zdiffs_all_equal'] = True
        else:
            #choose the zdiff value as the mode, not as the min.
            #if you choose the min you will get warped resamplings sometimes.
            #you care about what is most frequently the zdiff;
            #it's usually around 0.625
            logdf.at[note_acc,'orig_zdiff'] = mode(list(zdiffs))
            logdf.at[note_acc,'zdiffs_all_equal'] = False
        
        #save other characteristics:
        assert ctvol.shape[0] == ctvol.shape[1], 'Error: non-square axial slices'
        logdf.at[note_acc,'orig_square'] = ctvol.shape[0]
        logdf.at[note_acc,'orig_numslices'] = ctvol.shape[2]
        logdf.at[note_acc,'transform'] = [transform]
        return ctvol, logdf
    
    @staticmethod
    def resample_volume(ctvol, original_spacing, note_acc, logdf, mode):
        """Resample the numpy array <ctvol> to [0.8,0.8,0.8] spacing and return.
        There are a lot of internal checks in this function to make sure
        all the dimensions are right, because:
        - converting a numpy array to a sitk image permutes the axes
        - we need to be sure that the original_spacing z axis is in the same
          place as the sitk image z axis
        - when we convert back to a numpy array at the end, we need to be sure
          that the z axis is once again in the place it used to be in the
          original input numpy array.
        
        If <mode>=='testing' then print more output.
        Modified from https://medium.com/tensorflow/an-introduction-to-biomedical-image-analysis-with-tensorflow-and-dltk-2c25304e7c13
        """
        if mode=='testing': print('ctvol before resampling',ctvol.shape) #e.g. [512, 512, 518]
        assert ctvol.shape[0]==ctvol.shape[1], 'Error in resample_volume: non-square axial slices in input ctvol'
        ctvol_itk = sitk.GetImageFromArray(ctvol)
        ctvol_itk.SetSpacing(original_spacing)
        original_size = ctvol_itk.GetSize()
        if mode=='testing': print('ctvol original shape after sitk conversion:',original_size) #e.g. [518, 512, 512]
        if mode=='testing': print('ctvol original spacing:',original_spacing) #e.g. [0.6, 0.732421875, 0.732421875]
        
        #Double check that the square positions (x and y) are in slots 1 and 2
        #for both the original size and the original spacing:
        #(which means that the z axis, or slices, is in position 0)
        assert original_size[1]==original_size[2], 'Error in resample_volume: non-square axial slices in the original_size'
        assert original_spacing[1]==original_spacing[2], 'Error in resample_volume: non-square pixels in the original_spacing'
        
        #Calculate out shape:
        out_spacing=[0.8, 0.8, 0.8]
        #Relationship: (origshape x origspacing) = (outshape x outspacing)
        #in other words, we want to be sure we are still representing
        #the same real-world lengths in each direction.
        out_shape = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
        if mode=='testing': print('desired out shape:',out_shape) #e.g. [388, 469, 469]
        
        #Perform resampling:
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_shape)
        resample.SetOutputDirection(ctvol_itk.GetDirection())
        resample.SetOutputOrigin(ctvol_itk.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkBSpline)
        resampled_ctvol = resample.Execute(ctvol_itk)
        if mode=='testing': print('actual out shape in sitk:',resampled_ctvol.GetSize()) #e.g. [388, 469, 469]
        assert [x for x in resampled_ctvol.GetSize()]==out_shape, 'Error in resample_volume: incorrect sitk resampling shape obtained' #make sure we got the shape we wanted
        assert out_shape[1]==out_shape[2], 'Error in resample_volume: non-square sitk axial slices after resampling ' #make sure square is in slots 1 and 2
        
        #Get numpy array. Note that in the transformation from a Simple ITK
        #image to a numpy array, the axes are permuted (1,2,0) so that
        #we have the z axis (slices) at the end again (in position 2)
        #In other words the z axis gets moved from position 0 (in sitk)
        #to position 2 (in numpy)
        final_result = sitk.GetArrayFromImage(resampled_ctvol)
        if mode=='testing': print('actual out shape in numpy:',final_result.shape) #e.g. [469, 469, 388]
        assert [x for x in final_result.shape] == [out_shape[1], out_shape[2], out_shape[0]], 'Error in resample_volume: incorrect numpy resampling shape obtained'
        #we're back to having the z axis (slices) in position 2 just like
        #they were in the original ctvol: [square, square, slices]
        assert final_result.shape[0]==final_result.shape[1], 'Error in resample_volume: non-square numpy axial slices after resampling'
        
        #Update the logdf
        logdf.at[note_acc,'final_square'] = final_result.shape[0]
        logdf.at[note_acc,'final_numslices'] = final_result.shape[2]
        logdf.at[note_acc,'final_spacing']=0.8
        return final_result, logdf
    
    @staticmethod
    def represent_volume_efficiently_and_transpose(ctvol):
        """Clip the Hounsfield units and cast from float32 to int16 to
        dramatically reduce the amount of space it will take to store a
        compressed version of the <ctvol>. Also transpose.
        
        We don't care about the Hounsfield units less than -1000 (the HU of air)
        and we don't care about values higher than +1000 (bone).
        Quote, "The CT Hounsfield scale places water density at a value of
        zero with air and bone at opposite extreme values of -1000HU
        and +1000HU."
        From https://www.sciencedirect.com/topics/medicine-and-dentistry/hounsfield-scale
        By clipping the values we dramatically reduce the size of the
        compressed ctvol.
        
        It is a waste of precious space to save the CT volumes as float32
        (the default type.) Even float16 results in a final dataset size too big
        to fit in the 3.5 TB hard drive and HUs are supposed to be ints anyway
        (we only get floats due to resampling step.)
        Thus represent pixel values using int16."""
        #Clip and transpose:
        ctvol = np.clip(ctvol, a_min = -1000,a_max = 1000)
        ctvol = np.transpose(ctvol, axes=[2,1,0]) #so we get slices, square, square
        #Round and cast to integer:
        ctvol_int = np.rint(ctvol) #round each element to the nearest integer
        ctvol_int = ctvol_int.astype(np.int16) #cast to int16 data type
        assert np.isfinite(ctvol_int).all(), 'Error: infinite values created when casting to integer' #check that no np.infs created in rounding/casting
        assert np.amax(np.absolute(ctvol - ctvol_int)) < 1, 'Error: difference from original is too great when casting to integer' #check that no element is off by more than 1 HU from the original
        assert isinstance(ctvol_int[0,0,0],np.int16), 'Error: casting to int16 failed'
        return ctvol_int
    
    #################
    # Visualization #-----------------------------------------------------------
    #################
    def visualize(self, ctvol, note_acc, descriptor):
        """Visualize a CT volume"""
        outprefix = os.path.join(self.logdir,note_acc+descriptor)
        
        visualize_volumes.plot_hu_histogram(ctvol, outprefix)
        print('finished plotting histogram')
        
        #Uncomment the next lines to plot the 3D skeleton (this step is slow)
        #visualize_volumes.plot_3d_skeleton(ctvol,'HU',outprefix)
        #print('finished plotting 3d skeleton')
        
        gifpath = os.path.join(self.logdir,'gifs')
        if not os.path.exists(gifpath):
            os.mkdir(gifpath)
        visualize_volumes.make_gifs(ctvol,os.path.join(gifpath,note_acc+descriptor),chosen_views=['axial','coronal','sagittal'])
        print('finished making gifs')
        
        np.save(outprefix+'.npy',ctvol)
        print('finished saving npy file')

        #combine_slices.py
#modified from: https://github.com/innolitics/dicom-numpy

import logging
import numpy as np

logger = logging.getLogger(__name__)

class DicomImportException(Exception):
    pass

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    '''
    This function is implemented in Python 3.

    To support Python 2, we include our own implementation.
    '''
    return abs(a-b) <= max(rel_tol*max(abs(a), abs(b)), abs_tol)

def combine_slices_func(slice_datasets, rescale=None):
    '''
    Given a list of pydicom datasets for an image series, stitch them together into a
    three-dimensional numpy array.  Also calculate a 4x4 affine transformation
    matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
    DICOM patient's coordinate system.

    Returns a two-tuple containing the 3D-ndarray and the affine matrix.

    If `rescale` is set to `None` (the default), then the image array dtype
    will be preserved, unless any of the DICOM images contain either the
    `Rescale Slope
    <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281053>`_ or the
    `Rescale Intercept <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281052>`_
    attributes.  If either of these attributes are present, they will be
    applied to each slice individually.

    If `rescale` is `True` the voxels will be cast to `float32`, if set to
    `False`, the original dtype will be preserved even if DICOM rescaling information is present.

    The returned array has the column-major byte-order.

    This function requires that the datasets:

    - Be in same series (have the same
      `Series Instance UID <https://dicom.innolitics.com/ciods/ct-image/general-series/0020000e>`_,
      `Modality <https://dicom.innolitics.com/ciods/ct-image/general-series/00080060>`_,
      and `SOP Class UID <https://dicom.innolitics.com/ciods/ct-image/sop-common/00080016>`_).
    - The binary storage of each slice must be the same (have the same
      `Bits Allocated <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280100>`_,
      `Bits Stored <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280101>`_,
      `High Bit <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280102>`_, and
      `Pixel Representation <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280103>`_).
    - The image slice must approximately form a grid. This means there can not
      be any missing internal slices (missing slices on the ends of the dataset
      are not detected).
    - It also means that  each slice must have the same
      `Rows <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280010>`_,
      `Columns <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280011>`_,
      `Pixel Spacing <https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030>`_, and
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute values.
    - The direction cosines derived from the
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute must, within 1e-4, have a magnitude of 1.  The cosines must
      also be approximately perpendicular (their dot-product must be within
      1e-4 of 0).  Warnings are displayed if any of these approximations are
      below 1e-8, however, since we have seen real datasets with values up to
      1e-4, we let them pass.
    - The `Image Position (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032>`_
      values must approximately form a line.

    If any of these conditions are not met, a `dicom_numpy.DicomImportException` is raised.
    '''
    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one DICOM dataset")

    _validate_slices_form_uniform_grid(slice_datasets)

    voxels, slice_positions = _merge_slice_pixel_arrays(slice_datasets, rescale)
    transform = _ijk_to_patient_xyz_transform_matrix(slice_datasets)
    
    return voxels, slice_positions, transform


def _merge_slice_pixel_arrays(slice_datasets, rescale=None):
    first_dataset = slice_datasets[0]
    num_rows = first_dataset.Rows
    num_columns = first_dataset.Columns
    num_slices = len(slice_datasets)
    
    pos_and_data = _sort_by_slice_z_positions(slice_datasets)
    sorted_slice_datasets = [data for (pos, data) in pos_and_data]
    slice_positions = [pos for (pos, data) in pos_and_data]

    if rescale is None:
        rescale = any(_requires_rescaling(d) for d in sorted_slice_datasets)

    if rescale:
        voxels = np.empty((num_columns, num_rows, num_slices), dtype=np.float32, order='F')
        for k, dataset in enumerate(sorted_slice_datasets):
            slope = float(getattr(dataset, 'RescaleSlope', 1))
            intercept = float(getattr(dataset, 'RescaleIntercept', 0))
            voxels[:, :, k] = dataset.pixel_array.T.astype(np.float32) * slope + intercept
    else:
        dtype = first_dataset.pixel_array.dtype
        voxels = np.empty((num_columns, num_rows, num_slices), dtype=dtype, order='F')
        for k, dataset in enumerate(sorted_slice_datasets):
            voxels[:, :, k] = dataset.pixel_array.T

    return voxels, slice_positions


def _requires_rescaling(dataset):
    return hasattr(dataset, 'RescaleSlope') or hasattr(dataset, 'RescaleIntercept')


def _ijk_to_patient_xyz_transform_matrix(slice_datasets):
    pos_and_data = _sort_by_slice_z_positions(slice_datasets)
    first_dataset = [data for (pos, data) in pos_and_data][0]
    
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(slice_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(slice_datasets):
    '''
    Perform various data checks to ensure that the list of slices form a
    evenly-spaced grid of data.
    Some of these checks are probably not required if the data follows the
    DICOM specification, however it seems pertinent to check anyway.
    '''
    invariant_properties = [
        'Modality',
        'SOPClassUID',
        'SeriesInstanceUID',
        'Rows',
        'Columns',
        'PixelSpacing',
        'PixelRepresentation',
        'BitsAllocated',
        'BitsStored',
        'HighBit',
    ]

    for property_name in invariant_properties:
        _slice_attribute_equal(slice_datasets, property_name)

    _validate_image_orientation(slice_datasets[0].ImageOrientationPatient)
    _slice_ndarray_attribute_almost_equal(slice_datasets, 'ImageOrientationPatient', 1e-5)

    slice_positions = _slice_positions(slice_datasets)
    _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation):
    '''
    Ensure that the image orientation is supported
    - The direction cosines have magnitudes of 1 (just in case)
    - The direction cosines are perpendicular
    '''
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    if not _almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
        raise DicomImportException("Non-orthogonal direction cosines: {}, {}".format(row_cosine, column_cosine))
    elif not _almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
        logger.warning("Direction cosines aren't quite orthogonal: {}, {}".format(row_cosine, column_cosine))

    if not _almost_one(np.linalg.norm(row_cosine), 1e-4):
        raise DicomImportException("The row direction cosine's magnitude is not 1: {}".format(row_cosine))
    elif not _almost_one(np.linalg.norm(row_cosine), 1e-8):
        logger.warning("The row direction cosine's magnitude is not quite 1: {}".format(row_cosine))

    if not _almost_one(np.linalg.norm(column_cosine), 1e-4):
        raise DicomImportException("The column direction cosine's magnitude is not 1: {}".format(column_cosine))
    elif not _almost_one(np.linalg.norm(column_cosine), 1e-8):
        logger.warning("The column direction cosine's magnitude is not quite 1: {}".format(column_cosine))


def _almost_zero(value, abs_tol):
    return isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value, abs_tol):
    return isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(slice_datasets, property_name):
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = 'All slices must have the same value for "{}": {} != {}'
            raise DicomImportException(msg.format(property_name, value, initial_value))


def _slice_ndarray_attribute_almost_equal(slice_datasets, property_name, abs_tol):
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if not np.allclose(value, initial_value, atol=abs_tol):
            msg = 'All slices must have the same value for "{}" within "{}": {} != {}'
            raise DicomImportException(msg.format(property_name, abs_tol, value, initial_value))


def _slice_positions(slice_datasets):
    image_orientation = slice_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]


def _check_for_missing_slices(slice_positions):
    slice_positions_diffs = np.diff(sorted(slice_positions))
    if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
        print('\tWarning: combine_slices detects that there may be missing slices')
        
        
def _slice_spacing(slice_datasets):
    if len(slice_datasets) > 1:
        slice_positions = _slice_positions(slice_datasets)
        slice_positions_diffs = np.diff(sorted(slice_positions))
        return np.mean(slice_positions_diffs)
    else:
        return 0.0


def _sort_by_slice_z_positions(slice_datasets):
    """Example:
    >>> slice_spacing = [1.1,7,2.2,100]
    >>> slice_datasets=['a','b','c','d']
    >>> sorted(zip(slice_spacing, slice_datasets), reverse=True)
    [(100, 'd'), (7, 'b'), (2.2, 'c'), (1.1, 'a')]
    >>> [d for (s, d) in sorted(zip(slice_spacing, slice_datasets), reverse=True)]
    ['d', 'b', 'c', 'a']
    """
    slice_positions = _slice_positions(slice_datasets)
    return sorted(zip(slice_positions, slice_datasets), reverse=True)
    
    #test_combine_slices.py
#modified from https://github.com/innolitics/dicom-numpy

import numpy as np

from combine_slices import combine_slices_func, _validate_slices_form_uniform_grid, _merge_slice_pixel_arrays, DicomImportException

# direction cosines
x_cos = (1, 0, 0)
y_cos = (0, 1, 0)
z_cos = (0, 0, 1)
negative_x_cos = (-1, 0, 0)
negative_y_cos = (0, -1, 0)
negative_z_cos = (0, 0, -1)

arbitrary_shape = (10, 11)

class MockSlice:
    '''
    A minimal DICOM dataset representing a dataslice at a particular
    slice location.  The `slice_position` is the coordinate value along the
    remaining unused axis (i.e. the axis perpendicular to the direction
    cosines).
    '''

    def __init__(self, pixel_array, slice_position, row_cosine=None, column_cosine=None, **kwargs):
        if row_cosine is None:
            row_cosine = x_cos

        if column_cosine is None:
            column_cosine = y_cos

        na, nb = pixel_array.shape

        self.pixel_array = pixel_array

        self.SeriesInstanceUID = 'arbitrary uid'
        self.SOPClassUID = 'arbitrary sopclass uid'
        self.PixelSpacing = [1.0, 1.0]
        self.Rows = na
        self.Columns = nb
        self.Modality = 'MR'

        # assume that the images are centered on the remaining unused axis
        a_component = [-na/2.0*c for c in row_cosine]
        b_component = [-nb/2.0*c for c in column_cosine]
        c_component = [(slice_position if c == 0 and cc == 0 else 0) for c, cc in zip(row_cosine, column_cosine)]
        patient_position = [a + b + c for a, b, c in zip(a_component, b_component, c_component)]

        self.ImagePositionPatient = patient_position

        self.ImageOrientationPatient = list(row_cosine) + list(column_cosine)

        for k, v in kwargs.items():
            setattr(self, k, v)

def axial_slices():
    return [MockSlice(randi(*arbitrary_shape), 0),
        MockSlice(randi(*arbitrary_shape), 1),
        MockSlice(randi(*arbitrary_shape), 2),
        MockSlice(randi(*arbitrary_shape), 3),]

def randi(*shape):
    return np.random.randint(1000, size=shape, dtype='uint16')

class TestCombineSlices:
    def __init__(self):
        self.test_simple_axial_set(axial_slices())
    
    def test_simple_axial_set(self, axial_slices):
        combined, _, _ = combine_slices_func(axial_slices[0:2])

        manually_combined = np.dstack((axial_slices[1].pixel_array.T, axial_slices[0].pixel_array.T))
        assert np.array_equal(combined, manually_combined)
        print('Passed test_simple_axial_set()')


class TestMergeSlicePixelArrays:
    def __init__(self):
        self.test_casting_if_only_rescale_slope()
        self.test_casting_rescale_slope_and_intercept()
        self.test_robust_to_ordering(axial_slices())
        self.test_rescales_if_forced_true()
        self.test_no_rescale_if_forced_false()
    
    def test_casting_if_only_rescale_slope(self):
        '''
        If the `RescaleSlope` DICOM attribute is present, the
        `RescaleIntercept` attribute should also be present, however, we handle
        this case anyway.
        '''
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2),
        ]

        voxels, _ = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 2.0
        print('Passed test_casting_if_only_rescale_slope()')

    def test_casting_rescale_slope_and_intercept(self):
        '''
        Some DICOM modules contain the `RescaleSlope` and `RescaleIntercept` DICOM attributes.
        '''
        slices = [
            MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3),
            MockSlice(np.ones((10, 20), dtype=np.uint8), 1, RescaleSlope=2, RescaleIntercept=3),
        ]

        voxels, _ = _merge_slice_pixel_arrays(slices)
        assert voxels.dtype == np.dtype('float32')
        assert voxels[0, 0, 0] == 5.0
        print('Passed test_casting_rescale_slope_and_intercept()')

    def test_robust_to_ordering(self, axial_slices):
        '''
        The DICOM slices should be able to be passed in in any order, and they
        should be recombined appropriately.
        '''
        a, _ = _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]])
        b, _ = _merge_slice_pixel_arrays([axial_slices[1], axial_slices[0], axial_slices[2]])
        assert np.array_equal(a,b)

        c, _ = _merge_slice_pixel_arrays([axial_slices[0], axial_slices[1], axial_slices[2]])
        d, _ = _merge_slice_pixel_arrays([axial_slices[2], axial_slices[0], axial_slices[1]])
        assert np.array_equal(c,d)
        print('Passed test_robust_to_ordering()')

    def test_rescales_if_forced_true(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0)]
        voxels, _ = _merge_slice_pixel_arrays(slice_datasets, rescale=True)
        assert voxels.dtype == np.float32
        print('Passed test_rescales_if_forced_true')

    def test_no_rescale_if_forced_false(self):
        slice_datasets = [MockSlice(np.ones((10, 20), dtype=np.uint8), 0, RescaleSlope=2, RescaleIntercept=3)]
        voxels, _ = _merge_slice_pixel_arrays(slice_datasets, rescale=False)
        assert voxels.dtype == np.uint8
        print('Passed test_no_rescale_if_forced_false()')


class TestValidateSlicesFormUniformGrid:
    def __init__(self):
        #self.test_missing_middle_slice(axial_slices())
        self.test_insignificant_difference_in_direction_cosines(axial_slices())
        self.test_significant_difference_in_direction_cosines(axial_slices())
        self.test_slices_from_different_series(axial_slices())
    
    # def test_missing_middle_slice(self, axial_slices):
    #     '''
    #     All slices must be present.  Slice position is determined using the
    #     ImagePositionPatient (0020,0032) tag.
    #     '''
    #     #Pass the test only if this raises DicomImportException
    #     try:
    #         _validate_slices_form_uniform_grid([axial_slices[0], axial_slices[2], axial_slices[3]])
    #     except DicomImportException:
    #         print('Passed test_missing_middle_slice()')
    #         return
    #     raise Exception

    def test_insignificant_difference_in_direction_cosines(self, axial_slices):
        '''
        We have seen DICOM series in the field where slices have lightly
        different direction cosines.
        '''
        axial_slices[0].ImageOrientationPatient[0] += 1e-6
        _validate_slices_form_uniform_grid(axial_slices)
        print('Passed test_insignificant_difference_in_direction_cosines()')

    def test_significant_difference_in_direction_cosines(self, axial_slices):
        #Pass the test only if this raises DicomImportException
        axial_slices[0].ImageOrientationPatient[0] += 1e-4
        try: 
            _validate_slices_form_uniform_grid(axial_slices)
        except DicomImportException:
            print('Passed test_significant_difference_in_direction_cosines()')
            return
        raise Exception

    def test_slices_from_different_series(self, axial_slices):
        '''
        As a sanity check, slices that don't come from the same DICOM series should
        be rejected.
        '''
        #Pass the test only if this raises DicomImportException
        axial_slices[2].SeriesInstanceUID += 'Ooops'
        try:
            _validate_slices_form_uniform_grid(axial_slices)
        except DicomImportException:
            print('Passed test_slices_from_different_series()')
            return
        raise Exception

if __name__ == '__main__':
    TestCombineSlices()
    TestMergeSlicePixelArrays()
    TestValidateSlicesFormUniformGrid()
    
    #visualize_volumes.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

from PIL import Image
import SimpleITK as sitk

import os
import imageio
import timeit
import datetime
import operator
import numpy as np
import pandas as pd
from numpy.linalg import inv, det, norm
from math import sqrt, pi
from functools import partial
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

import ipywidgets as ipyw

class ImageSliceViewer3D:
    #Copied from https://github.com/mohakpatel/ImageSliceViewer3D
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image (3D numpy array)
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    """
    
    def __init__(self, volume, figsize=(8,8)):
        self.volume = volume
        self.figsize = figsize
        self.cmap = plt.cm.gray #alternative: cmap='plasma'
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
                        options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])

def plot_hu_histogram(ctvol, outprefix):
    #Histogram of HUs https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    plt.hist(ctvol.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.savefig(outprefix+'_HU_Hist.png')
    plt.close()

def plot_middle_slice(ctvol, outprefix):
    # Show some slice in the middle
    plt.imshow(ctvol[100,:,:], cmap=plt.cm.gray)
    plt.savefig(outprefix+'_Middle_Slice.png')
    plt.close()

def plot_3d_skeleton(ctvol, units, outprefix):
    """Make a 3D plot of the skeleton.
    <units> either 'HU' or 'processed' (normalized) determines the thresholds"""
    #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    #Get high threshold to show mostly bones
    if units == 'HU':
        threshold = 400
    elif units == 'processed':
        threshold = 0.99
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = ctvol.transpose(2,1,0)
    p = np.flip(p, axis = 0) #need this line or else the patient is upside-down
    
    #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, _ignore1, _ignore2 = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.savefig(outprefix+'_3D_Bones.png')
    plt.close()    

def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].
    
    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    #First fix the grayscale colors.
    #imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    #If you do not truncate to a 256 range, imageio will do so on a per-slice
    #basis, which creates weird brightening and darkening artefacts in the gif.
    #Thus, before making the gif, you should truncate to the 0-256 range
    #and cast to a uint8 (the dtype imageio requires):
    #how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (  ((ctvol+800)*(255))/(400+800)  ).astype('uint8')
    
    #Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[0]):
            images.append(ctvol[slicenum,:,:])
        imageio.mimsave(outprefix+'_axial.gif',images)
        print('\t\tdone with axial gif')
    
    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:,slicenum,:])
        imageio.mimsave(outprefix+'_coronal.gif',images)
        print('\t\tdone with coronal gif')
    
    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:,:,slicenum])
        imageio.mimsave(outprefix+'_sagittal.gif',images)
        print('\t\tdone with sagittal gif')