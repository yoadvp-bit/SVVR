import SimpleITK as sitk
import vtk
import os
import sys
import pydicom
import numpy as np
from vtk.util import numpy_support
from collections import defaultdict

# --- CONFIGURATION ---
target_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/DICOM-2" 

# The specific shift requested: 798 + 1027.5
Z_SHIFT_OFFSET = 798.0 + 1027.5 

def get_verified_series(root_path):
    print("1. Scanning for valid Axial volumes...")
    series_map = defaultdict(list)
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                # Fast Header Read
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Filter: Must be AXIAL
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if not np.allclose(vec_z, [0, 0, 1], atol=0.2): 
                        continue 

                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
            except: continue

    valid_series = []
    print(f"   -> Found {len(series_map)} candidates.")
    
    for uid, data in series_map.items():
        if len(data) < 20: continue
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 50.0: continue 
        
        valid_series.append({
            'uid': uid, 'paths': [x[1] for x in data], 
            'span': span, 'z_min': z_min, 'z_max': z_max
        })
    return valid_series

def remove_subsets(series_list):
    print(f"2. Deduplicating {len(series_list)} series...")
    series_list.sort(key=lambda x: x['span'], reverse=True)
    final = []
    covered = []
    
    for s in series_list:
        is_subset = False
        for c_min, c_max in covered:
            overlap = max(0, min(s['z_max'], c_max) - max(s['z_min'], c_min))
            if s['span'] > 0 and (overlap / s['span']) > 0.8: is_subset = True; break
        
        if not is_subset:
            final.append(s)
            covered.append((s['z_min'], s['z_max']))
            
    return final

def load_volume_with_translation(s_data):
    # Load Image
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(s_data['paths'])
    sitk_img = reader.Execute()
    
    origin = list(sitk_img.GetOrigin()) # Convert tuple to list to modify
    direction = sitk_img.GetDirection()
    spacing = sitk_img.GetSpacing()
    dims = sitk_img.GetSize()
    
    # --- MANUAL TRANSLATION LOGIC ---
    # Check if this series starts around -798 (Tolerance of +/- 5.0)
    if abs(s_data['z_min'] - (-798.0)) < 5.0:
        print(f"   [APPLYING FIX] Detected Leg Group (Start {s_data['z_min']}). Adding {Z_SHIFT_OFFSET} to Z.")
        print(f"   -> Old Z: {origin[2]:.2f}")
        origin[2] = origin[2] + Z_SHIFT_OFFSET
        print(f"   -> New Z: {origin[2]:.2f}")
    else:
        print(f"   [NO CHANGE] Series starts at {s_data['z_min']}. Keeping original coordinates.")
    
    # VTK Conversion
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims[0], dims[1], dims[2])
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(0,0,0)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    # Build Matrix with MODIFIED Origin
    dicom_mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3): dicom_mat.SetElement(i, j, direction[i + j*3])
    dicom_mat.SetElement(0, 3, origin[0])
    dicom_mat.SetElement(1, 3, origin[1])
    dicom_mat.SetElement(2, 3, origin[2])
    
    volume = vtk.vtkVolume()
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    volume.SetMapper(mapper)
    volume.SetUserMatrix(dicom_mat)
    
    return volume

# --- MAIN EXECUTION ---
series_data = get_verified_series(target_folder)
if not series_data: 
    print("Error: No valid series found.")
    sys.exit(1)

to_render = remove_subsets(series_data)

renderer = vtk.vtkRenderer()
renderer.SetBackground(0.05, 0.05, 0.05)

# Bone Preset
opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(-1000, 0.0)
opacity.AddPoint(100, 0.0)
opacity.AddPoint(250, 0.2)
opacity.AddPoint(1000, 0.9)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
color.AddRGBPoint(200, 0.9, 0.8, 0.7)
color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

print("3. Building 3D Scene with Manual Offsets...")
all_bounds = []

for s in to_render:
    try:
        print(f"   -> Processing ...{s['uid'][-6:]}")
        vol = load_volume_with_translation(s)
        
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        vol.SetProperty(prop)
        
        renderer.AddVolume(vol)
        vol.Update()
        all_bounds.append(vol.GetBounds())
    except Exception as e:
        print(f"Error: {e}")

# --- CAMERA SETUP ---
print("4. Positioning Camera...")
render_window = vtk.vtkRenderWindow()
render_window.SetSize(1200, 1200)
render_window.SetWindowName("Manual Translation Fix")
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

if all_bounds:
    # 1. Find Center
    min_x = min(b[0] for b in all_bounds); max_x = max(b[1] for b in all_bounds)
    min_y = min(b[2] for b in all_bounds); max_y = max(b[3] for b in all_bounds)
    min_z = min(b[4] for b in all_bounds); max_z = max(b[5] for b in all_bounds)
    center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
    
    cam = renderer.GetActiveCamera()
    cam.SetFocalPoint(center)
    
    # Position: Front View (-Y)
    cam.SetPosition(center[0], center[1] - 3500, center[2])
    
    # View Up: Head at Top (-Z is Up)
    cam.SetViewUp(0, 0, -1)
    
    renderer.ResetCameraClippingRange()

print("   -> Rendering...")
render_window.Render()
interactor.Start()