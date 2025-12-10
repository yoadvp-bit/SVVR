"""
Fully Automatic Body Assembly System
Uses anatomical feature detection and image analysis to automatically:
1. Identify what body part each series contains
2. Determine the correct order for assembly
3. Find optimal stitching points
4. Validate the result
"""

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import pydicom
import os
import math
from collections import defaultdict
from scipy.ndimage import center_of_mass, binary_erosion

# --- CONFIGURATION ---
# Simply change this to process different datasets:
# Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"
DATASET = "DICOM-Maria"

base_path = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
target_folder = os.path.join(base_path, DATASET)

def detect_anatomical_features(volume_array):
    """
    Analyze a 3D volume to detect anatomical features
    Returns feature dictionary with detected body parts
    """
    # Volume is (Z, Y, X) where Z is slice index
    features = {
        'has_head': False,
        'has_arms': False,
        'has_legs': False,
        'body_width_profile': [],
        'tissue_density_profile': [],
        'separation_detected': False
    }
    
    # Analyze each slice
    num_slices = volume_array.shape[0]
    
    for z in range(0, num_slices, max(1, num_slices // 50)):  # Sample slices
        slice_2d = volume_array[z, :, :]
        
        # Threshold to get body mask
        body_mask = (slice_2d > -500) & (slice_2d < 2000)
        
        if np.sum(body_mask) < 100:  # Skip empty slices
            features['body_width_profile'].append(0)
            features['tissue_density_profile'].append(0)
            continue
        
        # Measure horizontal extent (width)
        horizontal_proj = np.sum(body_mask, axis=0)
        body_width = np.sum(horizontal_proj > 0)
        features['body_width_profile'].append(body_width)
        
        # Measure tissue density
        tissue_pixels = np.sum(body_mask)
        features['tissue_density_profile'].append(tissue_pixels)
    
    # Analyze profile for features
    if len(features['body_width_profile']) > 10:
        width_arr = np.array(features['body_width_profile'])
        density_arr = np.array(features['tissue_density_profile'])
        
        # Head detection: Top slices have circular/compact structure
        # Arms detection: Wide horizontal span in middle sections
        # Legs detection: Two separated vertical structures in lower sections
        
        # Analyze top 20% of volume
        top_20_percent = int(len(width_arr) * 0.2)
        if top_20_percent > 0:
            top_density = np.mean(density_arr[:top_20_percent])
            
            # Head typically has high density, compact structure
            if top_density > np.mean(density_arr) * 1.2:
                features['has_head'] = True
        
        # Analyze width variation for arms
        if np.std(width_arr) > np.mean(width_arr) * 0.3:
            features['has_arms'] = True
        
        # Check for leg separation: analyze bottom slices
        bottom_20_percent = int(len(width_arr) * 0.2)
        if bottom_20_percent > 0:
            # Sample a slice from bottom region
            sample_z = int(num_slices * 0.8)
            if sample_z < num_slices:
                bottom_slice = volume_array[sample_z, :, :]
                body_mask = (bottom_slice > -500) & (bottom_slice < 2000)
                
                # Check for vertical separation (legs)
                vertical_proj = np.sum(body_mask, axis=1)
                if np.sum(vertical_proj > 0) > 0:
                    # Look for gap in horizontal projection
                    horizontal_proj = np.sum(body_mask, axis=0)
                    if len(horizontal_proj) > 10:
                        # Simple gap detection
                        mid_point = len(horizontal_proj) // 2
                        left_half = horizontal_proj[:mid_point]
                        right_half = horizontal_proj[mid_point:]
                        
                        if np.sum(left_half > 0) > 10 and np.sum(right_half > 0) > 10:
                            # Check for separation
                            mid_region = horizontal_proj[mid_point-10:mid_point+10]
                            if np.min(mid_region) < np.max(horizontal_proj) * 0.3:
                                features['has_legs'] = True
                                features['separation_detected'] = True
    
    return features

def classify_body_region_automatic(series_info, volume_array):
    """
    Automatically classify a series as HEAD/TORSO/TORSO_UPPER/PELVIS/LEGS
    based on anatomical features and Z-position
    """
    features = detect_anatomical_features(volume_array)
    z_center = series_info['z_center']
    z_span = series_info['span']
    
    # Decision logic based on features and position
    if features['has_head']:
        return 'TORSO_WITH_HEAD', 1000
    elif features['has_legs'] or features['separation_detected']:
        return 'LEGS_PELVIS', -500
    elif z_center > 200 and z_span > 500:
        # High Z position, large span = likely torso
        return 'TORSO_UPPER', 500
    elif z_center > 0 and z_span > 300:
        return 'TORSO_MID', 200
    elif z_center < -300:
        return 'PELVIS_LOWER', -400
    else:
        return 'TORSO_MID', 0

def scan_and_classify_all_series(root_path):
    """Scan all series and automatically classify them"""
    print(f"\n{'='*80}")
    print(f"AUTOMATIC BODY ASSEMBLY: {os.path.basename(root_path)}")
    print(f"{'='*80}\n")
    
    print("Step 1: Scanning DICOM files...")
    series_map = defaultdict(list)
    series_metadata = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Check if axial
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if not np.allclose(vec_z, [0, 0, 1], atol=0.3): 
                        continue
                
                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'spacing': getattr(dcm, 'PixelSpacing', [1.0, 1.0]),
                        'slice_thickness': getattr(dcm, 'SliceThickness', 5.0)
                    }
            except:
                continue
    
    print(f"  Found {len(series_map)} series candidates\n")
    
    print("Step 2: Loading and analyzing series content...")
    all_series = []
    
    for uid, data in series_map.items():
        if len(data) < 20: continue
        
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 100: continue
        
        series_info = {
            'uid': uid,
            'desc': series_metadata[uid]['desc'],
            'paths': [x[1] for x in data],
            'z_min': z_min,
            'z_max': z_max,
            'z_center': (z_min + z_max) / 2,
            'num_slices': len(data),
            'span': span,
            'spacing': series_metadata[uid]['spacing'],
            'slice_thickness': series_metadata[uid]['slice_thickness']
        }
        
        # Load volume for analysis
        try:
            print(f"  Analyzing: {series_info['desc'][:40]}...")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(series_info['paths'])
            img = reader.Execute()
            volume_arr = sitk.GetArrayFromImage(img)
            
            # Classify based on content
            region, priority = classify_body_region_automatic(series_info, volume_arr)
            series_info['region'] = region
            series_info['priority'] = priority
            series_info['image'] = img  # Store for later use
            
            print(f"    → Classified as: {region}")
            
            all_series.append(series_info)
        except Exception as e:
            print(f"    → Skipped (error: {e})")
            continue
    
    print(f"\n  Successfully analyzed {len(all_series)} series\n")
    
    # Remove duplicates (same Z-range and region)
    unique_series = []
    seen = set()
    for s in all_series:
        key = (round(s['z_min'], 0), round(s['z_max'], 0), s['region'])
        if key not in seen:
            unique_series.append(s)
            seen.add(key)
    
    print(f"Step 3: Series classification summary:")
    print(f"{'REGION':<20} | {'Z-RANGE (mm)':<25} | {'SLICES':<7} | {'DESCRIPTION'}")
    print(f"{'-'*85}")
    for s in sorted(unique_series, key=lambda x: x['priority'], reverse=True):
        z_range = f"{s['z_min']:.1f} to {s['z_max']:.1f}"
        print(f"{s['region']:<20} | {z_range:<25} | {s['num_slices']:<7} | {s['desc'][:30]}")
    
    return unique_series

def find_best_stitching_pair(all_series):
    """Automatically find the best two series to stitch together"""
    print(f"\nStep 4: Finding optimal series pair for stitching...")
    
    # Sort by anatomical priority (head to feet)
    sorted_series = sorted(all_series, key=lambda x: x['priority'], reverse=True)
    
    best_pair = None
    best_score = -1
    
    # Try all combinations
    for i in range(len(sorted_series)):
        for j in range(i+1, len(sorted_series)):
            upper = sorted_series[i]
            lower = sorted_series[j]
            
            # Calculate interface quality
            gap = upper['z_min'] - lower['z_max']
            
            # Scoring criteria
            gap_score = 0
            if -150 < gap < 150:  # Reasonable gap/overlap
                gap_score = 1.0 / (1.0 + abs(gap) / 100.0)
            
            # Priority score (prefer connecting different regions)
            priority_diff = abs(upper['priority'] - lower['priority'])
            priority_score = min(priority_diff / 500.0, 1.0)
            
            # Size score (prefer larger volumes)
            size_score = min(upper['span'], lower['span']) / 1000.0
            
            total_score = gap_score * 2.0 + priority_score + size_score
            
            if total_score > best_score:
                best_score = total_score
                best_pair = (upper, lower)
    
    if best_pair:
        upper, lower = best_pair
        gap = upper['z_min'] - lower['z_max']
        print(f"  ✓ Selected pair:")
        print(f"    UPPER: {upper['region']:<20} Z: {upper['z_min']:.1f} to {upper['z_max']:.1f}")
        print(f"    LOWER: {lower['region']:<20} Z: {lower['z_min']:.1f} to {lower['z_max']:.1f}")
        print(f"    Interface gap: {gap:.1f}mm")
        print(f"    Confidence score: {best_score:.2f}/4.0")
        return upper, lower
    
    return None, None

def perform_intelligent_stitch(upper, lower):
    """Perform stitching with intelligent overlap detection"""
    print(f"\nStep 5: Performing intelligent stitching...")
    
    upper_img = upper['image']
    lower_img = lower['image']
    
    # XY Alignment using center of mass
    print("  → Aligning XY centers...")
    def get_interface_com(img, is_bottom):
        size = img.GetSize()
        if is_bottom:
            slice_idx = size[2] - 1
        else:
            slice_idx = 0
        
        slice_img = img[:, :, slice_idx]
        mask = sitk.BinaryThreshold(slice_img, lowerThreshold=-500, upperThreshold=3000)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetNumberOfLabels() > 0:
            return stats.GetCentroid(1)[:2]
        return img.GetOrigin()[:2]
    
    com_upper = get_interface_com(upper_img, True)
    com_lower = get_interface_com(lower_img, False)
    
    dx = com_upper[0] - com_lower[0]
    dy = com_upper[1] - com_lower[1]
    print(f"    XY shift: ({dx:.1f}, {dy:.1f}) mm")
    
    # Resample lower to match upper's XY position
    transform = sitk.TranslationTransform(3)
    transform.SetOffset((-dx, -dy, 0))
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(upper_img)
    resampler.SetOutputOrigin([upper_img.GetOrigin()[0], upper_img.GetOrigin()[1], lower_img.GetOrigin()[2]])
    resampler.SetSize([upper_img.GetSize()[0], upper_img.GetSize()[1], lower_img.GetSize()[2]])
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000)
    lower_img_aligned = resampler.Execute(lower_img)
    
    # Find optimal overlap using masked NCC
    print("  → Finding optimal Z-overlap...")
    upper_arr = sitk.GetArrayFromImage(upper_img)
    lower_arr = sitk.GetArrayFromImage(lower_img_aligned)
    
    def masked_similarity(a, b):
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)
        
        # Only compare tissue (ignore air)
        mask = (a_flat > -500) & (b_flat > -500)
        if np.sum(mask) < 1000:  # Need enough tissue
            return 0.0
        
        a_tissue = a_flat[mask]
        b_tissue = b_flat[mask]
        
        # Normalize
        a_tissue = a_tissue - np.mean(a_tissue)
        b_tissue = b_tissue - np.mean(b_tissue)
        
        # NCC
        num = np.sum(a_tissue * b_tissue)
        denom = np.linalg.norm(a_tissue) * np.linalg.norm(b_tissue)
        
        return (num / denom) if denom > 0 else 0.0
    
    upper_len = len(upper_arr)
    lower_len = len(lower_arr)
    max_overlap = min(upper_len // 3, lower_len // 3, 100)  # Max 1/3 overlap
    
    best_score = -1
    best_overlap = 1
    scores = []
    
    for k in range(1, max_overlap + 1):
        upper_block = upper_arr[-k:]
        lower_block = lower_arr[:k]
        
        score = masked_similarity(upper_block, lower_block)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_overlap = k
    
    # Validate: look for peak and ensure it's stable
    if len(scores) > 10:
        # Find where score peaks and stabilizes
        scores_arr = np.array(scores)
        # Use derivative to find where improvement stops
        if len(scores_arr) > 5:
            deriv = np.diff(scores_arr)
            # Find first local maximum
            for i in range(5, len(deriv)):
                if deriv[i] < 0.001 and scores_arr[i] > 0.3:
                    best_overlap = i
                    best_score = scores_arr[i]
                    break
    
    overlap_mm = best_overlap * upper['slice_thickness']
    print(f"    Optimal overlap: {best_overlap} slices ({overlap_mm:.1f}mm)")
    print(f"    Similarity score: {best_score:.4f}")
    
    # Stitch
    stitched_arr = np.concatenate((upper_arr, lower_arr[best_overlap:]), axis=0)
    stitched_img = sitk.GetImageFromArray(stitched_arr)
    stitched_img.SetOrigin(upper_img.GetOrigin())
    stitched_img.SetSpacing(upper_img.GetSpacing())
    stitched_img.SetDirection(upper_img.GetDirection())
    
    print(f"    Final volume: {stitched_arr.shape[0]} slices ({stitched_arr.shape[0] * upper['slice_thickness']:.1f}mm)")
    
    # Validation
    print(f"\nStep 6: Validating result...")
    print(f"  ✓ Anatomical continuity: {'GOOD' if best_score > 0.3 else 'POOR'}")
    print(f"  ✓ Total body coverage: {stitched_arr.shape[0] * upper['slice_thickness'] / 10:.1f} cm")
    
    return stitched_img

def render_result(sitk_img, title):
    """Render the final volume"""
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    
    dims = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims)
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(0, 0, 0)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            mat.SetElement(i, j, direction[i + j*3])
    mat.SetElement(0, 3, origin[0])
    mat.SetElement(1, 3, origin[1])
    mat.SetElement(2, 3, origin[2])
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-200, 0.0)
    opacity.AddPoint(-100, 0.1)
    opacity.AddPoint(40, 0.3)
    opacity.AddPoint(200, 0.6)
    opacity.AddPoint(500, 1.0)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-100, 0.8, 0.6, 0.5)
    color.AddRGBPoint(40, 0.8, 0.2, 0.2)
    color.AddRGBPoint(200, 1.0, 0.9, 0.9)
    color.AddRGBPoint(500, 1.0, 1.0, 1.0)
    
    volume = vtk.vtkVolume()
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    volume.SetMapper(mapper)
    volume.SetUserMatrix(mat)
    
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()
    volume.SetProperty(prop)
    
    renderer.AddVolume(volume)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    volume.Update()
    bounds = volume.GetBounds()
    if bounds:
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
        cam = renderer.GetActiveCamera()
        cam.SetFocalPoint(center)
        cam.SetPosition(center[0], center[1] - 3000, center[2])
        cam.SetViewUp(0, 0, -1)
        renderer.ResetCameraClippingRange()
    
    print(f"\nStep 7: Rendering...")
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    # Full automatic pipeline
    all_series = scan_and_classify_all_series(target_folder)
    
    if len(all_series) < 2:
        print("\n✗ ERROR: Need at least 2 series to stitch")
        exit(1)
    
    upper, lower = find_best_stitching_pair(all_series)
    
    if upper is None or lower is None:
        print("\n✗ ERROR: Could not find suitable series pair")
        exit(1)
    
    final_volume = perform_intelligent_stitch(upper, lower)
    
    render_result(final_volume, f"Automatic Assembly: {DATASET}")
    
    print(f"\n{'='*80}")
    print(f"✓ AUTOMATIC ASSEMBLY COMPLETE")
    print(f"{'='*80}\n")
