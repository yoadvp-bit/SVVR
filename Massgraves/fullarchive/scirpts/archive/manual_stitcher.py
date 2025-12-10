"""
Interactive DICOM Series Selector and Stitcher
This tool helps you:
1. View all available series with visualizations
2. Manually select which two series to stitch
3. Test the stitching with real-time feedback
"""

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import pydicom
import os
import math
from collections import defaultdict

# --- CONFIGURATION ---
DATASET = "DICOM-Maria"  # Change to test different datasets
base_path = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
target_folder = os.path.join(base_path, DATASET)

def scan_and_list_series(root_path):
    """Scan and list all viable axial series"""
    print(f"\n{'='*80}")
    print(f"SCANNING: {os.path.basename(root_path)}")
    print(f"{'='*80}\n")
    
    series_map = defaultdict(list)
    series_info = {}
    
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
                
                if uid not in series_info:
                    series_info[uid] = getattr(dcm, 'SeriesDescription', 'Unknown')
            except:
                continue
    
    valid_series = []
    for uid, data in series_map.items():
        if len(data) < 20: continue
        
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 100: continue
        
        valid_series.append({
            'uid': uid,
            'desc': series_info.get(uid, 'Unknown'),
            'paths': [x[1] for x in data],
            'z_min': z_min,
            'z_max': z_max,
            'z_center': (z_min + z_max) / 2,
            'num_slices': len(data),
            'span': span
        })
    
    # Sort by Z center (top to bottom anatomically)
    valid_series.sort(key=lambda x: x['z_center'], reverse=True)
    
    # Remove duplicates (same Z range)
    unique_series = []
    seen_ranges = set()
    for s in valid_series:
        range_key = (round(s['z_min'], 1), round(s['z_max'], 1))
        if range_key not in seen_ranges:
            unique_series.append(s)
            seen_ranges.add(range_key)
    
    print(f"Found {len(unique_series)} unique axial series:\n")
    print(f"{'ID':<4} | {'DESCRIPTION':<30} | {'Z-RANGE (mm)':<25} | {'SLICES':<7} | {'SPAN (mm)':<10}")
    print(f"{'-'*95}")
    
    for i, series in enumerate(unique_series):
        z_range = f"{series['z_min']:.1f} to {series['z_max']:.1f}"
        print(f"{i+1:<4} | {series['desc'][:30]:<30} | {z_range:<25} | {series['num_slices']:<7} | {series['span']:.1f}")
    
    print(f"\n{'='*80}\n")
    print("INSTRUCTIONS:")
    print("1. Look at the visualization PNG files created earlier")
    print(f"2. Check: {DATASET}_visualization.png")
    print("3. Identify which series contain upper body (head/torso) vs lower body (pelvis/legs)")
    print("4. Edit MANUAL_CONFIG below to specify which series to stitch")
    print(f"{'='*80}\n")
    
    return unique_series

# --- MANUAL CONFIGURATION ---
# After running once and seeing the list, configure this:
MANUAL_CONFIG = {
    "DICOM-Maria": {
        "upper_series_index": 1,  # Series ID from the list (1-indexed)
        "lower_series_index": 3,  # Series ID from the list (1-indexed)
        "expected_overlap_mm": 20,  # Expected anatomical overlap in mm
        "notes": "Upper=Torso with head, Lower=Legs/pelvis"
    },
    "DICOM-Jan": {
        "upper_series_index": 2,  # UPDATE AFTER VIEWING VISUALIZATIONS
        "lower_series_index": 4,  # UPDATE AFTER VIEWING VISUALIZATIONS  
        "expected_overlap_mm": 50,
        "notes": "VERIFY: May have multiple torso scans (with different arms)"
    },
    "DICOM-Jarek": {
        "upper_series_index": 1,  # UPDATE AFTER VIEWING VISUALIZATIONS
        "lower_series_index": 2,  # UPDATE AFTER VIEWING VISUALIZATIONS
        "expected_overlap_mm": 75,
        "notes": "VERIFY: May have separate thorax and abdomen scans"
    }
}

def get_manual_config():
    """Get the manual configuration for current dataset"""
    if DATASET in MANUAL_CONFIG:
        config = MANUAL_CONFIG[DATASET]
        print(f"Using MANUAL configuration for {DATASET}:")
        print(f"  Upper body: Series #{config['upper_series_index']}")
        print(f"  Lower body: Series #{config['lower_series_index']}")
        print(f"  Expected overlap: ~{config['expected_overlap_mm']}mm")
        print(f"  Notes: {config['notes']}\n")
        return config
    return None

def perform_manual_stitch(upper_series, lower_series):
    """Perform stitching with the manually selected series"""
    print(f"\n{'='*80}")
    print(f"STITCHING:")
    print(f"  UPPER: {upper_series['desc']} (Z: {upper_series['z_min']:.1f} to {upper_series['z_max']:.1f})")
    print(f"  LOWER: {lower_series['desc']} (Z: {lower_series['z_min']:.1f} to {lower_series['z_max']:.1f})")
    print(f"{'='*80}\n")
    
    # Load volumes
    reader = sitk.ImageSeriesReader()
    
    print("Loading UPPER volume...")
    reader.SetFileNames(upper_series['paths'])
    upper_img = reader.Execute()
    
    print("Loading LOWER volume...")
    reader.SetFileNames(lower_series['paths'])
    lower_img = reader.Execute()
    
    # Simple XY alignment
    print("Aligning XY centers...")
    upper_size = upper_img.GetSize()
    lower_size = lower_img.GetSize()
    
    # Get center of mass for alignment
    def get_com(img, slice_idx):
        slice_img = img[:, :, slice_idx]
        mask = sitk.BinaryThreshold(slice_img, lowerThreshold=-500, upperThreshold=3000)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetNumberOfLabels() > 0:
            return stats.GetCentroid(1)[:2]
        return img.GetOrigin()[:2]
    
    com_upper = get_com(upper_img, upper_size[2]-1)
    com_lower = get_com(lower_img, 0)
    
    dx = com_upper[0] - com_lower[0]
    dy = com_upper[1] - com_lower[1]
    
    print(f"  Shift: X={dx:.1f}mm, Y={dy:.1f}mm")
    
    transform = sitk.TranslationTransform(3)
    transform.SetOffset((-dx, -dy, 0))
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(upper_img)
    resampler.SetOutputOrigin([upper_img.GetOrigin()[0], upper_img.GetOrigin()[1], lower_img.GetOrigin()[2]])
    resampler.SetSize([upper_img.GetSize()[0], upper_img.GetSize()[1], lower_img.GetSize()[2]])
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000)
    
    lower_img_aligned = resampler.Execute(lower_img)
    
    # Find optimal Z overlap
    print("\nSearching for optimal Z-overlap...")
    upper_arr = sitk.GetArrayFromImage(upper_img)
    lower_arr = sitk.GetArrayFromImage(lower_img_aligned)
    
    upper_len = len(upper_arr)
    lower_len = len(lower_arr)
    max_overlap = min(upper_len // 2, lower_len // 2, 150)
    
    best_score = -1
    best_overlap = 0
    
    def calc_similarity(a, b):
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)
        mask = (a_flat > -500) & (b_flat > -500)
        if np.sum(mask) == 0: return 0.0
        a_t = a_flat[mask]
        b_t = b_flat[mask]
        a_t -= np.mean(a_t)
        b_t -= np.mean(b_t)
        num = np.sum(a_t * b_t)
        denom = np.linalg.norm(a_t) * np.linalg.norm(b_t)
        return (num / denom) if denom > 0 else 0.0
    
    print(f"\nTesting overlaps from 1 to {max_overlap} slices...")
    for k in range(1, max_overlap + 1):
        upper_block = upper_arr[-k:]
        lower_block = lower_arr[:k]
        
        score = calc_similarity(upper_block, lower_block)
        
        if score > best_score:
            best_score = score
            best_overlap = k
        
        if k % 20 == 0 or score > 0.7:
            print(f"  Overlap {k:3d} slices: score = {score:.4f}")
    
    print(f"\n✓ OPTIMAL OVERLAP: {best_overlap} slices (score: {best_score:.4f})")
    print(f"  This is {best_overlap * upper_img.GetSpacing()[2]:.1f}mm of anatomical overlap\n")
    
    # Create stitched volume
    stitched_arr = np.concatenate((upper_arr, lower_arr[best_overlap:]), axis=0)
    stitched_img = sitk.GetImageFromArray(stitched_arr)
    stitched_img.SetOrigin(upper_img.GetOrigin())
    stitched_img.SetSpacing(upper_img.GetSpacing())
    stitched_img.SetDirection(upper_img.GetDirection())
    
    print(f"Final stitched volume: {stitched_arr.shape} slices\n")
    
    return stitched_img

def render_volume(sitk_img):
    """Render the stitched volume"""
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    
    dims = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims[0], dims[1], dims[2])
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(0,0,0)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3): mat.SetElement(i, j, direction[i + j*3])
    mat.SetElement(0, 3, origin[0])
    mat.SetElement(1, 3, origin[1])
    mat.SetElement(2, 3, origin[2])
    
    # Setup rendering
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
    render_window.SetWindowName(f"Manual Stitch Result: {DATASET}")
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Set camera
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
    
    print(f"Rendering {DATASET}...")
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    # Step 1: Scan and list series
    all_series = scan_and_list_series(target_folder)
    
    # Step 2: Get manual configuration
    config = get_manual_config()
    
    if config and len(all_series) >= max(config['upper_series_index'], config['lower_series_index']):
        upper_idx = config['upper_series_index'] - 1  # Convert to 0-indexed
        lower_idx = config['lower_series_index'] - 1
        
        upper_series = all_series[upper_idx]
        lower_series = all_series[lower_idx]
        
        # Step 3: Perform stitching
        stitched_volume = perform_manual_stitch(upper_series, lower_series)
        
        # Step 4: Render
        render_volume(stitched_volume)
    else:
        print("\n⚠ MANUAL CONFIGURATION NEEDED")
        print("1. View the visualization PNG files")
        print("2. Identify which series to stitch")
        print("3. Update MANUAL_CONFIG in this script")
        print("4. Run again\n")
