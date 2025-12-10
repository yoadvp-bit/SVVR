"""
Universal Body Assembly System
===============================

Assembles complete human body from ALL available DICOM series, handling:
- Multiple orientations (axial, sagittal, coronal)
- Different voxel spacing (0.7mm, 1.0mm, 3.0mm, 5.0mm)
- Different reconstruction kernels (B20f, B30f, B60f, B70f, B80f, etc.)
- Overlapping series
- Missing slices

Key Principle: Use EVERY series found, resample to common space, merge optimally

Version: 3.0 - Universal Multi-Series Assembly
Date: 2025-12-01
"""

import vtk
from vtk.util import numpy_support
import numpy as np
import pydicom
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

DATASET = "DICOM-Jarek"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"
BASE_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"

# Target spacing for resampling (mm) - use finest common spacing
TARGET_SPACING = [0.7, 0.7, 1.0]  # [x, y, z] in mm


# ==================== DICOM LOADING ====================

def scan_all_dicom_series(root_path):
    """
    Scan directory and find ALL DICOM series (no filtering)
    Returns map of SeriesInstanceUID -> list of (z_pos, filepath)
    """
    print(f"\n{'='*100}")
    print(f"UNIVERSAL BODY ASSEMBLY: {os.path.basename(root_path)}")
    print(f"{'='*100}\n")
    
    print("Step 1: Scanning ALL DICOM files...")
    series_map = defaultdict(list)
    series_metadata = {}
    files_checked = 0
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            files_checked += 1
            
            if files_checked % 500 == 0:
                print(f"  Scanned {files_checked} files...")
            
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                uid = dcm.SeriesInstanceUID
                z_pos = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z_pos, filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'rows': int(dcm.Rows),
                        'cols': int(dcm.Columns),
                        'pixel_spacing': list(dcm.PixelSpacing) if 'PixelSpacing' in dcm else [1.0, 1.0],
                        'slice_thickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                        'orientation': list(dcm.ImageOrientationPatient) if 'ImageOrientationPatient' in dcm else None,
                        'position': list(dcm.ImagePositionPatient)
                    }
                    
            except Exception as e:
                continue
    
    print(f"  → Found {len(series_map)} series from {files_checked} files\n")
    return series_map, series_metadata


def load_volume_from_dicom(file_list, metadata):
    """
    Load DICOM series into numpy array with proper HU values and spacing
    Returns: (volume_array, spacing, origin, direction)
    """
    # Sort by Z position
    file_list.sort(key=lambda x: x[0])
    sorted_paths = [x[1] for x in file_list]
    
    # Get dimensions
    rows = metadata['rows']
    cols = metadata['cols']
    depth = len(sorted_paths)
    
    # Get spacing
    pixel_spacing = metadata['pixel_spacing']  # [x, y]
    
    # Calculate Z-spacing from actual slice positions
    if depth > 1:
        z_positions = [x[0] for x in file_list]
        z_diffs = [abs(z_positions[i+1] - z_positions[i]) for i in range(len(z_positions)-1)]
        z_spacing = np.median(z_diffs) if z_diffs else metadata['slice_thickness']
    else:
        z_spacing = metadata['slice_thickness']
    
    spacing = [pixel_spacing[0], pixel_spacing[1], z_spacing]
    
    # Get origin from first slice (actual physical position)
    origin = metadata['position']
    
    # Get direction/orientation
    direction = metadata['orientation'] if metadata['orientation'] else [1,0,0,0,1,0]
    
    # Load pixel data
    volume_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    for i, path in enumerate(sorted_paths):
        dcm = pydicom.dcmread(path)
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        
        slice_data = dcm.pixel_array.astype(np.float64)
        slice_data = (slice_data * slope) + intercept
        volume_array[i, :, :] = slice_data.astype(np.int16)
    
    return volume_array, spacing, origin, direction


def classify_series_orientation(metadata):
    """
    Determine if series is axial, sagittal, coronal, or oblique
    """
    if metadata['orientation'] is None:
        return 'UNKNOWN'
    
    iop = metadata['orientation']
    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
    vec_z = vec_z / np.linalg.norm(vec_z)
    
    if np.allclose(vec_z, [0, 0, 1], atol=0.3) or np.allclose(vec_z, [0, 0, -1], atol=0.3):
        return 'AXIAL'
    elif np.allclose(vec_z, [1, 0, 0], atol=0.3) or np.allclose(vec_z, [-1, 0, 0], atol=0.3):
        return 'SAGITTAL'
    elif np.allclose(vec_z, [0, 1, 0], atol=0.3) or np.allclose(vec_z, [0, -1, 0], atol=0.3):
        return 'CORONAL'
    else:
        return 'OBLIQUE'


# ==================== SERIES LOADING ====================

def load_all_series(series_map, series_metadata):
    """
    Load all PRIMARY series (axial acquisitions with real Z-spacing)
    Skip reformatted views (sagittal/coronal) which are derived data
    """
    print("Step 2: Loading all PRIMARY series volumes...")
    
    all_loaded = []
    skipped = []
    
    for i, (uid, file_list) in enumerate(series_map.items(), 1):
        if len(file_list) < 5:  # Only skip VERY small series
            continue
        
        metadata = series_metadata[uid]
        orientation = classify_series_orientation(metadata)
        
        desc = metadata['desc'][:50]
        num_slices = len(file_list)
        
        file_list.sort(key=lambda x: x[0])
        z_min = file_list[0][0]
        z_max = file_list[-1][0]
        z_span = abs(z_max - z_min)
        
        # Calculate actual Z-spacing
        if len(file_list) > 1:
            z_spacing = z_span / (len(file_list) - 1)
        else:
            z_spacing = metadata['slice_thickness']
        
        # Skip reformats (coronal/sagittal) or series with suspicious Z-spacing
        if orientation in ['CORONAL', 'SAGITTAL', 'OBLIQUE']:
            skipped.append(f"  {i:2}. SKIPPED: {desc:<50} ({num_slices:4} slices, {orientation} - reformat)")
            continue
        
        if z_spacing < 0.3 or z_spacing > 10.0:  # Unrealistic spacing
            skipped.append(f"  {i:2}. SKIPPED: {desc:<50} (Z-spacing: {z_spacing:.2f}mm - suspicious)")
            continue
        
        print(f"  {i:2}. Loading: {desc:<50} ({num_slices:4} slices, {orientation})")
        
        try:
            volume_arr, spacing, origin, direction = load_volume_from_dicom(file_list, metadata)
            
            # Verify spacing is reasonable
            if spacing[2] < 0.3 or spacing[2] > 10.0:
                skipped.append(f"  {i:2}. SKIPPED: {desc:<50} (Invalid Z-spacing: {spacing[2]:.2f}mm)")
                continue
            
            series_info = {
                'uid': uid,
                'desc': metadata['desc'],
                'orientation': orientation,
                'volume': volume_arr,
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'z_min': z_min,
                'z_max': z_max,
                'z_span': z_span,
                'num_slices': num_slices,
                'shape': volume_arr.shape
            }
            
            all_loaded.append(series_info)
            
            print(f"      → Shape: {volume_arr.shape}, Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f}mm, Origin: [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]")
            
        except Exception as e:
            print(f"      → FAILED: {e}")
            continue
    
    print(f"\n  Successfully loaded {len(all_loaded)} PRIMARY series")
    
    if skipped:
        print(f"  Skipped {len(skipped)} reformatted/invalid series:\n")
        for s in skipped:
            print(s)
    
    print()
    return all_loaded


# ==================== RESAMPLING ====================

def resample_volume_to_target(volume_arr, original_spacing, target_spacing):
    """
    Resample volume to target spacing using scipy
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors for each dimension
    zoom_factors = [
        original_spacing[2] / target_spacing[2],  # Z
        original_spacing[1] / target_spacing[1],  # Y
        original_spacing[0] / target_spacing[0],  # X
    ]
    
    # Only resample if needed
    if np.allclose(zoom_factors, [1, 1, 1], atol=0.01):
        return volume_arr
    
    print(f"    Resampling: {volume_arr.shape} with factors {[f'{z:.3f}' for z in zoom_factors]}")
    
    # Use order=1 (linear) for speed, order=3 (cubic) for quality
    resampled = zoom(volume_arr, zoom_factors, order=1, mode='nearest')
    
    print(f"    → New shape: {resampled.shape}")
    
    return resampled


# ==================== INTELLIGENT MERGING ====================

def find_axial_series_for_base(all_series):
    """
    Find best axial series to use as base reference
    Prefer: largest Z-span, most slices, finest spacing
    """
    axial_series = [s for s in all_series if s['orientation'] == 'AXIAL']
    
    if not axial_series:
        # Fall back to any series
        axial_series = all_series
    
    # Score by: z_span * 0.5 + num_slices * 0.3 + (1/spacing[2]) * 100
    scored = []
    for s in axial_series:
        score = s['z_span'] * 0.5 + s['num_slices'] * 0.3 + (1.0 / s['spacing'][2]) * 100
        scored.append((s, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def merge_all_volumes(all_series, target_spacing):
    """
    Merge all series into single volume using ACTUAL physical coordinates
    """
    print("Step 3: Analyzing physical coordinate system...")
    
    # Find global bounding box in physical space (mm)
    all_x_mins, all_x_maxs = [], []
    all_y_mins, all_y_maxs = [], []
    all_z_mins, all_z_maxs = [], []
    
    for s in all_series:
        origin = s['origin']  # [x, y, z] in mm
        spacing = s['spacing']  # [x, y, z] spacing in mm
        shape = s['shape']  # [z, y, x] voxels
        
        # Calculate physical extent
        x_min = origin[0]
        y_min = origin[1]
        z_min = origin[2]
        
        x_max = origin[0] + shape[2] * spacing[0]
        y_max = origin[1] + shape[1] * spacing[1]
        z_max = origin[2] + shape[0] * spacing[2]
        
        all_x_mins.append(x_min)
        all_x_maxs.append(x_max)
        all_y_mins.append(y_min)
        all_y_maxs.append(y_max)
        all_z_mins.append(z_min)
        all_z_maxs.append(z_max)
        
        print(f"  Series: {s['desc'][:40]}")
        print(f"    Physical bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]")
    
    # Global bounding box
    global_x_min = min(all_x_mins)
    global_x_max = max(all_x_maxs)
    global_y_min = min(all_y_mins)
    global_y_max = max(all_y_maxs)
    global_z_min = min(all_z_mins)
    global_z_max = max(all_z_maxs)
    
    print(f"\n  Global physical bounds:")
    print(f"    X: [{global_x_min:.1f}, {global_x_max:.1f}] mm → {global_x_max - global_x_min:.1f}mm")
    print(f"    Y: [{global_y_min:.1f}, {global_y_max:.1f}] mm → {global_y_max - global_y_min:.1f}mm")
    print(f"    Z: [{global_z_min:.1f}, {global_z_max:.1f}] mm → {global_z_max - global_z_min:.1f}mm")
    
    # Calculate target dimensions
    target_dims = [
        int(np.ceil((global_z_max - global_z_min) / target_spacing[2])),  # Z slices
        int(np.ceil((global_y_max - global_y_min) / target_spacing[1])),  # Y
        int(np.ceil((global_x_max - global_x_min) / target_spacing[0]))   # X
    ]
    
    print(f"\n  Target volume dimensions: {target_dims} (Z×Y×X)")
    print(f"  Target spacing: {target_spacing} mm")
    print(f"  Target origin: [{global_x_min:.1f}, {global_y_min:.1f}, {global_z_min:.1f}]\n")
    
    # Create accumulator volumes
    merged_volume = np.full(target_dims, -1024, dtype=np.float32)
    weight_volume = np.zeros(target_dims, dtype=np.float32)
    
    print("Step 4: Placing series in physical space...")
    
    for i, series in enumerate(all_series, 1):
        print(f"\n  {i}/{len(all_series)}: {series['desc'][:50]}")
        
        # Resample to target spacing if needed
        if not np.allclose(series['spacing'], target_spacing, atol=0.01):
            print(f"    Resampling from {series['spacing'][0]:.2f}×{series['spacing'][1]:.2f}×{series['spacing'][2]:.2f} to {target_spacing[0]}×{target_spacing[1]}×{target_spacing[2]}...")
            resampled_vol = resample_volume_to_target(series['volume'], series['spacing'], target_spacing)
        else:
            resampled_vol = series['volume']
        
        # Calculate where this volume maps in the target grid
        # Physical position of first voxel
        origin = series['origin']
        
        # Convert physical position to target grid indices
        x_idx_start = int(np.round((origin[0] - global_x_min) / target_spacing[0]))
        y_idx_start = int(np.round((origin[1] - global_y_min) / target_spacing[1]))
        z_idx_start = int(np.round((origin[2] - global_z_min) / target_spacing[2]))
        
        # End indices
        x_idx_end = x_idx_start + resampled_vol.shape[2]
        y_idx_end = y_idx_start + resampled_vol.shape[1]
        z_idx_end = z_idx_start + resampled_vol.shape[0]
        
        print(f"    Resampled shape: {resampled_vol.shape}")
        print(f"    Target indices: X=[{x_idx_start}:{x_idx_end}], Y=[{y_idx_start}:{y_idx_end}], Z=[{z_idx_start}:{z_idx_end}]")
        
        # Clamp to valid range and calculate overlap
        dst_x_start = max(0, x_idx_start)
        dst_x_end = min(target_dims[2], x_idx_end)
        dst_y_start = max(0, y_idx_start)
        dst_y_end = min(target_dims[1], y_idx_end)
        dst_z_start = max(0, z_idx_start)
        dst_z_end = min(target_dims[0], z_idx_end)
        
        # Source indices (handle negative starts)
        src_x_start = max(0, -x_idx_start)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        src_y_start = max(0, -y_idx_start)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_z_start = max(0, -z_idx_start)
        src_z_end = src_z_start + (dst_z_end - dst_z_start)
        
        # Clamp source to actual volume size
        src_x_end = min(src_x_end, resampled_vol.shape[2])
        src_y_end = min(src_y_end, resampled_vol.shape[1])
        src_z_end = min(src_z_end, resampled_vol.shape[0])
        
        # Adjust destination to match actual source
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_z_end = dst_z_start + (src_z_end - src_z_start)
        
        if dst_z_end <= dst_z_start or dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
            print(f"    ✗ Skipped: Empty region after clamping")
            continue
        
        # Extract region
        region = resampled_vol[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        
        if region.size == 0:
            print(f"    ✗ Skipped: Zero-size region")
            continue
        
        # Mask for valid data
        mask = region > -900
        num_valid = np.sum(mask)
        
        # Weight based on slice thickness (finer = better quality)
        weight = 1.0 / np.sqrt(series['spacing'][2])
        
        # Place in target volume
        try:
            merged_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end][mask] += region[mask] * weight
            weight_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end][mask] += weight
            
            # Check for overlap
            overlap_count = np.sum(weight_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] > weight)
            overlap_pct = 100 * overlap_count / num_valid if num_valid > 0 else 0
            
            print(f"    ✓ Placed: {num_valid:,} voxels, {overlap_pct:.1f}% overlaps with existing data")
            
        except Exception as e:
            print(f"    ✗ Placement error: {e}")
            continue
    
    # Normalize by weights
    print("\nStep 5: Normalizing merged volume...")
    valid_mask = weight_volume > 0
    merged_volume[valid_mask] = merged_volume[valid_mask] / weight_volume[valid_mask]
    
    # Convert to int16
    final_volume = merged_volume.astype(np.int16)
    
    non_air = np.sum(final_volume > -900)
    print(f"  → Final shape: {final_volume.shape}")
    print(f"  → Non-air voxels: {non_air:,} ({100*non_air/final_volume.size:.1f}%)")
    print(f"  → Value range: {np.min(final_volume)} to {np.max(final_volume)} HU\n")
    
    return final_volume, target_spacing, [global_x_min, global_y_min, global_z_min]


# ==================== RENDERING ====================

def render_merged_volume(volume_arr, spacing, origin, title):
    """
    Render final merged volume with VTK
    """
    print("Step 6: Rendering final merged volume...")
    
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=volume_arr.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(volume_arr.shape[2], volume_arr.shape[1], volume_arr.shape[0])  # X, Y, Z
    vtk_img.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_img.SetOrigin(origin)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    # Volume rendering setup
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-200, 0.0)
    opacity.AddPoint(-100, 0.05)
    opacity.AddPoint(40, 0.15)
    opacity.AddPoint(200, 0.5)
    opacity.AddPoint(500, 0.9)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-100, 0.7, 0.5, 0.4)
    color.AddRGBPoint(40, 0.9, 0.3, 0.3)
    color.AddRGBPoint(200, 1.0, 0.95, 0.95)
    color.AddRGBPoint(500, 1.0, 1.0, 1.0)
    
    volume_prop = vtk.vtkVolumeProperty()
    volume_prop.SetColor(color)
    volume_prop.SetScalarOpacity(opacity)
    volume_prop.ShadeOn()
    volume_prop.SetInterpolationTypeToLinear()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_prop)
    
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1400, 1400)
    render_window.SetWindowName(f"{title} | All {len(all_loaded)} Series Merged")
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Camera setup
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Elevation(-90)  # Look from side
    renderer.ResetCameraClippingRange()
    
    print("  Opening 3D viewer...\n")
    render_window.Render()
    interactor.Start()


# ==================== MAIN PIPELINE ====================

if __name__ == "__main__":
    target_path = os.path.join(BASE_PATH, DATASET)
    
    # Step 1: Scan all DICOM files
    series_map, series_metadata = scan_all_dicom_series(target_path)
    
    if len(series_map) == 0:
        print("✗ No DICOM series found\n")
        exit(1)
    
    # Step 2: Load all series
    all_loaded = load_all_series(series_map, series_metadata)
    
    if len(all_loaded) == 0:
        print("✗ No series could be loaded\n")
        exit(1)
    
    # Steps 3-5: Merge all volumes
    final_volume, final_spacing, final_origin = merge_all_volumes(all_loaded, TARGET_SPACING)
    
    # Step 6: Render
    render_merged_volume(final_volume, final_spacing, final_origin, f"Universal Assembly: {DATASET}")
    
    print(f"{'='*100}")
    print(f"✓ UNIVERSAL ASSEMBLY COMPLETE")
    print(f"  Merged {len(all_loaded)} series into single volume")
    print(f"  Final dimensions: {final_volume.shape}")
    print(f"  Spacing: {final_spacing} mm")
    print(f"{'='*100}\n")
