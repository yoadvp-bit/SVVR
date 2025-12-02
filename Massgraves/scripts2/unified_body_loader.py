"""
3D Slicer-Style DICOM Loader with Study-Level Merging
======================================================

Replicates 3D Slicer's complete workflow:
1. Scan and group by Patient > Study > Series hierarchy
2. Within each study, identify primary axial series
3. Merge all primary series into a single unified volume
4. Apply proper coordinate transformations (LPS → RAS)
5. Handle overlapping regions with weighted averaging

This matches how 3D Slicer combines multiple body region scans into one volume.
"""

import os
import numpy as np
import pydicom
from collections import defaultdict
import vtk
from vtk.util import numpy_support


# Configuration
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Jarek"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"
MAX_TEXTURE_SIZE = 2048  # OpenGL limit for 3D textures


def scan_dicom_hierarchy(folder_path):
    """
    Step 1: Database Scan - Build Patient > Study > Series hierarchy
    """
    print(f"\n{'='*120}")
    print(f"3D SLICER-STYLE DICOM LOADER WITH STUDY MERGING: {os.path.basename(folder_path)}")
    print(f"{'='*120}\n")
    
    print("Step 1: Scanning DICOM hierarchy...")
    
    # Hierarchical structure: Patient > Study > Series
    hierarchy = {}
    
    files_scanned = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            files_scanned += 1
            
            if files_scanned % 500 == 0:
                print(f"  {files_scanned} files scanned...")
            
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                # Extract hierarchy
                patient_id = getattr(dcm, 'PatientID', 'UNKNOWN')
                patient_name = str(getattr(dcm, 'PatientName', 'UNKNOWN'))
                study_uid = dcm.StudyInstanceUID
                series_uid = dcm.SeriesInstanceUID
                
                patient_key = f"{patient_name} ({patient_id})"
                
                # Initialize patient if needed
                if patient_key not in hierarchy:
                    hierarchy[patient_key] = {}
                
                # Initialize study if needed
                if study_uid not in hierarchy[patient_key]:
                    hierarchy[patient_key][study_uid] = {
                        'study_metadata': {
                            'StudyDate': getattr(dcm, 'StudyDate', 'UNKNOWN'),
                            'StudyDescription': getattr(dcm, 'StudyDescription', 'UNKNOWN'),
                        },
                        'series': {}
                    }
                
                study_info = hierarchy[patient_key][study_uid]
                
                # Initialize series if needed
                if series_uid not in study_info['series']:
                    study_info['series'][series_uid] = {
                        'metadata': {
                            'SeriesNumber': int(getattr(dcm, 'SeriesNumber', 0)),
                            'SeriesDescription': getattr(dcm, 'SeriesDescription', 'UNKNOWN'),
                            'Modality': getattr(dcm, 'Modality', 'UNKNOWN'),
                            'ImageType': getattr(dcm, 'ImageType', []),
                            'ImageOrientationPatient': [float(x) for x in dcm.ImageOrientationPatient],
                            'PixelSpacing': [float(x) for x in dcm.PixelSpacing],
                            'SliceThickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                            'Rows': int(dcm.Rows),
                            'Columns': int(dcm.Columns),
                            'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1.0)),
                            'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', 0.0)),
                        },
                        'files': []
                    }
                
                # Add file to series
                study_info['series'][series_uid]['files'].append({
                    'filepath': filepath,
                    'position': [float(x) for x in dcm.ImagePositionPatient],
                    'instance_number': int(getattr(dcm, 'InstanceNumber', 0))
                })
                
            except Exception as e:
                continue
    
    print(f"  ✓ Scanned {files_scanned} files\n")
    
    return hierarchy


def is_derived_series(image_type):
    """Check if series is derived (reformatted) or primary acquisition"""
    if isinstance(image_type, list):
        image_type_str = '\\'.join(image_type)
    else:
        image_type_str = str(image_type)
    
    return 'DERIVED' in image_type_str.upper()


def calculate_slice_normal(image_orientation):
    """Calculate slice normal vector from ImageOrientationPatient"""
    row_direction = np.array(image_orientation[:3])
    col_direction = np.array(image_orientation[3:6])
    slice_normal = np.cross(row_direction, col_direction)
    slice_normal = slice_normal / np.linalg.norm(slice_normal)
    return slice_normal


def classify_orientation(image_orientation):
    """Classify series by orientation (axial, sagittal, coronal)"""
    slice_normal = calculate_slice_normal(image_orientation)
    
    if np.allclose(np.abs(slice_normal), [0, 0, 1], atol=0.3):
        return "AXIAL"
    elif np.allclose(np.abs(slice_normal), [1, 0, 0], atol=0.3):
        return "SAGITTAL"
    elif np.allclose(np.abs(slice_normal), [0, 1, 0], atol=0.3):
        return "CORONAL"
    else:
        return "OBLIQUE"


def sort_slices_by_position(files_list, image_orientation):
    """Sort slices by projection onto slice normal"""
    slice_normal = calculate_slice_normal(image_orientation)
    
    for file_info in files_list:
        position = np.array(file_info['position'])
        projection = np.dot(position, slice_normal)
        file_info['projection'] = projection
    
    files_list.sort(key=lambda x: x['projection'])
    return files_list


def load_series_volume(series_metadata, series_files):
    """
    Load a single series as a 3D numpy array with physical coordinates
    Returns: (volume_array, origin, spacing, direction_matrix, last_slice_position)
    """
    # Sort slices
    sorted_files = sort_slices_by_position(series_files, series_metadata['ImageOrientationPatient'])
    
    # Calculate spacing
    pixel_spacing = series_metadata['PixelSpacing']
    
    if len(sorted_files) > 1:
        pos_first = np.array(sorted_files[0]['position'])
        pos_second = np.array(sorted_files[1]['position'])
        z_spacing = np.linalg.norm(pos_second - pos_first)
    else:
        z_spacing = series_metadata['SliceThickness']
    
    # CRITICAL: In DICOM, PixelSpacing is [row_spacing, col_spacing]
    # We need [col_spacing, row_spacing, slice_spacing] for [x, y, z]
    spacing = np.array([pixel_spacing[1], pixel_spacing[0], z_spacing])  # [x, y, z]
    origin = np.array(sorted_files[0]['position'])  # [x, y, z] in mm
    
    # Get last slice position for bounds calculation
    last_slice_position = np.array(sorted_files[-1]['position'])
    
    # Direction matrix
    iop = series_metadata['ImageOrientationPatient']
    row_cosine = np.array(iop[:3])  # Direction of rows (Y in image space)
    col_cosine = np.array(iop[3:6])  # Direction of columns (X in image space)
    slice_cosine = np.cross(row_cosine, col_cosine)  # Direction perpendicular to slice
    direction_matrix = np.column_stack([col_cosine, row_cosine, slice_cosine])  # [X, Y, Z] directions
    
    # Allocate volume
    rows = series_metadata['Rows']
    cols = series_metadata['Columns']
    depth = len(sorted_files)
    
    volume_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    # Load pixel data with HU correction
    rescale_slope = series_metadata['RescaleSlope']
    rescale_intercept = series_metadata['RescaleIntercept']
    
    for i, file_info in enumerate(sorted_files):
        dcm = pydicom.dcmread(file_info['filepath'])
        pixel_data = dcm.pixel_array.astype(np.float32)
        hu_data = (pixel_data * rescale_slope) + rescale_intercept
        volume_array[i, :, :] = hu_data.astype(np.int16)
    
    return volume_array, origin, spacing, direction_matrix, last_slice_position


def merge_series_into_unified_volume(series_list, series_metadata_dict):
    """
    Step 2: Merge all primary series into a single unified volume
    
    This is the key step that 3D Slicer performs - combining multiple
    body region scans into one continuous volume.
    """
    print(f"\nStep 2: Merging {len(series_list)} series into unified volume...")
    
    # Load all series
    loaded_series = []
    
    for series_uid in series_list:
        series_meta = series_metadata_dict[series_uid]['metadata']
        series_files = series_metadata_dict[series_uid]['files']
        
        print(f"  Loading series {series_meta['SeriesNumber']}: {series_meta['SeriesDescription']}")
        print(f"    {len(series_files)} slices")
        
        volume, origin, spacing, direction, last_pos = load_series_volume(series_meta, series_files)
        
        # Calculate physical bounds - SIMPLE approach for axial images
        # Volume shape: (depth, rows, cols) -> numpy order is [z, y, x]
        shape = volume.shape
        
        # Physical extent calculation (standard axial orientation)
        # origin is [x, y, z] in mm, shape is [z_slices, y_rows, x_cols]
        x_min = origin[0]
        y_min = origin[1]
        z_min = origin[2]
        
        x_max = origin[0] + shape[2] * spacing[0]  # cols * x_spacing
        y_max = origin[1] + shape[1] * spacing[1]  # rows * y_spacing
        z_max = origin[2] + shape[0] * spacing[2]  # depth * z_spacing
        
        bbox_min = np.array([x_min, y_min, z_min])
        bbox_max = np.array([x_max, y_max, z_max])
        
        loaded_series.append({
            'volume': volume,
            'origin': origin,
            'spacing': spacing,
            'direction': direction,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'shape': shape,
            'series_number': series_meta['SeriesNumber'],
            'description': series_meta['SeriesDescription']
        })
        
        print(f"    Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
        print(f"    Extent: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]")
        print(f"    Shape: {shape}, Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f}mm")
        print(f"    HU range: [{np.min(volume)}, {np.max(volume)}]")
    
    # Calculate global bounding box
    print(f"\n  Calculating global bounding box...")
    
    all_bbox_mins = np.array([s['bbox_min'] for s in loaded_series])
    all_bbox_maxs = np.array([s['bbox_max'] for s in loaded_series])
    
    # CRITICAL FIX: Align all series to have the SAME X-Y origin (use first series as reference)
    # This ensures all series occupy the same X-Y grid region, preventing gaps in rendering
    # We align the bbox_min (top-left corner) rather than centers to handle different FOV sizes
    
    reference_x_min = loaded_series[0]['bbox_min'][0]
    reference_y_min = loaded_series[0]['bbox_min'][1]
    
    print(f"    Reference X-Y origin: X={reference_x_min:.1f}, Y={reference_y_min:.1f}")
    
    # Adjust all series to start at the same X-Y origin
    for s in loaded_series:
        x_shift = reference_x_min - s['bbox_min'][0]
        y_shift = reference_y_min - s['bbox_min'][1]
        
        # Apply shift
        s['origin'][0] += x_shift
        s['origin'][1] += y_shift
        s['bbox_min'][0] += x_shift
        s['bbox_max'][0] += x_shift
        s['bbox_min'][1] += y_shift
        s['bbox_max'][1] += y_shift
        
        if abs(x_shift) > 1 or abs(y_shift) > 1:
            print(f"    Aligned series {s['series_number']}: shifted X={x_shift:.1f}mm, Y={y_shift:.1f}mm")
    
    # Now recalculate global bounds with aligned series
    all_bbox_mins = np.array([s['bbox_min'] for s in loaded_series])
    all_bbox_maxs = np.array([s['bbox_max'] for s in loaded_series])
    
    # CRITICAL FIX: Sort series by Z-position (inferior to superior / feet to head)
    # This ensures legs are placed at bottom (low Z) and head at top (high Z)
    loaded_series.sort(key=lambda s: s['bbox_min'][2])
    print(f"\n  Sorted series by Z-position (inferior → superior):")
    for s in loaded_series:
        z_range = [s['bbox_min'][2], s['bbox_max'][2]]
        print(f"    Series {s['series_number']}: Z=[{z_range[0]:.1f}, {z_range[1]:.1f}] mm - {s['description']}")
    
    global_min = np.min(all_bbox_mins, axis=0)
    global_max = np.max(all_bbox_maxs, axis=0)
    
    print(f"\n    Global min: [{global_min[0]:.2f}, {global_min[1]:.2f}, {global_min[2]:.2f}] mm")
    print(f"    Global max: [{global_max[0]:.2f}, {global_max[1]:.2f}, {global_max[2]:.2f}] mm")
    
    # Use finest spacing (assume all series have same spacing)
    reference_spacing = loaded_series[0]['spacing']
    
    # Calculate unified grid dimensions
    global_extent = global_max - global_min
    unified_dims = np.ceil(global_extent / reference_spacing).astype(int)
    
    print(f"    Unified spacing: [{reference_spacing[0]:.3f}, {reference_spacing[1]:.3f}, {reference_spacing[2]:.3f}] mm")
    print(f"    Unified dimensions: {unified_dims[0]} × {unified_dims[1]} × {unified_dims[2]}")
    print(f"    Total voxels: {np.prod(unified_dims):,}")
    
    # Allocate unified volume
    print(f"\n  Allocating unified volume...")
    unified_volume = np.full((unified_dims[2], unified_dims[1], unified_dims[0]), -1024, dtype=np.int16)  # (z, y, x)
    weight_map = np.zeros((unified_dims[2], unified_dims[1], unified_dims[0]), dtype=np.float32)
    
    # Place each series into unified grid
    print(f"\n  Merging series into unified grid...")
    
    for i, series_info in enumerate(loaded_series, 1):
        print(f"    [{i}/{len(loaded_series)}] Placing: {series_info['description']}")
        
        volume = series_info['volume']
        origin = series_info['origin']
        spacing = series_info['spacing']
        shape = series_info['shape']  # (depth, rows, cols)
        
        # Calculate where this volume maps in the target grid
        # Convert physical position to target grid indices
        x_idx_start = int(np.round((origin[0] - global_min[0]) / reference_spacing[0]))
        y_idx_start = int(np.round((origin[1] - global_min[1]) / reference_spacing[1]))
        z_idx_start = int(np.round((origin[2] - global_min[2]) / reference_spacing[2]))
        
        # End indices based on volume shape
        x_idx_end = x_idx_start + shape[2]  # cols
        y_idx_end = y_idx_start + shape[1]  # rows
        z_idx_end = z_idx_start + shape[0]  # depth
        
        # Clamp to valid range
        dst_x_start = max(0, x_idx_start)
        dst_x_end = min(unified_dims[0], x_idx_end)
        dst_y_start = max(0, y_idx_start)
        dst_y_end = min(unified_dims[1], y_idx_end)
        dst_z_start = max(0, z_idx_start)
        dst_z_end = min(unified_dims[2], z_idx_end)
        
        # Source indices (handle negative starts)
        src_x_start = max(0, -x_idx_start)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        src_y_start = max(0, -y_idx_start)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_z_start = max(0, -z_idx_start)
        src_z_end = src_z_start + (dst_z_end - dst_z_start)
        
        # Get data slice (numpy order: depth, rows, cols)
        data_slice = volume[src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Check for overlap (numpy order: z, y, x)
        existing_region = unified_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        has_data = existing_region > -1024
        overlap_voxels = np.sum(has_data)
        total_voxels = data_slice.size
        overlap_percentage = (overlap_voxels / total_voxels * 100) if total_voxels > 0 else 0
        
        print(f"      Grid position: X=[{dst_x_start}:{dst_x_end}], Y=[{dst_y_start}:{dst_y_end}], Z=[{dst_z_start}:{dst_z_end}]")
        print(f"      Overlap: {overlap_voxels:,} / {total_voxels:,} voxels ({overlap_percentage:.1f}%)")
        
        # Weighted averaging for overlaps
        target_region = unified_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        target_weights = weight_map[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end]
        
        # Convert to float for weighted average
        target_region_float = target_region.astype(np.float32)
        data_slice_float = data_slice.astype(np.float32)
        
        # Where we have overlap, do weighted average
        mask = has_data
        if np.any(mask):
            target_region_float[mask] = (
                target_region_float[mask] * target_weights[mask] + data_slice_float[mask]
            ) / (target_weights[mask] + 1)
        
        # Where no overlap, just place data
        no_overlap_mask = ~mask
        target_region_float[no_overlap_mask] = data_slice_float[no_overlap_mask]
        
        # Update unified volume and weights
        unified_volume[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = target_region_float.astype(np.int16)
        weight_map[dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end] += 1
    
    print(f"\n  ✓ Unified volume created")
    print(f"    Final HU range: [{np.min(unified_volume)}, {np.max(unified_volume)}]")
    
    return unified_volume, global_min, reference_spacing


def create_vtk_volume(volume_array, origin, spacing):
    """
    Step 3: Create VTK ImageData with LPS→RAS transformation
    Includes downsampling if volume exceeds OpenGL limits
    """
    print(f"\n  Creating VTK volume...")
    
    # Apply LPS → RAS transformation
    ras_origin = np.array([-origin[0], -origin[1], origin[2]])
    
    print(f"    LPS Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
    print(f"    RAS Origin: [{ras_origin[0]:.2f}, {ras_origin[1]:.2f}, {ras_origin[2]:.2f}] mm")
    
    # Check if downsampling is needed
    dims = volume_array.shape  # (z, y, x)
    max_dim = max(dims)
    
    if max_dim > MAX_TEXTURE_SIZE:
        print(f"    ⚠️  Volume too large for OpenGL ({max_dim} > {MAX_TEXTURE_SIZE})")
        print(f"    Downsampling...")
        
        # Calculate downsample factor
        downsample_factor = int(np.ceil(max_dim / MAX_TEXTURE_SIZE))
        
        # Downsample using slicing (fast, no interpolation needed)
        downsampled = volume_array[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        # Update spacing
        adjusted_spacing = spacing * downsample_factor
        
        print(f"    Downsample factor: {downsample_factor}x")
        print(f"    New dimensions: {downsampled.shape[2]}×{downsampled.shape[1]}×{downsampled.shape[0]}")
        print(f"    New spacing: [{adjusted_spacing[0]:.3f}, {adjusted_spacing[1]:.3f}, {adjusted_spacing[2]:.3f}] mm")
        
        volume_array = downsampled
        spacing = adjusted_spacing
    
    # Convert to VTK
    # Numpy: (z, y, x) order, VTK: (x, y, z) order
    # Need to transpose
    vtk_array = np.transpose(volume_array, (2, 1, 0))  # (x, y, z)
    
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=vtk_array.ravel(order='F'),  # Fortran order for VTK
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    dims = vtk_array.shape  # (x, y, z)
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(dims[0], dims[1], dims[2])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(ras_origin[0], ras_origin[1], ras_origin[2])
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    print(f"    ✓ VTK volume ready: {dims[0]}×{dims[1]}×{dims[2]} voxels")
    
    return vtk_image


def render_volume(vtk_image, title):
    """Render the unified volume and capture screenshots"""
    print(f"\n  Step 4: Rendering volume...")
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
    # Transfer functions
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-200, 0.0)
    opacity.AddPoint(-100, 0.05)
    opacity.AddPoint(40, 0.2)
    opacity.AddPoint(200, 0.6)
    opacity.AddPoint(500, 1.0)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-100, 0.7, 0.5, 0.4)
    color.AddRGBPoint(40, 0.9, 0.3, 0.3)
    color.AddRGBPoint(200, 1.0, 0.95, 0.95)
    color.AddRGBPoint(500, 1.0, 1.0, 1.0)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(1)  # Enable off-screen rendering
    
    # Render once
    render_window.Render()
    
    # Capture screenshots from different angles
    output_dir = "/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/scripts2"
    dataset_name = DATASET.replace("DICOM-", "")
    
    # Window to image filter
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    
    # PNG writer
    writer = vtk.vtkPNGWriter()
    
    # Capture front view
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 1)
    camera.SetViewUp(0, 1, 0)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()
    render_window.Render()
    w2if.Modified()
    w2if.Update()
    
    front_file = f"{output_dir}/{dataset_name}_merged_front.png"
    writer.SetFileName(front_file)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()
    print(f"    ✓ Saved front view: {front_file}")
    
    # Capture side view
    camera.SetPosition(1, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()
    render_window.Render()
    w2if.Modified()
    w2if.Update()
    
    side_file = f"{output_dir}/{dataset_name}_merged_side.png"
    writer.SetFileName(side_file)
    writer.Write()
    print(f"    ✓ Saved side view: {side_file}")
    
    # Capture top view
    camera.SetPosition(0, 0, 1)
    camera.SetViewUp(0, 1, 0)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()
    camera.Azimuth(45)
    camera.Elevation(30)
    renderer.ResetCamera()
    render_window.Render()
    w2if.Modified()
    w2if.Update()
    
    angle_file = f"{output_dir}/{dataset_name}_merged_angle.png"
    writer.SetFileName(angle_file)
    writer.Write()
    print(f"    ✓ Saved angled view: {angle_file}")
    
    # Skip interactive mode - just save screenshots and exit
    print(f"    ✓ Screenshots saved successfully")
    print(f"\n" + "="*80)
    print("COMPLETE - All screenshots generated")
    print("="*80)


def main():
    """
    Main pipeline - scan, identify primary series, merge, and visualize
    """
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Step 1: Scan hierarchy
    hierarchy = scan_dicom_hierarchy(dataset_path)
    
    if len(hierarchy) == 0:
        print("✗ No DICOM data found")
        return
    
    # Process each patient/study
    for patient_name, studies in hierarchy.items():
        print(f"\n{'='*120}")
        print(f"PATIENT: {patient_name}")
        print(f"{'='*120}")
        
        for study_uid, study_info in studies.items():
            study_meta = study_info['study_metadata']
            print(f"\nStudy: {study_meta['StudyDescription']} ({study_meta['StudyDate']})")
            
            # Identify primary axial series
            primary_series = []
            
            print(f"  Analyzing {len(study_info['series'])} series:")
            
            for series_uid, series_info in study_info['series'].items():
                series_meta = series_info['metadata']
                
                is_primary = not is_derived_series(series_meta['ImageType'])
                orientation = classify_orientation(series_meta['ImageOrientationPatient'])
                num_files = len(series_info['files'])
                
                print(f"    Series {series_meta['SeriesNumber']}: {series_meta['SeriesDescription'][:40]:<40} | {num_files:4d} slices | {'PRIMARY' if is_primary else 'DERIVED'} | {orientation}")
                
                # Filter: primary acquisitions, axial orientation, enough slices
                if is_primary and orientation == 'AXIAL' and num_files >= 10:
                    primary_series.append(series_uid)
                    print(f"      ✓ Selected for merging")
            
            if len(primary_series) == 0:
                print("  ✗ No primary axial series found")
                continue
            
            print(f"  Found {len(primary_series)} primary axial series to merge")
            
            # Merge series
            unified_volume, origin, spacing = merge_series_into_unified_volume(
                primary_series,
                study_info['series']
            )
            
            # Create VTK volume
            vtk_image = create_vtk_volume(unified_volume, origin, spacing)
            
            # Render
            title = f"{DATASET} - {patient_name} - {study_meta['StudyDescription']}"
            
            print(f"\n{'='*120}")
            print(f"DISPLAYING UNIFIED VOLUME: {title}")
            print(f"{'='*120}\n")
            
            render_volume(vtk_image, title)
    
    print(f"\n{'='*120}")
    print(f"✓ COMPLETE")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
