"""
3D Slicer-Style DICOM Loader
=============================

Replicates 3D Slicer's exact approach to loading and assembling DICOM series:
1. Group by SeriesInstanceUID (ignore filenames)
2. Sort by ImagePositionPatient (Z-coordinate projection)
3. Build geometry matrix from ImageOrientationPatient
4. Apply proper coordinate transformation (LPS → RAS)
5. Fill voxels with Hounsfield Unit correction

This matches 3D Slicer's vtkImageData construction pipeline.
"""

import os
import numpy as np
import pydicom
from collections import defaultdict
import vtk
from vtk.util import numpy_support


# Configuration
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Maria"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"


def scan_dicom_folder(folder_path):
    """
    Step 1: Database Scan - Group files by SeriesInstanceUID
    Ignores filenames completely, only uses DICOM headers
    """
    print(f"\n{'='*100}")
    print(f"3D SLICER-STYLE DICOM LOADER: {os.path.basename(folder_path)}")
    print(f"{'='*100}\n")
    
    print("Step 1: Database Scan - Grouping by SeriesInstanceUID...")
    
    series_groups = defaultdict(list)
    series_metadata = {}
    files_scanned = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            files_scanned += 1
            
            if files_scanned % 500 == 0:
                print(f"  Scanned {files_scanned} files...")
            
            try:
                # Read only header (fast)
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                # Check if this is a valid image
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                # Group by SeriesInstanceUID (the key grouping criterion)
                series_uid = dcm.SeriesInstanceUID
                
                # Extract critical spatial information
                image_position = list(dcm.ImagePositionPatient)  # [X, Y, Z] in mm
                
                series_groups[series_uid].append({
                    'filepath': filepath,
                    'position': image_position,
                    'instance_number': int(getattr(dcm, 'InstanceNumber', 0))
                })
                
                # Store series metadata (from first file in series)
                if series_uid not in series_metadata:
                    series_metadata[series_uid] = {
                        'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'SeriesNumber': int(getattr(dcm, 'SeriesNumber', 0)),
                        'Modality': getattr(dcm, 'Modality', 'Unknown'),
                        'ImageOrientationPatient': list(dcm.ImageOrientationPatient),
                        'PixelSpacing': list(dcm.PixelSpacing),
                        'SliceThickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                        'Rows': int(dcm.Rows),
                        'Columns': int(dcm.Columns),
                        'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1.0)),
                        'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', 0.0)),
                    }
                
            except Exception as e:
                continue
    
    print(f"  → Scanned {files_scanned} files")
    print(f"  → Found {len(series_groups)} unique series\n")
    
    return series_groups, series_metadata


def calculate_slice_normal(image_orientation):
    """
    Step 2: Calculate slice normal vector from ImageOrientationPatient
    
    ImageOrientationPatient contains 6 values:
    [row_x, row_y, row_z, col_x, col_y, col_z]
    
    The slice normal is the cross product: row_direction × col_direction
    """
    row_direction = np.array(image_orientation[:3])
    col_direction = np.array(image_orientation[3:6])
    
    # Cross product gives the normal to the slice plane
    slice_normal = np.cross(row_direction, col_direction)
    slice_normal = slice_normal / np.linalg.norm(slice_normal)  # Normalize
    
    return slice_normal


def sort_slices_by_position(files_list, image_orientation):
    """
    Step 3: Spatial Sorting - Sort slices by projection onto slice normal
    
    This is how 3D Slicer determines slice order, NOT by InstanceNumber!
    """
    slice_normal = calculate_slice_normal(image_orientation)
    
    # Project each slice position onto the normal vector
    for file_info in files_list:
        position = np.array(file_info['position'])
        projection = np.dot(position, slice_normal)
        file_info['projection'] = projection
    
    # Sort by projection (this gives us the anatomical order)
    files_list.sort(key=lambda x: x['projection'])
    
    return files_list


def build_volume_geometry(series_metadata, sorted_files):
    """
    Step 4: Construct the Matrix - Calculate physical geometry
    
    Returns:
    - spacing: [x, y, z] voxel size in mm
    - origin: [x, y, z] physical position of first voxel
    - direction: 3x3 orientation matrix
    """
    # Get spacing from metadata
    pixel_spacing = series_metadata['PixelSpacing']  # [row_spacing, col_spacing]
    
    # Calculate Z-spacing from actual slice positions
    if len(sorted_files) > 1:
        pos_first = np.array(sorted_files[0]['position'])
        pos_second = np.array(sorted_files[1]['position'])
        z_spacing = np.linalg.norm(pos_second - pos_first)
    else:
        z_spacing = series_metadata['SliceThickness']
    
    # Spacing: [X, Y, Z] in mm per voxel
    spacing = [pixel_spacing[1], pixel_spacing[0], z_spacing]  # Note: VTK uses [x,y,z] order
    
    # Origin: Physical position of the first slice
    origin = sorted_files[0]['position']
    
    # Direction matrix from ImageOrientationPatient
    iop = series_metadata['ImageOrientationPatient']
    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:6])
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    # Build 3x3 direction matrix (columns are the basis vectors)
    # VTK/ITK convention: [row_direction, col_direction, slice_direction]
    direction_matrix = np.column_stack([row_cosine, col_cosine, slice_cosine])
    
    return spacing, origin, direction_matrix


def apply_lps_to_ras_transform(direction_matrix):
    """
    Step 5: Coordinate Transformation - LPS to RAS
    
    DICOM uses LPS (Left, Posterior, Superior)
    3D Slicer uses RAS (Right, Anterior, Superior)
    
    Transformation: Flip X and Y axes
    """
    # Create LPS to RAS transformation matrix
    lps_to_ras = np.diag([-1, -1, 1])
    
    # Apply to direction matrix
    ras_direction = lps_to_ras @ direction_matrix
    
    return ras_direction


def load_and_build_volume(series_uid, series_metadata, series_files):
    """
    Step 6: Voxel Filling - Load pixel data and construct 3D volume
    
    Returns vtkImageData object (same as 3D Slicer)
    """
    print(f"\nProcessing Series: {series_metadata['SeriesDescription']}")
    print(f"  Series UID: ...{series_uid[-12:]}")
    print(f"  Number of slices: {len(series_files)}")
    
    # Sort slices by spatial position
    sorted_files = sort_slices_by_position(series_files, series_metadata['ImageOrientationPatient'])
    
    # Build geometry
    spacing, origin, direction_matrix = build_volume_geometry(series_metadata, sorted_files)
    
    print(f"  Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm")
    print(f"  Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
    print(f"  Direction matrix:")
    for row in direction_matrix:
        print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")
    
    # Apply LPS → RAS transformation
    ras_direction = apply_lps_to_ras_transform(direction_matrix)
    ras_origin = np.array([-origin[0], -origin[1], origin[2]])  # Flip X and Y
    
    print(f"  After LPS→RAS transform:")
    print(f"    Origin: [{ras_origin[0]:.2f}, {ras_origin[1]:.2f}, {ras_origin[2]:.2f}] mm")
    
    # Allocate 3D grid
    rows = series_metadata['Rows']
    cols = series_metadata['Columns']
    depth = len(sorted_files)
    
    volume_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    print(f"  Allocated volume: {depth}×{rows}×{cols} = {depth*rows*cols:,} voxels")
    print(f"  Loading pixel data...")
    
    # Fill voxels with Hounsfield Unit correction
    rescale_slope = series_metadata['RescaleSlope']
    rescale_intercept = series_metadata['RescaleIntercept']
    
    for i, file_info in enumerate(sorted_files):
        if (i + 1) % 100 == 0:
            print(f"    Loading slice {i+1}/{depth}...")
        
        dcm = pydicom.dcmread(file_info['filepath'])
        
        # Get pixel data
        pixel_data = dcm.pixel_array.astype(np.float32)
        
        # Apply Hounsfield Unit correction: HU = (PixelValue × Slope) + Intercept
        hu_data = (pixel_data * rescale_slope) + rescale_intercept
        
        volume_array[i, :, :] = hu_data.astype(np.int16)
    
    print(f"  ✓ Volume loaded")
    print(f"    HU range: [{np.min(volume_array)}, {np.max(volume_array)}]")
    
    # Convert to VTK ImageData (same as 3D Slicer's internal format)
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=volume_array.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(cols, rows, depth)  # VTK uses (X, Y, Z) order
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(ras_origin)
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    # Store direction matrix as metadata (VTK doesn't have built-in direction support)
    # In a full implementation, you'd need ITK or a custom transform
    
    return vtk_image, ras_direction


def classify_series_type(series_metadata):
    """
    Classify series by orientation (axial, sagittal, coronal)
    """
    iop = series_metadata['ImageOrientationPatient']
    slice_normal = calculate_slice_normal(iop)
    
    # Check which axis the normal aligns with
    if np.allclose(np.abs(slice_normal), [0, 0, 1], atol=0.3):
        return "AXIAL"
    elif np.allclose(np.abs(slice_normal), [1, 0, 0], atol=0.3):
        return "SAGITTAL"
    elif np.allclose(np.abs(slice_normal), [0, 1, 0], atol=0.3):
        return "CORONAL"
    else:
        return "OBLIQUE"


def render_volume(vtk_image, title):
    """
    Render the volume using VTK (same approach as 3D Slicer)
    """
    print(f"\n  Rendering volume...")
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
    # Transfer functions (opacity and color)
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
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print(f"  Opening viewer...\n")
    render_window.Render()
    interactor.Start()


def main():
    """
    Main pipeline - replicates 3D Slicer's exact workflow
    """
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Step 1: Scan and group by SeriesInstanceUID
    series_groups, series_metadata = scan_dicom_folder(dataset_path)
    
    if len(series_groups) == 0:
        print("✗ No DICOM series found")
        return
    
    # Display all series found
    print("Step 2: Series Summary")
    print(f"{'#':<4} | {'Type':<10} | {'Description':<50} | {'Slices':<7}")
    print("-" * 100)
    
    series_list = []
    for i, (uid, files) in enumerate(sorted(series_groups.items(), key=lambda x: len(x[1]), reverse=True), 1):
        metadata = series_metadata[uid]
        series_type = classify_series_type(metadata)
        desc = metadata['SeriesDescription']
        num_slices = len(files)
        
        print(f"{i:<4} | {series_type:<10} | {desc:<50} | {num_slices:<7}")
        
        series_list.append({
            'number': i,
            'uid': uid,
            'type': series_type,
            'metadata': metadata,
            'files': files,
            'num_slices': num_slices
        })
    
    print()
    
    # Step 3: Process primary series (axial acquisitions)
    print("Step 3: Loading Primary Series (Axial acquisitions only)...")
    
    primary_series = [s for s in series_list if s['type'] == 'AXIAL' and s['num_slices'] >= 10]
    
    if len(primary_series) == 0:
        print("✗ No primary axial series found")
        return
    
    print(f"  Found {len(primary_series)} primary series\n")
    
    # Load each series and display one by one
    for series_info in primary_series:
        vtk_volume, direction = load_and_build_volume(
            series_info['uid'],
            series_info['metadata'],
            series_info['files']
        )
        
        title = f"{DATASET} - Series {series_info['number']}: {series_info['metadata']['SeriesDescription']}"
        
        print(f"\n{'='*100}")
        print(f"DISPLAYING: {title}")
        print(f"Close window to continue to next series...")
        print(f"{'='*100}\n")
        
        render_volume(vtk_volume, title)
    
    print(f"\n{'='*100}")
    print(f"✓ ALL SERIES DISPLAYED")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
