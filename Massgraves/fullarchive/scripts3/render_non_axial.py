"""
3D Slicer-Style DICOM Loader - Non-Axial Series Only
=====================================================

Displays only reconstructed series (CORONAL, SAGITTAL, OBLIQUE)
Excludes primary AXIAL acquisitions
"""

import os
import numpy as np
import pydicom
from collections import defaultdict
import vtk
from vtk.util import numpy_support


# Configuration
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/"
DATASET = "DICOM-Loes"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek", "DICOM-Gerda", "DICOM-Joop", "DICOM-Loes"


def scan_dicom_folder(folder_path):
    """
    Step 1: Database Scan - Group files by SeriesInstanceUID
    """
    print(f"\n{'='*100}")
    print(f"NON-AXIAL SERIES RENDERER: {os.path.basename(folder_path)}")
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
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                series_uid = dcm.SeriesInstanceUID
                
                image_position = list(dcm.ImagePositionPatient)
                
                series_groups[series_uid].append({
                    'filepath': filepath,
                    'position': image_position,
                    'instance_number': int(getattr(dcm, 'InstanceNumber', 0))
                })
                
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
                
            except Exception:
                continue
    
    print(f"  → Scanned {files_scanned} files")
    print(f"  → Found {len(series_groups)} unique series\n")
    
    return series_groups, series_metadata


def calculate_slice_normal(image_orientation):
    """Calculate slice normal vector from ImageOrientationPatient"""
    row_direction = np.array(image_orientation[:3])
    col_direction = np.array(image_orientation[3:6])
    
    slice_normal = np.cross(row_direction, col_direction)
    slice_normal = slice_normal / np.linalg.norm(slice_normal)
    
    return slice_normal


def sort_slices_by_position(files_list, image_orientation):
    """Sort slices by projection onto slice normal"""
    slice_normal = calculate_slice_normal(image_orientation)
    
    for file_info in files_list:
        position = np.array(file_info['position'])
        projection = np.dot(position, slice_normal)
        file_info['projection'] = projection
    
    files_list.sort(key=lambda x: x['projection'])
    
    return files_list


def build_volume_geometry(series_metadata, sorted_files):
    """Construct the Matrix - Calculate physical geometry"""
    pixel_spacing = series_metadata['PixelSpacing']
    
    if len(sorted_files) > 1:
        pos_first = np.array(sorted_files[0]['position'])
        pos_second = np.array(sorted_files[1]['position'])
        z_spacing = np.linalg.norm(pos_second - pos_first)
    else:
        z_spacing = series_metadata['SliceThickness']
    
    spacing = [pixel_spacing[1], pixel_spacing[0], z_spacing]
    origin = sorted_files[0]['position']
    
    iop = series_metadata['ImageOrientationPatient']
    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:6])
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    direction_matrix = np.column_stack([row_cosine, col_cosine, slice_cosine])
    
    return spacing, origin, direction_matrix


def apply_lps_to_ras_transform(direction_matrix):
    """LPS to RAS transformation"""
    lps_to_ras = np.diag([-1, -1, 1])
    ras_direction = lps_to_ras @ direction_matrix
    return ras_direction


def load_and_build_volume(series_uid, series_metadata, series_files):
    """Load pixel data and construct 3D volume"""
    print(f"\nProcessing Series: {series_metadata['SeriesDescription']}")
    print(f"  Series UID: ...{series_uid[-12:]}")
    print(f"  Number of slices: {len(series_files)}")
    
    sorted_files = sort_slices_by_position(series_files, series_metadata['ImageOrientationPatient'])
    
    spacing, origin, direction_matrix = build_volume_geometry(series_metadata, sorted_files)
    
    print(f"  Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm")
    print(f"  Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
    
    ras_direction = apply_lps_to_ras_transform(direction_matrix)
    ras_origin = np.array([-origin[0], -origin[1], origin[2]])
    
    print(f"  After LPS→RAS transform:")
    print(f"    Origin: [{ras_origin[0]:.2f}, {ras_origin[1]:.2f}, {ras_origin[2]:.2f}] mm")
    
    depth = len(sorted_files)
    
    # Get maximum dimensions across all slices (handle varying dimensions)
    print(f"  Checking dimensions across slices...")
    max_rows = 0
    max_cols = 0
    for file_info in sorted_files[:min(10, len(sorted_files))]:  # Check first 10 slices
        dcm = pydicom.dcmread(file_info['filepath'], stop_before_pixels=True)
        max_rows = max(max_rows, dcm.Rows)
        max_cols = max(max_cols, dcm.Columns)
    
    rows, cols = max_rows, max_cols
    
    volume_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    print(f"  Allocated volume: {depth}×{rows}×{cols} = {depth*rows*cols:,} voxels")
    print(f"  Loading pixel data...")
    
    rescale_slope = series_metadata['RescaleSlope']
    rescale_intercept = series_metadata['RescaleIntercept']
    
    for i, file_info in enumerate(sorted_files):
        if (i + 1) % 50 == 0:
            print(f"    Loading slice {i+1}/{depth}...")
        
        dcm = pydicom.dcmread(file_info['filepath'])
        pixel_data = dcm.pixel_array.astype(np.float32)
        hu_data = (pixel_data * rescale_slope) + rescale_intercept
        
        # Handle varying dimensions - center the slice
        slice_rows, slice_cols = hu_data.shape
        row_start = (rows - slice_rows) // 2
        col_start = (cols - slice_cols) // 2
        volume_array[i, row_start:row_start+slice_rows, col_start:col_start+slice_cols] = hu_data.astype(np.int16)
    
    print(f"  ✓ Volume loaded")
    print(f"    HU range: [{np.min(volume_array)}, {np.max(volume_array)}]")
    
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=volume_array.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(cols, rows, depth)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(ras_origin)
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    return vtk_image, ras_direction


def classify_series_type(series_metadata):
    """Classify series by orientation"""
    iop = series_metadata['ImageOrientationPatient']
    slice_normal = calculate_slice_normal(iop)
    
    if np.allclose(np.abs(slice_normal), [0, 0, 1], atol=0.3):
        return "AXIAL"
    elif np.allclose(np.abs(slice_normal), [1, 0, 0], atol=0.3):
        return "SAGITTAL"
    elif np.allclose(np.abs(slice_normal), [0, 1, 0], atol=0.3):
        return "CORONAL"
    else:
        return "OBLIQUE"


def render_volume(vtk_image, title):
    """Render the volume using VTK"""
    print(f"\n  Rendering volume...")
    
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
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
    
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print(f"  Opening viewer...\n")
    render_window.Render()
    interactor.Start()


def main():
    """Main pipeline - render non-axial series only"""
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    series_groups, series_metadata = scan_dicom_folder(dataset_path)
    
    if len(series_groups) == 0:
        print("✗ No DICOM series found")
        return
    
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
    
    # Filter for NON-AXIAL series only
    print("Step 3: Loading Non-Axial Series (CORONAL, SAGITTAL, OBLIQUE)...")
    
    non_axial_series = [s for s in series_list if s['type'] != 'AXIAL' and s['num_slices'] >= 10]
    
    if len(non_axial_series) == 0:
        print("✗ No non-axial series found")
        return
    
    print(f"  Found {len(non_axial_series)} non-axial series:\n")
    for s in non_axial_series:
        print(f"    - {s['type']}: {s['metadata']['SeriesDescription']} ({s['num_slices']} slices)")
    print()
    
    # Load and display each non-axial series
    for series_info in non_axial_series:
        vtk_volume, direction = load_and_build_volume(
            series_info['uid'],
            series_info['metadata'],
            series_info['files']
        )
        
        title = f"{DATASET} - {series_info['type']} - Series {series_info['number']}: {series_info['metadata']['SeriesDescription']}"
        
        print(f"\n{'='*100}")
        print(f"DISPLAYING: {title}")
        print(f"Close window to continue to next series...")
        print(f"{'='*100}\n")
        
        render_volume(vtk_volume, title)
    
    print(f"\n{'='*100}")
    print(f"✓ ALL NON-AXIAL SERIES DISPLAYED")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
