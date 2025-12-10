#!/usr/bin/env python3
"""
Simple Volume Stitcher - Using DICOM Physical Coordinates
==========================================================

Uses the physical coordinate system from DICOM metadata directly.
No artificial alignment - DICOM coordinates are correct.
"""

import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import pydicom
from collections import defaultdict


def scan_dicom_folder(dicom_root):
    """Scan folder and group DICOM files by series"""
    series_files = defaultdict(list)
    series_info = {}
    
    print(f"Scanning {dicom_root}...")
    
    for root, dirs, files in os.walk(dicom_root):
        for filename in files:
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if not hasattr(dcm, 'SeriesInstanceUID'):
                    continue
                
                series_uid = dcm.SeriesInstanceUID
                series_files[series_uid].append(filepath)
                
                if series_uid not in series_info:
                    series_info[series_uid] = {
                        'description': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'orientation': 'UNKNOWN'
                    }
                    
                    if hasattr(dcm, 'ImageOrientationPatient'):
                        iop = dcm.ImageOrientationPatient
                        row_vec = np.array(iop[:3])
                        col_vec = np.array(iop[3:])
                        slice_vec = np.cross(row_vec, col_vec)
                        
                        abs_slice = np.abs(slice_vec)
                        if abs_slice[2] > 0.9:
                            series_info[series_uid]['orientation'] = 'AXIAL'
                        elif abs_slice[0] > 0.9:
                            series_info[series_uid]['orientation'] = 'SAGITTAL'
                        elif abs_slice[1] > 0.9:
                            series_info[series_uid]['orientation'] = 'CORONAL'
                    
            except:
                continue
    
    print(f"  → Found {len(series_files)} unique series")
    return series_files, series_info


def load_series_to_sitk(file_list):
    """Load DICOM series using SimpleITK"""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(file_list))
    return reader.Execute()


def remove_duplicate_volumes(volumes):
    """Remove volumes that are >95% identical in Z-range"""
    print(f"\n  Checking for duplicate volumes...")
    
    unique_volumes = []
    seen_ranges = []
    
    for vol in volumes:
        origin = vol.GetOrigin()
        size = vol.GetSize()
        spacing = vol.GetSpacing()
        z_min = origin[2]
        z_max = origin[2] + (size[2] - 1) * spacing[2]
        
        is_duplicate = False
        for seen_z in seen_ranges:
            # Check if >95% overlap
            overlap_min = max(z_min, seen_z[0])
            overlap_max = min(z_max, seen_z[1])
            
            if overlap_max > overlap_min:
                overlap = overlap_max - overlap_min
                range1 = z_max - z_min
                range2 = seen_z[1] - seen_z[0]
                
                if overlap / min(range1, range2) > 0.95:
                    print(f"    Skipping duplicate: Z={z_min:.1f} to {z_max:.1f} mm")
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_volumes.append(vol)
            seen_ranges.append((z_min, z_max))
    
    print(f"  → Kept {len(unique_volumes)} unique volumes (removed {len(volumes) - len(unique_volumes)} duplicates)")
    return unique_volumes


def stitch_volumes_spatially(volumes):
    """
    Stitch volumes using their physical coordinates from DICOM.
    Removes overlaps by keeping the first volume's data in overlap regions.
    """
    print(f"\n  Stitching {len(volumes)} volumes using physical coordinates...")
    
    if len(volumes) == 0:
        return None
    if len(volumes) == 1:
        return volumes[0]
    
    # Determine union bounds
    union_min = np.array([np.inf, np.inf, np.inf])
    union_max = np.array([-np.inf, -np.inf, -np.inf])
    finest_spacing = np.array([np.inf, np.inf, np.inf])
    
    for vol in volumes:
        origin = np.array(vol.GetOrigin())
        size = np.array(vol.GetSize())
        spacing = np.array(vol.GetSpacing())
        end = origin + (size - 1) * spacing
        
        union_min = np.minimum(union_min, origin)
        union_max = np.maximum(union_max, end)
        finest_spacing = np.minimum(finest_spacing, spacing)
    
    # Create union canvas
    union_size = ((union_max - union_min) / finest_spacing + 1).astype(int)
    
    print(f"    Union bounds:")
    print(f"      X: {union_min[0]:.1f} to {union_max[0]:.1f} mm")
    print(f"      Y: {union_min[1]:.1f} to {union_max[1]:.1f} mm")
    print(f"      Z: {union_min[2]:.1f} to {union_max[2]:.1f} mm")
    print(f"    Union size: {union_size[0]}×{union_size[1]}×{union_size[2]} voxels")
    print(f"    Spacing: {finest_spacing[0]:.3f} × {finest_spacing[1]:.3f} × {finest_spacing[2]:.3f} mm")
    
    union_vol = sitk.Image(union_size.tolist(), sitk.sitkFloat32)
    union_vol.SetOrigin(union_min.tolist())
    union_vol.SetSpacing(finest_spacing.tolist())
    union_vol.SetDirection(volumes[0].GetDirection())
    
    # Fill with air
    union_array = np.full(union_size[::-1], -1000.0, dtype=np.float32)  # ZYX order
    
    # Sort volumes by Z-position (bottom to top)
    volumes_sorted = sorted(volumes, key=lambda v: v.GetOrigin()[2])
    
    print(f"\n    Placing volumes (in Z-order):")
    
    # Resample each volume onto the union grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(union_vol)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    
    for idx, vol in enumerate(volumes_sorted, 1):
        origin = vol.GetOrigin()
        size = vol.GetSize()
        spacing = vol.GetSpacing()
        z_min = origin[2]
        z_max = origin[2] + (size[2] - 1) * spacing[2]
        
        print(f"      Volume {idx}/{len(volumes_sorted)}: Z={z_min:.1f} to {z_max:.1f} mm")
        
        # Resample to union grid
        resampled = resampler.Execute(vol)
        resampled_array = sitk.GetArrayFromImage(resampled)
        
        # Place data where union is still empty (air)
        mask = resampled_array > -900
        empty = union_array <= -900
        place_mask = mask & empty
        
        voxels_placed = np.sum(place_mask)
        union_array[place_mask] = resampled_array[place_mask]
        
        print(f"        → Placed {voxels_placed:,} voxels")
    
    # Create final volume
    final_vol = sitk.GetImageFromArray(union_array)
    final_vol.CopyInformation(union_vol)
    
    print(f"\n  ✓ Stitching complete!")
    
    return final_vol


def render_volume(sitk_image):
    """Render volume with VTK"""
    print("\nRendering 3D Volume...")
    
    # Convert to VTK
    array = sitk.GetArrayFromImage(sitk_image)
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_image.GetSize())
    vtk_image.SetSpacing(sitk_image.GetSpacing())
    vtk_image.SetOrigin(sitk_image.GetOrigin())
    
    vtk_array = numpy_support.numpy_to_vtk(array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    
    # Volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    
    # Transfer functions
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
    color_func.AddRGBPoint(-500, 0.5, 0.3, 0.2)
    color_func.AddRGBPoint(0, 0.9, 0.7, 0.6)
    color_func.AddRGBPoint(500, 1.0, 0.9, 0.9)
    color_func.AddRGBPoint(1500, 1.0, 1.0, 1.0)
    
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(-1000, 0.0)
    opacity_func.AddPoint(-500, 0.0)
    opacity_func.AddPoint(-200, 0.05)
    opacity_func.AddPoint(200, 0.3)
    opacity_func.AddPoint(500, 0.6)
    opacity_func.AddPoint(1500, 0.8)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)
    volume_property.ShadeOn()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
    render_window.SetWindowName("Simple Volume Stitcher - Physical Coordinates")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Camera
    renderer.ResetCamera()
    
    print("  Opening viewer...")
    render_window.Render()
    interactor.Start()


def main():
    print("="*85)
    print("SIMPLE VOLUME STITCHER - USING DICOM PHYSICAL COORDINATES")
    print("="*85)
    
    DATA_PATH = '/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data'
    DATASET = 'DICOM-Maria'
    
    # Step 1: Scan DICOM
    print("\nStep 1: Scanning DICOM files...")
    series_files, series_info = scan_dicom_folder(os.path.join(DATA_PATH, DATASET))
    
    # Step 2: Filter axial
    print("\nStep 2: Filtering axial series...")
    axial_series = []
    for series_uid, info in series_info.items():
        if info['orientation'] == 'AXIAL':
            n_files = len(series_files[series_uid])
            print(f"  Found: {info['description']} ({n_files} slices)")
            axial_series.append((series_uid, n_files, info['description']))
    
    print(f"\n  → Found {len(axial_series)} axial series")
    
    # Step 3: Load volumes
    print("\nStep 3: Loading axial volumes...")
    loaded_volumes = []
    
    for series_uid, n_files, description in axial_series:
        print(f"\n  Loading: {description}")
        try:
            volume = load_series_to_sitk(series_files[series_uid])
            
            origin = volume.GetOrigin()
            size = volume.GetSize()
            spacing = volume.GetSpacing()
            z_extent = (origin[2], origin[2] + (size[2]-1)*spacing[2])
            
            print(f"    Origin: {origin}")
            print(f"    Size: {size}")
            print(f"    Z: {z_extent[0]:.1f} to {z_extent[1]:.1f} mm")
            
            loaded_volumes.append(volume)
        except Exception as e:
            print(f"    ERROR: {e}")
    
    print(f"\n  ✓ Loaded {len(loaded_volumes)} volumes")
    
    # Step 4: Remove duplicates
    unique_volumes = remove_duplicate_volumes(loaded_volumes)
    
    # Step 5: Stitch
    print("\nStep 4: Stitching volumes...")
    final_volume = stitch_volumes_spatially(unique_volumes)
    
    if final_volume is None:
        print("ERROR: Stitching failed!")
        return
    
    final_origin = final_volume.GetOrigin()
    final_size = final_volume.GetSize()
    final_spacing = final_volume.GetSpacing()
    final_z = (final_origin[2], final_origin[2] + (final_size[2]-1)*final_spacing[2])
    
    print(f"\n  Final volume:")
    print(f"    Size: {final_size}")
    print(f"    Z: {final_z[0]:.1f} to {final_z[1]:.1f} mm ({final_z[1]-final_z[0]:.1f}mm total)")
    
    # Step 6: Render
    print("\nStep 5: Rendering...")
    render_volume(final_volume)
    
    print("\n" + "="*85)
    print("✓ COMPLETE!")
    print("="*85)


if __name__ == "__main__":
    main()
