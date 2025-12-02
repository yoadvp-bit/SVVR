"""
Volume Merging using Translation Registration
==============================================

Merges multiple DICOM series (axial, coronal, sagittal) using:
1. Translation registration to find offset
2. Compute combined volume dimensions
3. Fill non-overlapping regions
4. Average overlapping regions to reduce noise
"""

import os
import numpy as np
import pydicom
from collections import defaultdict
import SimpleITK as sitk
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support


# Configuration
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Maria"


def scan_dicom_folder(folder_path):
    """Scan folder and group files by SeriesInstanceUID"""
    print(f"\nScanning: {os.path.basename(folder_path)}")
    
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
                series_groups[series_uid].append(filepath)
                
                if series_uid not in series_metadata:
                    iop = list(dcm.ImageOrientationPatient)
                    slice_normal = np.cross(np.array(iop[:3]), np.array(iop[3:6]))
                    
                    # Determine orientation
                    if np.allclose(np.abs(slice_normal), [0, 0, 1], atol=0.3):
                        orientation = "AXIAL"
                    elif np.allclose(np.abs(slice_normal), [1, 0, 0], atol=0.3):
                        orientation = "SAGITTAL"
                    elif np.allclose(np.abs(slice_normal), [0, 1, 0], atol=0.3):
                        orientation = "CORONAL"
                    else:
                        orientation = "OBLIQUE"
                    
                    series_metadata[series_uid] = {
                        'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'SeriesNumber': int(getattr(dcm, 'SeriesNumber', 0)),
                        'Orientation': orientation,
                        'PixelSpacing': list(dcm.PixelSpacing),
                        'SliceThickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                    }
                
            except Exception:
                continue
    
    print(f"  ✓ Scanned {files_scanned} files")
    print(f"  ✓ Found {len(series_groups)} unique series\n")
    
    return series_groups, series_metadata


def load_series_as_sitk_image(file_paths):
    """Load DICOM series as SimpleITK image"""
    print(f"    Loading {len(file_paths)} files as SimpleITK image...")
    
    # Sort files by instance number or position
    files_with_meta = []
    for fpath in file_paths:
        try:
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(dcm, 'SliceLocation'):
                files_with_meta.append((fpath, dcm.SliceLocation))
            elif hasattr(dcm, 'InstanceNumber'):
                files_with_meta.append((fpath, dcm.InstanceNumber))
        except:
            continue
    
    files_with_meta.sort(key=lambda x: x[1])
    sorted_files = [f[0] for f in files_with_meta]
    
    # Use SimpleITK's DICOM reader
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    image = reader.Execute()
    
    print(f"    ✓ Image size: {image.GetSize()}")
    print(f"    ✓ Spacing: {image.GetSpacing()}")
    print(f"    ✓ Origin: {image.GetOrigin()}")
    
    return image


def perform_translation_registration(fixed_image, moving_image):
    """
    Perform translation-only registration
    Returns: transform, registered image
    """
    print(f"\n  Performing translation registration...")
    print(f"    Fixed image size: {fixed_image.GetSize()}")
    print(f"    Moving image size: {moving_image.GetSize()}")
    
    # Initialize registration
    registration = sitk.ImageRegistrationMethod()
    
    # Use translation transform only
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration.SetInitialTransform(initial_transform)
    
    # Metric: Mean Squares (good for same-modality registration)
    registration.SetMetricAsMeanSquares()
    
    # Optimizer: Gradient Descent
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Multi-resolution framework (optional, speeds up registration)
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Execute registration
    print(f"    Running registration...")
    final_transform = registration.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32)
    )
    
    print(f"    ✓ Registration complete")
    print(f"    Final metric value: {registration.GetMetricValue():.4f}")
    print(f"    Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    
    # Get translation parameters
    translation = final_transform.GetParameters()
    print(f"    Translation (x, y, z): ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}) mm")
    
    return final_transform, translation


def compute_merged_volume_dimensions(fixed_image, moving_image, translation):
    """
    Compute dimensions of merged volume based on translation
    """
    print(f"\n  Computing merged volume dimensions...")
    
    fixed_size = np.array(fixed_image.GetSize())
    moving_size = np.array(moving_image.GetSize())
    fixed_spacing = np.array(fixed_image.GetSpacing())
    moving_spacing = np.array(moving_image.GetSpacing())
    
    # Translation in voxels
    translation_voxels = np.array(translation) / fixed_spacing
    
    print(f"    Translation in voxels: {translation_voxels}")
    
    # Compute bounds
    # Fixed image: from (0,0,0) to fixed_size
    # Moving image: from translation_voxels to translation_voxels + moving_size
    
    min_coords = np.minimum([0, 0, 0], translation_voxels)
    max_coords_fixed = fixed_size
    max_coords_moving = translation_voxels + moving_size
    max_coords = np.maximum(max_coords_fixed, max_coords_moving)
    
    merged_size = np.ceil(max_coords - min_coords).astype(int)
    merged_origin = np.array(fixed_image.GetOrigin()) + min_coords * fixed_spacing
    
    print(f"    Merged size: {merged_size}")
    print(f"    Merged origin: {merged_origin}")
    
    return merged_size, merged_origin, min_coords, translation_voxels


def merge_volumes_with_averaging(fixed_image, moving_image, final_transform):
    """
    Merge two volumes:
    - Non-overlapping regions: use original values
    - Overlapping regions: average to reduce noise
    """
    print(f"\n  Merging volumes...")
    
    # Get translation parameters
    translation = final_transform.GetParameters()
    
    # Compute merged dimensions
    merged_size, merged_origin, min_coords, translation_voxels = compute_merged_volume_dimensions(
        fixed_image, moving_image, translation
    )
    
    # Create empty merged volume
    merged_image = sitk.Image(merged_size.tolist(), fixed_image.GetPixelID())
    merged_image.SetOrigin(merged_origin.tolist())
    merged_image.SetSpacing(fixed_image.GetSpacing())
    merged_image.SetDirection(fixed_image.GetDirection())
    
    # Convert to numpy for easier manipulation (use float32 for averaging)
    merged_array = sitk.GetArrayFromImage(merged_image).astype(np.float32)
    fixed_array = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
    moving_array = sitk.GetArrayFromImage(moving_image).astype(np.float32)
    
    # Create count arrays to track overlap
    count_array = np.zeros_like(merged_array, dtype=np.float32)
    
    print(f"    Placing fixed image...")
    # Place fixed image
    fixed_offset = (-min_coords).astype(int)
    fixed_slice = tuple([
        slice(fixed_offset[2], fixed_offset[2] + fixed_array.shape[0]),
        slice(fixed_offset[1], fixed_offset[1] + fixed_array.shape[1]),
        slice(fixed_offset[0], fixed_offset[0] + fixed_array.shape[2])
    ])
    
    merged_array[fixed_slice] += fixed_array
    count_array[fixed_slice] += 1
    
    print(f"    Placing moving image (with translation)...")
    # Place moving image with translation
    moving_offset = (translation_voxels - min_coords).astype(int)
    moving_slice = tuple([
        slice(moving_offset[2], moving_offset[2] + moving_array.shape[0]),
        slice(moving_offset[1], moving_offset[1] + moving_array.shape[1]),
        slice(moving_offset[0], moving_offset[0] + moving_array.shape[2])
    ])
    
    merged_array[moving_slice] += moving_array
    count_array[moving_slice] += 1
    
    # Average overlapping regions
    print(f"    Averaging overlapping regions...")
    overlap_mask = count_array > 1
    merged_array[overlap_mask] /= count_array[overlap_mask]
    
    overlap_voxels = np.sum(overlap_mask)
    total_voxels = np.prod(merged_array.shape)
    print(f"    Overlap: {overlap_voxels} voxels ({overlap_voxels/total_voxels*100:.1f}%)")
    
    # Convert back to SimpleITK (cast back to int16 for DICOM compatibility)
    merged_array_int = merged_array.astype(np.int16)
    merged_image = sitk.GetImageFromArray(merged_array_int)
    merged_image.SetOrigin(merged_origin.tolist())
    merged_image.SetSpacing(fixed_image.GetSpacing())
    merged_image.SetDirection(fixed_image.GetDirection())
    
    print(f"    ✓ Merged volume created: {merged_image.GetSize()}")
    
    return merged_image


def visualize_merged_volume_vtk(merged_image, title):
    """Visualize merged volume using interactive VTK 3D rendering"""
    print(f"\n  Creating interactive VTK 3D visualization...")
    
    merged_array = sitk.GetArrayFromImage(merged_image)
    spacing = merged_image.GetSpacing()
    origin = merged_image.GetOrigin()
    
    print(f"    Volume shape (Z,Y,X): {merged_array.shape}")
    print(f"    Intensity range: [{np.min(merged_array):.0f}, {np.max(merged_array):.0f}]")
    print(f"    Spacing: {spacing}")
    print(f"    Origin: {origin}")
    
    # Convert SimpleITK (Z,Y,X) to VTK (X,Y,Z) format
    # SimpleITK uses (slices, rows, cols) while VTK uses (x, y, z)
    vtk_array = np.transpose(merged_array, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
    
    print(f"    VTK array shape (X,Y,Z): {vtk_array.shape}")
    
    # Convert numpy array to VTK using numpy_support
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=vtk_array.ravel(order='F'),  # Fortran order for VTK
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    # Create VTK image data
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(vtk_array.shape[0], vtk_array.shape[1], vtk_array.shape[2])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    print(f"    ✓ VTK image created: {vtk_array.shape[0]}×{vtk_array.shape[1]}×{vtk_array.shape[2]}")
    
    # Create volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    
    # Create volume property with transfer functions
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    # Opacity transfer function
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(-1000, 0.0)    # Air: transparent
    opacity_tf.AddPoint(-500, 0.0)     # Air/soft tissue boundary
    opacity_tf.AddPoint(-200, 0.1)     # Soft tissue: slightly visible
    opacity_tf.AddPoint(0, 0.2)        # Water
    opacity_tf.AddPoint(200, 0.5)      # Bone: more visible
    opacity_tf.AddPoint(1000, 0.8)     # Dense bone
    opacity_tf.AddPoint(3000, 1.0)     # Very dense
    
    # Color transfer function
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)     # Black (air)
    color_tf.AddRGBPoint(-500, 0.2, 0.1, 0.1)      # Dark red (soft tissue)
    color_tf.AddRGBPoint(0, 0.9, 0.6, 0.6)         # Pink (tissue)
    color_tf.AddRGBPoint(200, 1.0, 1.0, 0.9)       # Ivory (bone)
    color_tf.AddRGBPoint(1000, 1.0, 1.0, 1.0)      # White (dense bone)
    
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)
    
    # Create volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    # Add axes widget for orientation
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(100, 100, 100)
    
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    
    # Add text annotation
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(f"{title} - Merged Volume\nLeft click + drag: Rotate\nRight click + drag: Zoom\nMiddle click + drag: Pan")
    text_actor.GetTextProperty().SetFontSize(18)
    text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    text_actor.SetPosition(10, 10)
    renderer.AddActor2D(text_actor)
    
    renderer.ResetCamera()
    
    # Create render window (INTERACTIVE)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
    render_window.SetWindowName(f"{title} - Merged CT Volume (Interactive)")
    
    # Create interactor for mouse/keyboard interaction
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Set up the orientation marker widget
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes_widget.EnabledOn()
    axes_widget.InteractiveOff()
    
    # Use trackball camera style for intuitive interaction
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    print(f"    ✓ Volume rendering configured")
    print(f"    ✓ Interactive window created")
    print(f"\n{'='*80}")
    print(f"  INTERACTIVE CONTROLS:")
    print(f"  - Left click + drag: Rotate volume")
    print(f"  - Right click + drag: Zoom in/out")
    print(f"  - Middle click + drag: Pan camera")
    print(f"  - Mouse wheel: Zoom")
    print(f"  - 'r': Reset camera")
    print(f"  - 'q' or close window: Exit")
    print(f"{'='*80}\n")
    
    # Initialize and start interaction
    interactor.Initialize()
    render_window.Render()
    interactor.Start()
    
    print(f"    ✓ Interactive visualization closed")


def main():
    """Main function"""
    print("="*80)
    print("VOLUME MERGING USING TRANSLATION REGISTRATION")
    print("="*80)
    
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Scan folder
    series_groups, series_metadata = scan_dicom_folder(dataset_path)
    
    if len(series_groups) == 0:
        print("✗ No DICOM series found")
        return
    
    # Display available series
    print("Available series:")
    print(f"{'#':<4} | {'Type':<10} | {'Description':<40} | {'Slices':<7}")
    print("-" * 80)
    
    series_list = []
    for i, (uid, files) in enumerate(sorted(series_groups.items(), 
                                            key=lambda x: series_metadata[x[0]]['SeriesNumber']), 1):
        metadata = series_metadata[uid]
        desc = metadata['SeriesDescription'][:40]
        orientation = metadata['Orientation']
        num_slices = len(files)
        
        print(f"{i:<4} | {orientation:<10} | {desc:<40} | {num_slices:<7}")
        
        series_list.append({
            'number': i,
            'uid': uid,
            'metadata': metadata,
            'files': files,
            'num_slices': num_slices
        })
    
    # Select all AXIAL series to merge (only use axial to avoid dimension mismatches)
    axial_series = [s for s in series_list if s['metadata']['Orientation'] == 'AXIAL']
    
    if len(axial_series) == 0:
        print(f"\n✗ No axial series found")
        return
    
    print(f"\n✓ Found {len(axial_series)} axial series to merge")
    
    # Filter to unique series (remove duplicates based on description and slice count)
    unique_series = {}
    for s in axial_series:
        key = (s['metadata']['SeriesDescription'], s['num_slices'])
        if key not in unique_series:
            unique_series[key] = s
    
    axial_series = list(unique_series.values())
    print(f"✓ After removing duplicates: {len(axial_series)} unique series")
    
    if len(axial_series) < 2:
        print(f"\n⚠ Only {len(axial_series)} unique series found - loading single series")
        fixed_series = axial_series[0]
        print(f"\n{'='*80}")
        print(f"LOADING SERIES #{fixed_series['number']}: {fixed_series['metadata']['SeriesDescription']}")
        print(f"  Orientation: {fixed_series['metadata']['Orientation']}")
        print(f"  Slices: {fixed_series['num_slices']}")
        print(f"{'='*80}")
        merged_image = load_series_as_sitk_image(fixed_series['files'])
    else:
        # Sort by number of slices (largest first)
        axial_series_sorted = sorted(axial_series, key=lambda s: s['num_slices'], reverse=True)
        
        # Start with largest series as fixed
        fixed_series = axial_series_sorted[0]
        print(f"\n{'='*80}")
        print(f"FIXED IMAGE (Series #{fixed_series['number']}): {fixed_series['metadata']['SeriesDescription']}")
        print(f"  Orientation: {fixed_series['metadata']['Orientation']}")
        print(f"  Slices: {fixed_series['num_slices']}")
        print(f"{'='*80}")
        
        merged_image = load_series_as_sitk_image(fixed_series['files'])
        
        # Iteratively merge each remaining series
        for i, moving_series in enumerate(axial_series_sorted[1:], 1):
            print(f"\n{'='*80}")
            print(f"MERGING SERIES {i}/{len(axial_series_sorted)-1}")
            print(f"Series #{moving_series['number']}: {moving_series['metadata']['SeriesDescription']}")
            print(f"  Orientation: {moving_series['metadata']['Orientation']}")
            print(f"  Slices: {moving_series['num_slices']}")
            print(f"{'='*80}")
            
            # Load moving image
            moving_image = load_series_as_sitk_image(moving_series['files'])
            
            # Perform registration
            final_transform, translation = perform_translation_registration(merged_image, moving_image)
            
            # Merge volumes (merged becomes the new fixed image)
            merged_image = merge_volumes_with_averaging(merged_image, moving_image, final_transform)
            
            print(f"  ✓ Series {i} merged into combined volume")
    
    # Visualize with VTK 3D rendering
    visualize_merged_volume_vtk(merged_image, DATASET)
    
    # Save merged volume
    output_file = f"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/scripts3/{DATASET}_merged.nii"
    sitk.WriteImage(merged_image, output_file)
    print(f"\n  ✓ Saved merged volume: {output_file}")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
