"""
NIfTI Volume Stitcher using SimpleITK Registration
===================================================

Automatically stitch multiple axial NIfTI volumes using:
1. Intensity-based registration (faster than isosurface)
2. Multi-resolution approach for robustness
3. Overlap removal to prevent double visualization
4. Direct NIfTI I/O

Based on SimpleITK Image Registration framework.
"""

import SimpleITK as sitk
import numpy as np
import os
import vtk
from vtk.util import numpy_support

# === CONFIGURATION ===
DATASET = "DICOM-Maria"
INPUT_DIR = rf"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/nii_exports/{DATASET}"
OUTPUT_PATH = rf"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/nii_exports/{DATASET}/{DATASET}_stitched.nii"

# MODE: 'concatenate' = just stack volumes, 'surface_match' = match surfaces, 'spatial_merge' = use physical positions
STITCH_MODE = 'spatial_merge'  # Respect actual spatial positioning


def load_nifti_volumes(folder_path):
    """Load all NIfTI files from folder"""
    print(f"\n{'='*80}")
    print(f"NIFTI VOLUME STITCHER - {DATASET}")
    print(f"{'='*80}\n")
    
    print("Step 1: Loading NIfTI volumes...")
    
    files = sorted([f for f in os.listdir(folder_path) 
                    if f.endswith('.nii') and not f.endswith('_stitched.nii')])
    
    if not files:
        print(f"ERROR: No .nii files found in {folder_path}")
        return []
    
    volumes = []
    for idx, fname in enumerate(files, 1):
        path = os.path.join(folder_path, fname)
        
        try:
            img = sitk.ReadImage(path)
            
            # Get spatial info
            origin = img.GetOrigin()
            size = img.GetSize()
            spacing = img.GetSpacing()
            
            # Calculate Z extent
            z_min = origin[2]
            z_max = origin[2] + (size[2] - 1) * spacing[2]
            
            volumes.append({
                'filename': fname,
                'image': img,
                'origin': origin,
                'size': size,
                'spacing': spacing,
                'z_min': z_min,
                'z_max': z_max,
                'z_center': (z_min + z_max) / 2
            })
            
            print(f"  [{idx}] {fname}")
            print(f"      Size: {size[0]}×{size[1]}×{size[2]} voxels")
            print(f"      Z: {z_min:.1f} to {z_max:.1f} mm ({z_max - z_min:.1f} mm height)")
            
        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")
            continue
    
    print(f"\n  ✓ Loaded {len(volumes)} volumes\n")
    return volumes


def filter_duplicates(volumes):
    """Remove duplicate volumes based on Z-overlap"""
    print("Step 2: Filtering duplicate volumes...")
    
    unique = []
    for vol in volumes:
        is_duplicate = False
        
        for u in unique:
            # Calculate overlap
            overlap_min = max(vol['z_min'], u['z_min'])
            overlap_max = min(vol['z_max'], u['z_max'])
            overlap = max(0, overlap_max - overlap_min)
            
            vol_height = vol['z_max'] - vol['z_min']
            
            # If >95% overlap, consider EXACT duplicate (not just similar Z-range)
            if overlap > 0.95 * vol_height:
                print(f"  Dropping duplicate: {vol['filename']}")
                print(f"    → {overlap:.1f}mm overlap ({overlap/vol_height*100:.0f}%) with {u['filename']}")
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(vol)
            print(f"  ✓ Keeping: {vol['filename']}")
    
    print(f"\n  Result: {len(unique)} unique volumes\n")
    return unique


def register_volumes(fixed_vol, moving_vol):
    """
    Register two volumes using SimpleITK multi-resolution registration
    Returns: transform to apply to moving volume
    """
    print(f"  Registering: {moving_vol['filename']} → {fixed_vol['filename']}")
    
    fixed_img = fixed_vol['image']
    moving_img = moving_vol['image']
    
    # Convert to float for registration (required by SimpleITK)
    fixed_img_float = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving_img_float = sitk.Cast(moving_img, sitk.sitkFloat32)
    
    # Initialize registration method
    registration = sitk.ImageRegistrationMethod()
    
    # Similarity metric: Correlation works well for CT-CT registration
    registration.SetMetricAsCorrelation()
    
    # Optimizer: Regular step gradient descent
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.01,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-4
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Multi-resolution pyramid
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Initial transform: Translation only (Z-axis dominant for axial stacks)
    initial_transform = sitk.TranslationTransform(3)
    
    # Initialize with approximate Z-offset based on physical positions
    if 'z_min' in moving_vol and 'z_max' in moving_vol:
        # Calculate approximate offset (bottom of fixed to top of moving)
        fixed_origin = fixed_img.GetOrigin()
        fixed_size = fixed_img.GetSize()
        fixed_spacing = fixed_img.GetSpacing()
        fixed_z_min = fixed_origin[2]
        
        z_offset = fixed_z_min - moving_vol['z_max']
        initial_transform.SetOffset([0, 0, z_offset])
    else:
        # For composite, just use identity
        initial_transform.SetOffset([0, 0, 0])
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Execute registration
    try:
        final_transform = registration.Execute(fixed_img_float, moving_img_float)
        
        # Get final offset
        offset = final_transform.GetParameters()
        metric_value = registration.GetMetricValue()
        
        print(f"    Offset: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}] mm")
        print(f"    Metric: {metric_value:.4f}")
        print(f"    Iterations: {registration.GetOptimizerIteration()}")
        
        return final_transform
        
    except Exception as e:
        print(f"    ⚠️  Registration failed: {e}")
        print(f"    Using initial transform only")
        return initial_transform


def stitch_with_overlap_removal(vol1, vol2, transform):
    """
    Stitch two volumes, removing overlap from vol2
    vol1 takes priority in overlapping regions
    """
    print(f"    Stitching volumes...")
    
    # Resample vol2 using the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(vol1)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)  # Air HU value
    
    vol2_transformed = resampler.Execute(vol2)
    
    # Calculate union bounds
    origin1 = np.array(vol1.GetOrigin())
    origin2 = np.array(vol2_transformed.GetOrigin())
    
    size1 = np.array(vol1.GetSize())
    size2 = np.array(vol2_transformed.GetSize())
    
    spacing1 = np.array(vol1.GetSpacing())
    spacing2 = np.array(vol2_transformed.GetSpacing())
    
    end1 = origin1 + (size1 - 1) * spacing1
    end2 = origin2 + (size2 - 1) * spacing2
    
    # Union bounds
    union_origin = np.minimum(origin1, origin2)
    union_end = np.maximum(end1, end2)
    union_spacing = np.minimum(spacing1, spacing2)
    union_size = ((union_end - union_origin) / union_spacing + 1).astype(int)
    
    # Create union volume
    union_vol = sitk.Image(union_size.tolist(), sitk.sitkInt16)
    union_vol.SetOrigin(union_origin.tolist())
    union_vol.SetSpacing(union_spacing.tolist())
    union_vol.SetDirection(vol1.GetDirection())
    
    # Fill with air
    union_array = np.full(union_size[::-1], -1000, dtype=np.int16)  # ZYX order
    
    # Resample vol1 onto union grid
    resampler.SetReferenceImage(union_vol)
    resampler.SetTransform(sitk.Transform())  # Identity
    resampled1 = resampler.Execute(vol1)
    array1 = sitk.GetArrayFromImage(resampled1)
    
    # Place vol1 (takes priority)
    mask1 = array1 > -900
    union_array[mask1] = array1[mask1]
    
    # Resample vol2_transformed onto union grid
    resampled2 = resampler.Execute(vol2_transformed)
    array2 = sitk.GetArrayFromImage(resampled2)
    
    # Place vol2 only where vol1 is empty
    mask2 = array2 > -900
    empty_in_vol1 = union_array <= -900
    mask2_non_overlap = mask2 & empty_in_vol1
    
    union_array[mask2_non_overlap] = array2[mask2_non_overlap]
    
    overlap_voxels = np.sum(mask2 & ~empty_in_vol1)
    if overlap_voxels > 0:
        print(f"    Removed {overlap_voxels:,} overlapping voxels")
    
    # Create final image
    final_img = sitk.GetImageFromArray(union_array)
    final_img.CopyInformation(union_vol)
    
    final_size = final_img.GetSize()
    final_origin = final_img.GetOrigin()
    final_spacing = final_img.GetSpacing()
    final_height = final_size[2] * final_spacing[2]
    
    print(f"    ✓ Result: {final_size[0]}×{final_size[1]}×{final_size[2]} voxels ({final_height:.1f}mm height)")
    
    return final_img


def extract_isosurface_from_sitk(sitk_img, iso_value=-500):
    """Extract isosurface from SimpleITK image using VTK marching cubes"""
    # Convert to VTK
    arr = sitk.GetArrayFromImage(sitk_img)
    
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_img.GetSize())
    vtk_image.SetSpacing(sitk_img.GetSpacing())
    vtk_image.SetOrigin(sitk_img.GetOrigin())
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    # Marching cubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, iso_value)
    marching_cubes.Update()
    
    surface = marching_cubes.GetOutput()
    
    # Smooth
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(10)
    smoother.Update()
    
    return smoother.GetOutput()


def sample_surface_slices(surface, num_slices=50):
    """Sample surface at regular Z intervals"""
    bounds = surface.GetBounds()
    z_min, z_max = bounds[4], bounds[5]
    
    if z_max <= z_min:
        return []
    
    z_positions = np.linspace(z_min, z_max, num_slices)
    slices = []
    
    for z in z_positions:
        # Extract points near this Z
        points = []
        for i in range(surface.GetNumberOfPoints()):
            pt = surface.GetPoint(i)
            if abs(pt[2] - z) < 5.0:  # Within 5mm of slice
                points.append(pt)
        
        if len(points) > 10:  # Need enough points
            slices.append({
                'z': z,
                'points': np.array(points),
                'count': len(points)
            })
    
    return slices


def compare_slice_shapes(slice1, slice2):
    """Compare two slices and return similarity score (0-1)"""
    if len(slice1['points']) < 10 or len(slice2['points']) < 10:
        return 0.0
    
    # Calculate centroids
    centroid1 = np.mean(slice1['points'], axis=0)
    centroid2 = np.mean(slice2['points'], axis=0)
    
    # Calculate radial distances from centroid
    def get_radii(points, centroid):
        return np.sqrt(np.sum((points - centroid)**2, axis=1))
    
    radii1 = get_radii(slice1['points'], centroid1)
    radii2 = get_radii(slice2['points'], centroid2)
    
    # Compare distributions
    mean1, mean2 = np.mean(radii1), np.mean(radii2)
    std1, std2 = np.std(radii1), np.std(radii2)
    
    # Similarity based on mean radius ratio
    if mean1 == 0 or mean2 == 0:
        return 0.0
    
    mean_ratio = min(mean1, mean2) / max(mean1, mean2)
    
    # Similarity based on std ratio
    if std1 == 0 or std2 == 0:
        std_ratio = 0.5
    else:
        std_ratio = min(std1, std2) / max(std1, std2)
    
    # Combined score
    score = 0.6 * mean_ratio + 0.4 * std_ratio
    
    return score


def find_best_surface_match_exhaustive(surface1, surface2, is_connecting_parts=True):
    """
    Find best Z-offset between two surfaces using exhaustive slice comparison
    
    For connecting body parts (torso to legs), we should match:
    - Bottom slices of surface1 (torso) 
    - Top slices of surface2 (legs)
    """
    print(f"    Extracting surface slices...")
    
    slices1 = sample_surface_slices(surface1, num_slices=50)
    slices2 = sample_surface_slices(surface2, num_slices=50)
    
    if len(slices1) == 0 or len(slices2) == 0:
        print(f"    ⚠️  Failed to extract slices")
        return 0.0, 0.0
    
    if is_connecting_parts:
        # For body parts: match bottom 25% of volume1 with top 25% of volume2
        boundary_slices1 = int(len(slices1) * 0.75)  # Bottom 25%
        boundary_slices2 = int(len(slices2) * 0.25)  # Top 25%
        
        search_slices1 = slices1[boundary_slices1:]  # Bottom slices of torso
        search_slices2 = slices2[:boundary_slices2]   # Top slices of legs
        
        print(f"    Searching boundary regions: {len(search_slices1)} (torso bottom) × {len(search_slices2)} (legs top) = {len(search_slices1)*len(search_slices2)} pairs")
    else:
        search_slices1 = slices1
        search_slices2 = slices2
        print(f"    Comparing {len(slices1)}×{len(slices2)} = {len(slices1)*len(slices2)} slice pairs...")
    
    best_offset = 0.0
    best_quality = 0.0
    matches_found = []
    
    # Try all possible Z-alignments in the search region
    for i, s1 in enumerate(search_slices1):
        for j, s2 in enumerate(search_slices2):
            # Calculate Z-offset that would align these slices
            z_offset = s1['z'] - s2['z']
            
            # Calculate similarity
            similarity = compare_slice_shapes(s1, s2)
            
            if similarity > 0.5:  # Only consider good matches
                matches_found.append({
                    'offset': z_offset,
                    'similarity': similarity,
                    'z1': s1['z'],
                    'z2': s2['z']
                })
                
                if similarity > best_quality:
                    best_quality = similarity
                    best_offset = z_offset
    
    if matches_found:
        print(f"    Found {len(matches_found)} good matches (similarity > 0.5)")
        # Show top 3 matches
        matches_found.sort(key=lambda x: x['similarity'], reverse=True)
        for idx, match in enumerate(matches_found[:3], 1):
            print(f"      #{idx}: offset={match['offset']:+.1f}mm, similarity={match['similarity']:.4f}")
    
    print(f"    → Best offset: {best_offset:+.1f}mm, similarity: {best_quality:.4f}")
    
    return best_offset, best_quality


def stitch_with_surface_matching(volumes):
    """Stitch volumes using isosurface matching"""
    print("Step 3: Stitching with surface matching...\n")
    
    if len(volumes) == 0:
        return None
    if len(volumes) == 1:
        return volumes[0]['image']
    
    # Sort by Z (top to bottom)
    volumes.sort(key=lambda v: v['z_center'], reverse=True)
    
    print("  Processing volumes:")
    for idx, vol in enumerate(volumes, 1):
        print(f"    {idx}. {vol['filename']}")
    print()
    
    # Start with first volume
    composite = volumes[0]['image']
    print(f"  Base: {volumes[0]['filename']}\n")
    
    # Add each subsequent volume
    for idx, vol_data in enumerate(volumes[1:], 2):
        print(f"  Step {idx}/{len(volumes)}: Adding {vol_data['filename']}")
        
        # Extract isosurfaces
        print(f"    Extracting isosurfaces...")
        surface_composite = extract_isosurface_from_sitk(composite, iso_value=-500)
        surface_next = extract_isosurface_from_sitk(vol_data['image'], iso_value=-500)
        
        print(f"    Surfaces: {surface_composite.GetNumberOfPoints():,} and {surface_next.GetNumberOfPoints():,} vertices")
        
        # Find best match
        z_offset, similarity = find_best_surface_match_exhaustive(surface_composite, surface_next)
        
        # Apply Z-translation to next volume
        print(f"    Applying Z-offset: {z_offset:+.1f}mm")
        
        next_vol_aligned = vol_data['image']
        origin = list(next_vol_aligned.GetOrigin())
        origin[2] += z_offset
        next_vol_aligned.SetOrigin(origin)
        
        # Stitch with overlap removal
        composite = stitch_with_overlap_removal(composite, next_vol_aligned, sitk.Transform())
        print()
    
    print(f"  ✓ Stitched all {len(volumes)} volumes!\n")
    return composite


def concatenate_volumes(volumes):
    """
    Simple concatenation: stack all volumes along Z-axis
    No registration, just put them side by side to see what we have
    """
    print("Step 3: Concatenating volumes (no registration)...\n")
    
    if len(volumes) == 0:
        print("ERROR: No volumes to concatenate")
        return None
    
    if len(volumes) == 1:
        print("Only one volume")
        return volumes[0]['image']
    
    # Sort by Z-position (superior to inferior, head to feet)
    volumes.sort(key=lambda v: v['z_center'], reverse=True)
    
    print("  Volume order (top to bottom):")
    for idx, vol in enumerate(volumes, 1):
        print(f"    {idx}. {vol['filename']}: Z={vol['z_min']:.1f} to {vol['z_max']:.1f} mm")
    
    # Convert all to numpy arrays
    arrays = []
    for vol in volumes:
        arr = sitk.GetArrayFromImage(vol['image'])  # Z, Y, X
        arrays.append(arr)
        print(f"\n  Volume {vol['filename']}")
        print(f"    Shape: {arr.shape}")
    
    # Concatenate along Z-axis (axis=0)
    print(f"\n  Concatenating {len(arrays)} volumes along Z-axis...")
    combined_array = np.concatenate(arrays, axis=0)
    
    print(f"  Combined shape: {combined_array.shape}")
    
    # Create new image with combined data
    # Use spacing and direction from first volume
    first_vol = volumes[0]['image']
    combined_img = sitk.GetImageFromArray(combined_array)
    
    # Set metadata from first volume
    combined_img.SetSpacing(first_vol.GetSpacing())
    combined_img.SetOrigin(first_vol.GetOrigin())
    combined_img.SetDirection(first_vol.GetDirection())
    
    print(f"  ✓ Concatenated all {len(volumes)} volumes!")
    print(f"    Total slices: {combined_array.shape[0]}\n")
    
    return combined_img


def stitch_all_volumes(volumes):
    """Incrementally stitch all volumes"""
    print("Step 3: Stitching volumes with registration...\n")
    
    if len(volumes) == 0:
        print("ERROR: No volumes to stitch")
        return None
    
    if len(volumes) == 1:
        print("Only one volume, no stitching needed")
        return volumes[0]['image']
    
    # Sort by Z-position (top to bottom)
    volumes.sort(key=lambda v: v['z_center'], reverse=True)
    
    # Start with first volume
    composite = volumes[0]['image']
    print(f"  Base: {volumes[0]['filename']}\n")
    
    # Incrementally add each volume
    for idx, vol_data in enumerate(volumes[1:], 2):
        print(f"  Step {idx}/{len(volumes)}:")
        
        # Register moving to composite
        transform = register_volumes(
            {'image': composite, 'filename': 'composite'},
            vol_data
        )
        
        # Stitch with overlap removal
        composite = stitch_with_overlap_removal(
            composite,
            vol_data['image'],
            transform
        )
        print()
    
    print(f"  ✓ Stitched all {len(volumes)} volumes!\n")
    return composite


def render_volume(sitk_img, title):
    """Render volume with VTK"""
    print("Rendering 3D volume...")
    
    # Convert to numpy
    arr = sitk.GetArrayFromImage(sitk_img)
    
    # Create VTK image
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_img.GetSize())
    vtk_image.SetSpacing(sitk_img.GetSpacing())
    vtk_image.SetOrigin(sitk_img.GetOrigin())
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
    # Transfer functions
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-500, 0.0)
    opacity.AddPoint(-200, 0.05)
    opacity.AddPoint(200, 0.3)
    opacity.AddPoint(500, 0.6)
    opacity.AddPoint(1500, 0.8)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-500, 0.5, 0.3, 0.2)
    color.AddRGBPoint(0, 0.9, 0.7, 0.6)
    color.AddRGBPoint(500, 1.0, 0.9, 0.9)
    color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
    
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
    
    # Set camera to view from front with head at top
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0, 0, 1)  # Z-axis points up (superior direction)
    camera.SetPosition(0, -1000, 0)  # Look from front (negative Y)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()
    
    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 1000)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    render_window.Render()
    interactor.Start()


def stitch_spatially(volumes):
    """
    Stitch volumes by respecting their actual spatial positions.
    This merges volumes that overlap in physical space, with vol1 taking priority.
    """
    print("Step 3: Stitching with spatial merge (physical positioning)...\n")
    
    if len(volumes) == 0:
        return None
    if len(volumes) == 1:
        return volumes[0]['image']
    
    # Sort by Z position (head to feet, top to bottom)
    volumes.sort(key=lambda v: v['z_center'], reverse=True)
    
    print("  Processing volumes:")
    for idx, vol in enumerate(volumes, 1):
        info = vol
        print(f"    {idx}. {info['filename']}")
        print(f"       Z-range: {info['z_min']:.1f} to {info['z_max']:.1f} mm")
    print()
    
    # Start with first volume
    composite = volumes[0]['image']
    composite_origin = np.array(composite.GetOrigin())
    composite_spacing = np.array(composite.GetSpacing())
    composite_size = np.array(composite.GetSize())
    
    z_min = composite_origin[2]
    z_max = composite_origin[2] + composite_size[2] * composite_spacing[2]
    
    print(f"  Base volume: {volumes[0]['filename']}")
    print(f"    Initial Z-range: {z_min:.1f} to {z_max:.1f} mm\n")
    
    # Add each subsequent volume
    for idx in range(1, len(volumes)):
        vol2_info = volumes[idx]
        vol2 = vol2_info['image']
        
        print(f"  Step {idx}/{len(volumes)-1}: Adding {vol2_info['filename']}")
        
        # Volume 2's spatial extent
        vol2_origin = np.array(vol2.GetOrigin())
        vol2_spacing = np.array(vol2.GetSpacing())
        vol2_size = np.array(vol2.GetSize())
        vol2_z_min = vol2_origin[2]
        vol2_z_max = vol2_origin[2] + vol2_size[2] * vol2_spacing[2]
        
        print(f"    Vol2 Z-range: {vol2_z_min:.1f} to {vol2_z_max:.1f} mm")
        
        # Check if volumes overlap or are adjacent
        overlap_start = max(z_min, vol2_z_min)
        overlap_end = min(z_max, vol2_z_max)
        overlap_mm = overlap_end - overlap_start
        
        if overlap_mm > 0:
            print(f"    Overlap detected: {overlap_mm:.1f} mm")
        elif vol2_z_max < z_min:
            gap = z_min - vol2_z_max
            print(f"    Gap detected: {gap:.1f} mm above current bottom")
        else:
            gap = vol2_z_min - z_max
            print(f"    Gap detected: {gap:.1f} mm below current bottom")
        
        # Calculate union bounds
        new_z_min = min(z_min, vol2_z_min)
        new_z_max = max(z_max, vol2_z_max)
        
        # Use finest spacing
        final_spacing = np.minimum(composite_spacing, vol2_spacing)
        
        # Calculate new size
        new_z_size = int(np.ceil((new_z_max - new_z_min) / final_spacing[2]))
        new_size = (composite_size[0], composite_size[1], new_z_size)
        
        new_origin = composite_origin.copy()
        new_origin[2] = new_z_min
        
        print(f"    Creating union canvas: {new_size[0]}×{new_size[1]}×{new_size[2]} voxels")
        print(f"    New Z-range: {new_z_min:.1f} to {new_z_max:.1f} mm")
        
        # Create empty canvas
        union = sitk.Image(int(new_size[0]), int(new_size[1]), int(new_size[2]), sitk.sitkInt16)
        union.SetOrigin(tuple(new_origin))
        union.SetSpacing(tuple(final_spacing))
        union.SetDirection(composite.GetDirection())
        
        # Fill with air
        union_array = sitk.GetArrayFromImage(union)
        union_array.fill(-1024)
        
        # Resample composite to union grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(union)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)
        
        print(f"    Placing existing volume...")
        composite_resampled = resampler.Execute(composite)
        composite_array = sitk.GetArrayFromImage(composite_resampled)
        
        # Place composite (takes priority)
        mask_composite = composite_array > -900  # Non-air voxels
        union_array[mask_composite] = composite_array[mask_composite]
        
        print(f"    Placing new volume...")
        vol2_resampled = resampler.Execute(vol2)
        vol2_array = sitk.GetArrayFromImage(vol2_resampled)
        
        # Place vol2 only where union is still air
        mask_vol2 = (vol2_array > -900) & (union_array <= -900)
        voxels_added = np.sum(mask_vol2)
        union_array[mask_vol2] = vol2_array[mask_vol2]
        
        print(f"    Added {voxels_added:,} new voxels from vol2")
        
        # Convert back to SimpleITK
        union = sitk.GetImageFromArray(union_array)
        union.SetOrigin(tuple(new_origin))
        union.SetSpacing(tuple(final_spacing))
        union.SetDirection(composite.GetDirection())
        
        # Update for next iteration
        composite = union
        z_min = new_z_min
        z_max = new_z_max
        
        final_height = new_z_size * final_spacing[2]
        print(f"    ✓ Result: {new_size[0]}×{new_size[1]}×{new_size[2]} voxels ({final_height:.1f}mm height)\n")
    
    print(f"  ✓ Stitched all {len(volumes)} volumes!")
    
    return composite


def main():
    # Load volumes
    volumes = load_nifti_volumes(INPUT_DIR)
    if not volumes:
        return
    
    # Filter duplicates
    unique_volumes = filter_duplicates(volumes)
    if not unique_volumes:
        return
    
    # Stitch based on mode
    if STITCH_MODE == 'concatenate':
        stitched = concatenate_volumes(unique_volumes)
    elif STITCH_MODE == 'surface_match':
        stitched = stitch_with_surface_matching(unique_volumes)
    elif STITCH_MODE == 'spatial_merge':
        stitched = stitch_spatially(unique_volumes)
    else:
        stitched = stitch_all_volumes(unique_volumes)
    
    if stitched is None:
        return
    
    # Save result
    print("Step 4: Saving stitched volume...")
    sitk.WriteImage(stitched, OUTPUT_PATH)
    
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    final_size = stitched.GetSize()
    final_origin = stitched.GetOrigin()
    final_spacing = stitched.GetSpacing()
    final_height = final_size[2] * final_spacing[2]
    
    print(f"  ✓ Saved: {os.path.basename(OUTPUT_PATH)} ({file_size_mb:.2f} MB)")
    print(f"    Dimensions: {final_size[0]}×{final_size[1]}×{final_size[2]} voxels")
    print(f"    Height: {final_height:.1f} mm")
    print(f"    Full path: {OUTPUT_PATH}\n")
    
    # Visualize
    print(f"{'='*80}")
    print("RENDERING STITCHED VOLUME")
    print(f"{'='*80}\n")
    
    render_volume(stitched, f"{DATASET} - Stitched Volume")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
