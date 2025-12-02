#!/usr/bin/env python3
"""
Isosurface-Based Volume Stitching
==================================

Strategy:
1. Load first axial volume as base
2. For each additional axial:
   - Extract isosurface (triangle mesh) from both volumes
   - Find regions where surfaces match/overlap
   - Align volumes at best matching surface region
   - Stitch together and merge into single composite
3. Continue until all axials are processed

Uses VTK marching cubes for isosurface extraction and surface matching.
"""

import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import pydicom
from collections import defaultdict


# === PART 1: DICOM LOADING ===

def scan_dicom_folder(dicom_root):
    """Scan folder and group DICOM files by series"""
    series_files = defaultdict(list)
    series_info = {}
    
    print(f"Scanning {dicom_root}...")
    file_count = 0
    
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
                
                # Store metadata
                if series_uid not in series_info:
                    series_info[series_uid] = {
                        'description': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'orientation': 'UNKNOWN'
                    }
                    
                    # Determine orientation
                    if hasattr(dcm, 'ImageOrientationPatient'):
                        iop = dcm.ImageOrientationPatient
                        # Check dominant direction
                        row_vec = np.array(iop[:3])
                        col_vec = np.array(iop[3:])
                        slice_vec = np.cross(row_vec, col_vec)
                        
                        abs_slice = np.abs(slice_vec)
                        if abs_slice[2] > 0.9:  # Z-dominant
                            series_info[series_uid]['orientation'] = 'AXIAL'
                        elif abs_slice[0] > 0.9:  # X-dominant
                            series_info[series_uid]['orientation'] = 'SAGITTAL'
                        elif abs_slice[1] > 0.9:  # Y-dominant
                            series_info[series_uid]['orientation'] = 'CORONAL'
                
                file_count += 1
                if file_count % 500 == 0:
                    print(f"  Scanned {file_count} files...")
                    
            except Exception as e:
                continue
    
    print(f"  → Scanned {file_count} files")
    print(f"  → Found {len(series_files)} unique series")
    
    return series_files, series_info


def load_series_to_sitk(file_list):
    """Load DICOM series using SimpleITK"""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted(file_list))
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    image = reader.Execute()
    return image


# === PART 2: ISOSURFACE EXTRACTION ===

def sitk_to_vtk_image(sitk_image):
    """Convert SimpleITK image to VTK image"""
    array = sitk.GetArrayFromImage(sitk_image)
    
    # Create VTK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_image.GetSize())
    vtk_image.SetSpacing(sitk_image.GetSpacing())
    vtk_image.SetOrigin(sitk_image.GetOrigin())
    
    # Convert array to VTK format (need to transpose from ZYX to XYZ)
    vtk_array = numpy_support.numpy_to_vtk(array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)
    
    return vtk_image


def extract_isosurface(sitk_image, iso_value=-500):
    """
    Extract isosurface (triangle mesh) from volume using marching cubes.
    
    Args:
        sitk_image: SimpleITK image
        iso_value: HU threshold (default -500 for soft tissue boundary)
    
    Returns:
        vtk.vtkPolyData: Triangle mesh surface
    """
    print(f"    Extracting isosurface (threshold={iso_value} HU)...")
    
    # Convert to VTK
    vtk_image = sitk_to_vtk_image(sitk_image)
    
    # Marching cubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, iso_value)
    marching_cubes.Update()
    
    surface = marching_cubes.GetOutput()
    n_points = surface.GetNumberOfPoints()
    n_cells = surface.GetNumberOfCells()
    
    print(f"      → Surface: {n_points} vertices, {n_cells} triangles")
    
    # Smooth surface to reduce noise
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(10)
    smoother.Update()
    
    return smoother.GetOutput()


def get_surface_bounds(surface):
    """Get bounding box of surface"""
    bounds = surface.GetBounds()
    return {
        'x': (bounds[0], bounds[1]),
        'y': (bounds[2], bounds[3]),
        'z': (bounds[4], bounds[5])
    }


# === PART 3: SURFACE MATCHING ===

def sample_surface_slices(surface, axis='z', num_slices=20):
    """
    Sample surface at regular intervals along an axis.
    
    Returns list of point clouds (one per slice).
    """
    bounds = get_surface_bounds(surface)
    
    if axis == 'z':
        axis_idx = 2
        min_val, max_val = bounds['z']
    elif axis == 'y':
        axis_idx = 1
        min_val, max_val = bounds['y']
    else:  # x
        axis_idx = 0
        min_val, max_val = bounds['x']
    
    slice_positions = np.linspace(min_val, max_val, num_slices)
    slice_thickness = (max_val - min_val) / num_slices
    
    slices = []
    
    for slice_z in slice_positions:
        # Extract points near this slice
        points = []
        for i in range(surface.GetNumberOfPoints()):
            pt = surface.GetPoint(i)
            if abs(pt[axis_idx] - slice_z) < slice_thickness / 2:
                points.append(pt)
        
        if len(points) > 10:  # Only keep slices with enough points
            slices.append(np.array(points))
    
    return slices, slice_positions


def compare_slice_shapes(slice1, slice2, tolerance=50.0):
    """
    Compare two surface slices using shape descriptors.
    
    Returns similarity score (higher is better).
    """
    if len(slice1) < 10 or len(slice2) < 10:
        return 0.0
    
    # Compute 2D projections (remove Z coordinate)
    proj1 = slice1[:, :2]  # XY projection
    proj2 = slice2[:, :2]
    
    # Compute centroids
    centroid1 = proj1.mean(axis=0)
    centroid2 = proj2.mean(axis=0)
    
    # Center point clouds
    proj1_centered = proj1 - centroid1
    proj2_centered = proj2 - centroid2
    
    # Compute shape statistics
    # 1. Area (number of points as proxy)
    area1 = len(proj1)
    area2 = len(proj2)
    area_ratio = min(area1, area2) / max(area1, area2)
    
    # 2. Spread (average distance from centroid)
    spread1 = np.mean(np.linalg.norm(proj1_centered, axis=1))
    spread2 = np.mean(np.linalg.norm(proj2_centered, axis=1))
    spread_ratio = min(spread1, spread2) / max(spread1, spread2) if max(spread1, spread2) > 0 else 0
    
    # 3. Shape distribution (histogram of distances)
    dist1 = np.linalg.norm(proj1_centered, axis=1)
    dist2 = np.linalg.norm(proj2_centered, axis=1)
    
    # Normalize to same range
    if dist1.max() > 0:
        dist1 = dist1 / dist1.max()
    if dist2.max() > 0:
        dist2 = dist2 / dist2.max()
    
    # Compare histograms
    hist1, _ = np.histogram(dist1, bins=20, range=(0, 1))
    hist2, _ = np.histogram(dist2, bins=20, range=(0, 1))
    
    # Normalize histograms
    hist1 = hist1.astype(float) / (hist1.sum() + 1e-6)
    hist2 = hist2.astype(float) / (hist2.sum() + 1e-6)
    
    # Histogram correlation
    hist_corr = np.corrcoef(hist1, hist2)[0, 1]
    if np.isnan(hist_corr):
        hist_corr = 0.0
    
    # Combined score
    score = 0.3 * area_ratio + 0.3 * spread_ratio + 0.4 * max(0, hist_corr)
    
    return score


def find_best_surface_match_exhaustive(surface1, surface2):
    """
    Find best Z-offset where surfaces match using exhaustive slice-by-slice comparison.
    
    NEW Strategy:
    1. Extract ALL slices from both surfaces (not just sampled positions)
    2. For each slice in surface1, compare with ALL slices in surface2
    3. Find pairs of slices with highest similarity scores
    4. Determine optimal offset based on best matching slice pairs
    5. If tie, look for multiple consecutive matching slices for best alignment
    """
    print(f"    Finding best surface match (exhaustive slice-by-slice)...")
    
    # Get surface bounds
    bounds1 = get_surface_bounds(surface1)
    bounds2 = get_surface_bounds(surface2)
    
    z1_min, z1_max = bounds1['z']
    z2_min, z2_max = bounds2['z']
    
    print(f"      Surface 1 Z: {z1_min:.1f} to {z1_max:.1f} mm")
    print(f"      Surface 2 Z: {z2_min:.1f} to {z2_max:.1f} mm")
    
    # Sample surfaces at higher resolution for exhaustive comparison
    slices1, positions1 = sample_surface_slices(surface1, axis='z', num_slices=50)
    slices2, positions2 = sample_surface_slices(surface2, axis='z', num_slices=50)
    
    print(f"      Surface 1: {len(slices1)} slices sampled")
    print(f"      Surface 2: {len(slices2)} slices sampled")
    print(f"      Computing pairwise similarity matrix ({len(slices1)}×{len(slices2)} = {len(slices1)*len(slices2)} comparisons)...")
    
    # Build complete similarity matrix: slice1[i] vs slice2[j]
    similarity_matrix = np.zeros((len(slices1), len(slices2)))
    
    for i, slice1 in enumerate(slices1):
        for j, slice2 in enumerate(slices2):
            similarity_matrix[i, j] = compare_slice_shapes(slice1, slice2)
        
        if (i + 1) % 10 == 0:
            print(f"        Progress: {i+1}/{len(slices1)} slices compared")
    
    # Find best matching slice pairs
    print(f"      Analyzing matches...")
    
    # Strategy: For each possible offset (aligning slice i with slice j),
    # calculate total alignment quality
    best_offset = 0.0
    best_total_score = -1.0
    best_num_matches = 0
    
    # Try aligning each slice from surface2 with each slice from surface1
    for i in range(len(positions1)):
        for j in range(len(positions2)):
            # Calculate offset needed to align positions2[j] with positions1[i]
            offset = positions1[i] - positions2[j]
            
            # Calculate positions2 shifted by this offset
            positions2_shifted = positions2 + offset
            
            # Count how many slices overlap and their total similarity
            total_score = 0.0
            num_matches = 0
            matched_scores = []
            
            for k in range(len(positions1)):
                # Find closest shifted position from surface2
                distances = np.abs(positions2_shifted - positions1[k])
                closest_idx = np.argmin(distances)
                
                # If within reasonable tolerance (e.g., 20mm), count as match
                if distances[closest_idx] < 20.0:
                    score = similarity_matrix[k, closest_idx]
                    total_score += score
                    num_matches += 1
                    matched_scores.append(score)
            
            # Evaluate this alignment
            if num_matches > 0:
                avg_score = total_score / num_matches
                
                # Prefer alignments with:
                # 1. High average similarity score
                # 2. Many matching slices (more reliable)
                # Combined metric: avg_score * sqrt(num_matches)
                quality = avg_score * np.sqrt(num_matches)
                
                if quality > best_total_score or (quality == best_total_score and num_matches > best_num_matches):
                    best_total_score = quality
                    best_offset = offset
                    best_num_matches = num_matches
                    best_avg_score = avg_score
    
    print(f"      → Best offset: {best_offset:+.1f}mm")
    print(f"      → Matched slices: {best_num_matches}")
    print(f"      → Average similarity: {best_avg_score:.4f}")
    print(f"      → Quality score: {best_total_score:.4f}")
    
    return best_offset, best_avg_score


# === PART 4: VOLUME ALIGNMENT AND STITCHING ===

def apply_z_translation(img, z_offset):
    """Apply Z translation to a volume"""
    if abs(z_offset) < 0.01:
        return img
    
    new_origin = list(img.GetOrigin())
    new_origin[2] += z_offset
    
    translated = sitk.Image(img)
    translated.SetOrigin(new_origin)
    
    return translated


def stitch_two_volumes(vol1, vol2):
    """
    Stitch two volumes together with simple averaging in overlap regions.
    """
    print(f"      Stitching volumes...")
    
    # Get physical bounds
    origin1 = np.array(vol1.GetOrigin())
    origin2 = np.array(vol2.GetOrigin())
    
    size1 = np.array(vol1.GetSize())
    size2 = np.array(vol2.GetSize())
    
    spacing1 = np.array(vol1.GetSpacing())
    spacing2 = np.array(vol2.GetSpacing())
    
    end1 = origin1 + (size1 - 1) * spacing1
    end2 = origin2 + (size2 - 1) * spacing2
    
    # Union bounds
    union_origin = np.minimum(origin1, origin2)
    union_end = np.maximum(end1, end2)
    
    # Use finest spacing
    union_spacing = np.minimum(spacing1, spacing2)
    
    # Calculate union size
    union_size = ((union_end - union_origin) / union_spacing + 1).astype(int)
    
    print(f"        Union canvas: {union_size[0]}×{union_size[1]}×{union_size[2]} voxels")
    print(f"        Z extent: {union_origin[2]:.1f} to {union_end[2]:.1f} mm")
    
    # Create union volume
    union_vol = sitk.Image(union_size.tolist(), sitk.sitkFloat32)
    union_vol.SetOrigin(union_origin.tolist())
    union_vol.SetSpacing(union_spacing.tolist())
    union_vol.SetDirection(vol1.GetDirection())
    
    # Fill with air
    union_array = np.full(union_size[::-1], -1000.0, dtype=np.float32)  # ZYX order
    
    # Resample both volumes onto union grid
    for idx, vol in enumerate([vol1, vol2], 1):
        print(f"        Resampling volume {idx}...")
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(union_vol)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)
        
        resampled = resampler.Execute(vol)
        resampled_array = sitk.GetArrayFromImage(resampled)
        
        # Blend: take max where not air
        mask = resampled_array > -900
        union_array[mask] = np.maximum(union_array[mask], resampled_array[mask])
    
    # Create final volume
    final_vol = sitk.GetImageFromArray(union_array)
    final_vol.CopyInformation(union_vol)
    
    print(f"        ✓ Stitched")
    
    return final_vol


# === PART 5: INCREMENTAL ISOSURFACE-BASED ALIGNMENT ===

def align_volumes_incremental_isosurface(volumes):
    """
    Incrementally align volumes using isosurface matching.
    
    Process:
    1. Start with first volume
    2. For each new volume:
       - Extract isosurfaces from both
       - Find best surface match
       - Align and stitch
    3. Continue until all merged
    """
    print(f"\n  Incremental Isosurface-Based Alignment")
    print(f"  Starting with {len(volumes)} volumes\n")
    
    if len(volumes) == 0:
        return None
    if len(volumes) == 1:
        return volumes[0]
    
    # Start with first volume
    composite = volumes[0]
    origin = composite.GetOrigin()
    size = composite.GetSize()
    spacing = composite.GetSpacing()
    
    print(f"  Step 1/{len(volumes)}: Base volume established")
    print(f"    Size: {size[0]}×{size[1]}×{size[2]} voxels")
    print(f"    Z: {origin[2]:.1f} to {origin[2] + (size[2]-1)*spacing[2]:.1f} mm")
    
    # Process each additional volume
    for step_idx, next_vol in enumerate(volumes[1:], start=2):
        print(f"\n  Step {step_idx}/{len(volumes)}: Adding new volume")
        
        next_origin = next_vol.GetOrigin()
        next_size = next_vol.GetSize()
        next_spacing = next_vol.GetSpacing()
        
        print(f"    Next volume: {next_size[0]}×{next_size[1]}×{next_size[2]} voxels")
        print(f"    Z: {next_origin[2]:.1f} to {next_origin[2] + (next_size[2]-1)*next_spacing[2]:.1f} mm")
        
        # Extract isosurfaces
        surface_composite = extract_isosurface(composite, iso_value=-500)
        surface_next = extract_isosurface(next_vol, iso_value=-500)
        
        # Find best match using exhaustive slice-by-slice comparison
        best_offset, best_score = find_best_surface_match_exhaustive(
            surface_composite, 
            surface_next
        )
        
        print(f"    Applying offset: {best_offset:+.1f}mm")
        
        # Apply translation
        next_vol_aligned = apply_z_translation(next_vol, best_offset)
        
        # Stitch together
        composite = stitch_two_volumes(composite, next_vol_aligned)
        
        comp_origin = composite.GetOrigin()
        comp_size = composite.GetSize()
        comp_spacing = composite.GetSpacing()
        
        print(f"    ✓ New composite: {comp_size[0]}×{comp_size[1]}×{comp_size[2]} voxels")
        print(f"    Z: {comp_origin[2]:.1f} to {comp_origin[2] + (comp_size[2]-1)*comp_spacing[2]:.1f} mm")
    
    print(f"\n  ✓ Successfully merged all {len(volumes)} volumes!")
    
    return composite


# === PART 6: VISUALIZATION ===

def render_volume(sitk_image):
    """Render volume with VTK"""
    print("\nRendering 3D Volume...")
    
    # Convert to VTK
    vtk_image = sitk_to_vtk_image(sitk_image)
    
    # Create volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    
    # Create transfer functions
    color_func = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Air: black
    color_func.AddRGBPoint(-500, 0.5, 0.3, 0.2)   # Fat: brown
    color_func.AddRGBPoint(0, 0.9, 0.7, 0.6)      # Soft tissue: beige
    color_func.AddRGBPoint(500, 1.0, 0.9, 0.9)    # Bone: white
    color_func.AddRGBPoint(1500, 1.0, 1.0, 1.0)   # Dense bone: white
    
    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(-1000, 0.0)
    opacity_func.AddPoint(-500, 0.0)
    opacity_func.AddPoint(-200, 0.05)
    opacity_func.AddPoint(200, 0.3)
    opacity_func.AddPoint(500, 0.6)
    opacity_func.AddPoint(1500, 0.8)
    
    # Create volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_func)
    volume_property.SetScalarOpacity(opacity_func)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    # Create volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 900)
    render_window.SetWindowName("Isosurface-Based Body Reconstruction")
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Set camera
    renderer.ResetCamera()
    
    # Start
    print("  Opening interactive viewer...")
    render_window.Render()
    interactor.Start()


# === MAIN ===

def main():
    print("="*85)
    print("ISOSURFACE-BASED VOLUME STITCHER")
    print("="*85)
    
    DATA_PATH = '/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data'
    DATASET = 'DICOM-Maria'
    
    # Step 1: Scan DICOM files
    print("\nStep 1: Scanning DICOM files...")
    series_files, series_info = scan_dicom_folder(os.path.join(DATA_PATH, DATASET))
    
    # Step 2: Filter axial series
    print("\nStep 2: Filtering axial series...")
    axial_series = []
    for series_uid, info in series_info.items():
        if info['orientation'] == 'AXIAL':
            n_files = len(series_files[series_uid])
            print(f"  Found: {info['description']} ({n_files} slices)")
            axial_series.append((series_uid, n_files, info['description']))
    
    print(f"\n  → Found {len(axial_series)} axial series")
    
    if len(axial_series) == 0:
        print("ERROR: No axial series found!")
        return
    
    # Step 3: Load axial volumes
    print("\nStep 3: Loading axial volumes...")
    loaded_volumes = []
    
    for series_uid, n_files, description in axial_series:
        print(f"\n  Loading: {description} ({n_files} slices)")
        
        try:
            volume = load_series_to_sitk(series_files[series_uid])
            
            origin = volume.GetOrigin()
            size = volume.GetSize()
            spacing = volume.GetSpacing()
            
            print(f"    Origin: [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}] mm")
            print(f"    Size: {size[0]} × {size[1]} × {size[2]} voxels")
            print(f"    Spacing: [{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}] mm")
            print(f"    Z extent: {origin[2]:.1f} to {origin[2] + (size[2]-1)*spacing[2]:.1f} mm")
            
            loaded_volumes.append(volume)
            
        except Exception as e:
            print(f"    ERROR: Failed to load - {e}")
            continue
    
    print(f"\n  ✓ Loaded {len(loaded_volumes)} volumes")
    
    if len(loaded_volumes) == 0:
        print("ERROR: No volumes loaded!")
        return
    
    # Step 4: Incremental isosurface-based alignment
    print("\nStep 4: Isosurface-Based Alignment...")
    final_volume = align_volumes_incremental_isosurface(loaded_volumes)
    
    if final_volume is None:
        print("ERROR: Alignment failed!")
        return
    
    # Step 5: Render
    print("\nStep 5: Rendering...")
    render_volume(final_volume)
    
    print("\n" + "="*85)
    print("✓ COMPLETE!")
    print("="*85)


if __name__ == "__main__":
    main()
