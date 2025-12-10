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
from scipy import ndimage


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


def find_best_chunk_match(vol1, vol2, num_chunks=1000, search_range_mm=200):
    """
    Find best 3D offset (X, Y, Z) by directly comparing volumetric chunks.
    
    Strategy:
    1. Divide each volume into chunks evenly along Z-axis
    2. For each chunk pair, compute correlation on tissue voxels
    3. Test different Z-offsets to find where chunks match best
    4. Return offset that maximizes correlation
    
    This allows FREE MOVEMENT and finds the anatomically correct alignment.
    """
    print(f"    Finding best 3D alignment using volumetric chunk correlation...")
    print(f"    Strategy: {num_chunks} chunks, free movement in 3D space")
    
    # Get volume properties
    origin1 = np.array(vol1.GetOrigin())
    origin2 = np.array(vol2.GetOrigin())
    spacing1 = np.array(vol1.GetSpacing())
    spacing2 = np.array(vol2.GetSpacing())
    size1 = np.array(vol1.GetSize())
    size2 = np.array(vol2.GetSize())
    
    arr1 = sitk.GetArrayFromImage(vol1)  # ZYX
    arr2 = sitk.GetArrayFromImage(vol2)
    
    z1_min, z1_max = origin1[2], origin1[2] + (size1[2]-1) * spacing1[2]
    z2_min, z2_max = origin2[2], origin2[2] + (size2[2]-1) * spacing2[2]
    
    z1_height = z1_max - z1_min
    z2_height = z2_max - z2_min
    
    print(f"      Vol1: Z={z1_min:.1f} to {z1_max:.1f} mm ({z1_height:.1f}mm, {size1[2]} slices)")
    print(f"      Vol2: Z={z2_min:.1f} to {z2_max:.1f} mm ({z2_height:.1f}mm, {size2[2]} slices)")
    
    # Determine chunk size - aim for ~50mm chunks for good resolution
    chunk_height_mm = 50.0
    chunk_slices_1 = max(5, int(chunk_height_mm / spacing1[2]))
    chunk_slices_2 = max(5, int(chunk_height_mm / spacing2[2]))
    
    # Calculate actual number of chunks
    actual_chunks_1 = max(1, size1[2] // chunk_slices_1)
    actual_chunks_2 = max(1, size2[2] // chunk_slices_2)
    
    print(f"      Vol1: {chunk_slices_1} slices/chunk → {actual_chunks_1} chunks (~{chunk_slices_1 * spacing1[2]:.1f}mm each)")
    print(f"      Vol2: {chunk_slices_2} slices/chunk → {actual_chunks_2} chunks (~{chunk_slices_2 * spacing2[2]:.1f}mm each)")
    
    # Extract chunks from vol1
    chunks1 = []
    for chunk_idx in range(actual_chunks_1):
        start_idx = chunk_idx * chunk_slices_1
        end_idx = min(start_idx + chunk_slices_1, size1[2])
        
        chunk_data = arr1[start_idx:end_idx, :, :]
        
        # Only include chunks with tissue
        tissue_mask = chunk_data > -500
        tissue_fraction = np.mean(tissue_mask)
        
        if tissue_fraction > 0.01:  # At least 1% tissue
            z_min = origin1[2] + start_idx * spacing1[2]
            z_max = origin1[2] + end_idx * spacing1[2]
            z_center = (z_min + z_max) / 2
            
            # Compute chunk statistics for matching
            tissue_data = chunk_data[tissue_mask]
            
            chunks1.append({
                'data': chunk_data,
                'tissue_mask': tissue_mask,
                'z_min': z_min,
                'z_max': z_max,
                'z_center': z_center,
                'tissue_voxels': np.sum(tissue_mask),
                'mean_hu': np.mean(tissue_data),
                'std_hu': np.std(tissue_data),
                'slice_range': (start_idx, end_idx)
            })
    
    # Extract chunks from vol2
    chunks2 = []
    for chunk_idx in range(actual_chunks_2):
        start_idx = chunk_idx * chunk_slices_2
        end_idx = min(start_idx + chunk_slices_2, size2[2])
        
        chunk_data = arr2[start_idx:end_idx, :, :]
        
        tissue_mask = chunk_data > -500
        tissue_fraction = np.mean(tissue_mask)
        
        if tissue_fraction > 0.01:
            z_min = origin2[2] + start_idx * spacing2[2]
            z_max = origin2[2] + end_idx * spacing2[2]
            z_center = (z_min + z_max) / 2
            
            tissue_data = chunk_data[tissue_mask]
            
            chunks2.append({
                'data': chunk_data,
                'tissue_mask': tissue_mask,
                'z_min': z_min,
                'z_max': z_max,
                'z_center': z_center,
                'tissue_voxels': np.sum(tissue_mask),
                'mean_hu': np.mean(tissue_data),
                'std_hu': np.std(tissue_data),
                'slice_range': (start_idx, end_idx)
            })
    
    print(f"      Extracted {len(chunks1)} tissue chunks from vol1")
    print(f"      Extracted {len(chunks2)} tissue chunks from vol2")
    
    # Test different Z-offsets within search range
    # Try offsets from -search_range to +search_range in 10mm steps
    test_offsets = np.arange(-search_range_mm, search_range_mm + 10, 10.0)
    
    print(f"      Testing {len(test_offsets)} different Z-offsets...")
    
    best_offset = 0.0
    best_correlation = -1.0
    best_matches = 0
    
    for test_offset in test_offsets:
        # Apply this offset to vol2 chunks
        correlations = []
        
        for c2 in chunks2:
            # What would c2's Z-position be with this offset?
            c2_z_with_offset = c2['z_center'] + test_offset
            
            # Find vol1 chunks that would overlap with this position
            for c1 in chunks1:
                # Check if they would be close in Z (within 30mm)
                z_distance = abs(c1['z_center'] - c2_z_with_offset)
                
                if z_distance < 30.0:  # Only compare nearby chunks
                    # Compute correlation
                    d1 = c1['data']
                    d2 = c2['data']
                    m1 = c1['tissue_mask']
                    m2 = c2['tissue_mask']
                    
                    # Resize d2 to match d1
                    if d1.shape != d2.shape:
                        zoom_factors = (d1.shape[0] / d2.shape[0], 
                                      d1.shape[1] / d2.shape[1],
                                      d1.shape[2] / d2.shape[2])
                        d2 = ndimage.zoom(d2, zoom_factors, order=1)
                        m2 = ndimage.zoom(m2.astype(float), zoom_factors, order=0) > 0.5
                    
                    # Find common tissue
                    common_tissue = m1 & m2
                    n_common = np.sum(common_tissue)
                    
                    if n_common > 100:  # Need enough overlap
                        d1_tissue = d1[common_tissue]
                        d2_tissue = d2[common_tissue]
                        
                        d1_std = np.std(d1_tissue)
                        d2_std = np.std(d2_tissue)
                        
                        if d1_std > 1e-6 and d2_std > 1e-6:
                            d1_norm = (d1_tissue - np.mean(d1_tissue)) / d1_std
                            d2_norm = (d2_tissue - np.mean(d2_tissue)) / d2_std
                            corr = np.mean(d1_norm * d2_norm)
                            
                            correlations.append({
                                'correlation': corr,
                                'tissue_voxels': n_common,
                                'z_distance': z_distance
                            })
        
        if len(correlations) > 0:
            # Calculate aggregate score for this offset
            avg_corr = np.mean([c['correlation'] for c in correlations])
            num_matches = len(correlations)
            total_tissue = sum([c['tissue_voxels'] for c in correlations])
            
            # Score: correlation quality × number of matches × tissue amount
            score = avg_corr * np.sqrt(num_matches) * np.log1p(total_tissue / 1000)
            
            if score > best_correlation:
                best_correlation = score
                best_offset = test_offset
                best_matches = num_matches
                best_avg_corr = avg_corr
    
    print(f"      → Best Z-offset: {best_offset:+.1f}mm")
    print(f"      → Chunk pairs matched: {best_matches}")
    print(f"      → Average correlation: {best_avg_corr:.4f}")
    print(f"      → Quality score: {best_correlation:.4f}")
    
    return best_offset, best_avg_corr


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
    Stitch two volumes together, removing overlapping parts from vol2.
    Vol1 (older/composite) takes priority in overlap regions.
    
    NEW: Explicitly detects Z-overlap range and completely excludes vol2 data
    from that range, keeping only vol1 data.
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
    
    # Detect Z-overlap range
    z1_min, z1_max = origin1[2], end1[2]
    z2_min, z2_max = origin2[2], end2[2]
    
    overlap_z_min = max(z1_min, z2_min)
    overlap_z_max = min(z1_max, z2_max)
    has_overlap = overlap_z_max > overlap_z_min
    
    if has_overlap:
        overlap_mm = overlap_z_max - overlap_z_min
        print(f"        ⚠️  Z-overlap detected: {overlap_z_min:.1f} to {overlap_z_max:.1f} mm ({overlap_mm:.1f} mm)")
        print(f"        → Keeping vol1 in overlap, excluding vol2 completely in this range")
    else:
        print(f"        No Z-overlap detected")
    
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
    
    # Calculate Z-coordinate for each slice in the union grid
    z_coords = union_origin[2] + np.arange(union_size[2]) * union_spacing[2]
    
    # First, resample vol1 (composite/older) onto union grid
    print(f"        Resampling volume 1 (composite)...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(union_vol)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    
    resampled1 = resampler.Execute(vol1)
    resampled1_array = sitk.GetArrayFromImage(resampled1)
    
    # Place vol1 data (takes priority)
    mask1 = resampled1_array > -900
    union_array[mask1] = resampled1_array[mask1]
    
    vol1_voxels = np.sum(mask1)
    print(f"        Placed {vol1_voxels:,} voxels from vol1")
    
    # Now resample vol2
    print(f"        Resampling volume 2 (new)...")
    resampled2 = resampler.Execute(vol2)
    resampled2_array = sitk.GetArrayFromImage(resampled2)
    
    # Create mask for vol2 data
    mask2 = resampled2_array > -900
    
    if has_overlap:
        # Create Z-based exclusion mask: exclude slices in overlap range
        # z_coords has shape (Z,), we need to broadcast it to match array shape (Z, Y, X)
        z_in_overlap = (z_coords >= overlap_z_min) & (z_coords <= overlap_z_max)
        z_not_in_overlap = ~z_in_overlap
        
        # Broadcast Z-mask to 3D by reshaping to (Z, 1, 1)
        z_mask_3d = z_not_in_overlap[:, np.newaxis, np.newaxis]
        
        # Apply vol2 only where:
        # 1. vol2 has data (mask2)
        # 2. Z is NOT in overlap range (z_mask_3d)
        # 3. vol1 is empty (for safety, though overlap exclusion should handle it)
        empty_in_vol1 = union_array <= -900
        mask2_allowed = mask2 & z_mask_3d & empty_in_vol1
        
        # Count what we're excluding
        vol2_total_voxels = np.sum(mask2)
        vol2_in_overlap = np.sum(mask2 & ~z_mask_3d)
        vol2_added = np.sum(mask2_allowed)
        
        print(f"        Vol2 total voxels: {vol2_total_voxels:,}")
        print(f"        Vol2 voxels in overlap range (excluded): {vol2_in_overlap:,}")
        print(f"        Vol2 voxels added: {vol2_added:,}")
    else:
        # No overlap: just avoid placing vol2 where vol1 already exists
        empty_in_vol1 = union_array <= -900
        mask2_allowed = mask2 & empty_in_vol1
        vol2_added = np.sum(mask2_allowed)
        print(f"        Placed {vol2_added:,} voxels from vol2")
    
    union_array[mask2_allowed] = resampled2_array[mask2_allowed]
    
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
        
        # Find best alignment using volumetric chunk matching with FREE MOVEMENT
        print(f"    Finding alignment using volumetric chunk matching (free movement)...")
        best_offset, best_score = find_best_chunk_match(
            composite, 
            next_vol,
            num_chunks=1000,  # Not used anymore but kept for compatibility
            search_range_mm=1500  # Large search range to allow free movement
        )
        
        print(f"    → Applying Z-offset: {best_offset:+.1f}mm (correlation: {best_score:.4f})")
        
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
    DATASET = 'DICOM-Jan'
    
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
