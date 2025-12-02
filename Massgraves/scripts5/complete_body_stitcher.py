"""
Complete Body Volume Stitcher
==============================
Combines all axial DICOM series into a single stitched 3D volume using
3D Slicer's distance-weighted blending logic.

Features:
1. Loads ALL axial series (no filtering)
2. Proper physical space positioning (uses DICOM geometry)
3. Distance-weighted blending in overlap regions
4. Automatic rotation optimization (tests 180° flip)
5. Interactive VTK 3D rendering

Based on logic from:
- 3D Slicer's StitchVolumes module
- seperate_groups.py (DICOM loading)
"""

import os
import numpy as np
import pydicom
from collections import defaultdict
import vtk
from vtk.util import numpy_support
import SimpleITK as sitk

# === CONFIGURATION ===
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Maria"  # Options: DICOM-Maria, DICOM-Jan, DICOM-Jarek

# === PART 1: DICOM LOADING (From seperate_groups.py) ===

def scan_dicom_folder(folder_path):
    """Scan and group DICOM files by SeriesInstanceUID"""
    print(f"\n{'='*100}")
    print(f"COMPLETE BODY STITCHER: {os.path.basename(folder_path)}")
    print(f"{'='*100}\n")
    
    print("Step 1: Scanning DICOM files...")
    
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
                        'ImageOrientationPatient': list(dcm.ImageOrientationPatient),
                        'PixelSpacing': list(dcm.PixelSpacing),
                        'SliceThickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                        'Rows': int(dcm.Rows),
                        'Columns': int(dcm.Columns),
                        'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1.0)),
                        'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', 0.0)),
                    }
            except:
                continue
    
    print(f"  → Scanned {files_scanned} files")
    print(f"  → Found {len(series_groups)} unique series\n")
    
    return series_groups, series_metadata


def calculate_slice_normal(image_orientation):
    """Calculate slice normal from ImageOrientationPatient"""
    row_direction = np.array(image_orientation[:3])
    col_direction = np.array(image_orientation[3:6])
    slice_normal = np.cross(row_direction, col_direction)
    slice_normal = slice_normal / np.linalg.norm(slice_normal)
    return slice_normal


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


def sort_slices_by_position(files_list, image_orientation):
    """Sort slices by projection onto slice normal"""
    slice_normal = calculate_slice_normal(image_orientation)
    
    for file_info in files_list:
        position = np.array(file_info['position'])
        projection = np.dot(position, slice_normal)
        file_info['projection'] = projection
    
    files_list.sort(key=lambda x: x['projection'])
    return files_list


def load_series_to_sitk(series_uid, series_metadata, series_files):
    """Load a DICOM series as SimpleITK image with proper geometry"""
    desc = series_metadata['SeriesDescription']
    print(f"\n  Loading: {desc} ({len(series_files)} slices)")
    
    # Sort slices
    sorted_files = sort_slices_by_position(series_files, series_metadata['ImageOrientationPatient'])
    
    # Get file paths
    file_paths = [f['filepath'] for f in sorted_files]
    
    # Use SimpleITK ImageSeriesReader for proper geometry handling
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_paths)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
        
        # Get physical bounds
        origin = image.GetOrigin()
        size = image.GetSize()
        spacing = image.GetSpacing()
        
        # Calculate end point
        end_point = [origin[i] + (size[i] - 1) * spacing[i] for i in range(3)]
        
        print(f"    Origin: [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}] mm")
        print(f"    Size: {size[0]} × {size[1]} × {size[2]} voxels")
        print(f"    Spacing: [{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}] mm")
        print(f"    Physical extent Z: {origin[2]:.1f} to {end_point[2]:.1f} mm")
        
        return image
    except Exception as e:
        print(f"    ERROR loading series: {e}")
        return None


# === PART 2: ROTATION OPTIMIZATION ===

def rotate_volume_180_z(img):
    """Rotate volume 180° around Z-axis"""
    size = img.GetSize()
    center = img.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2.0 for sz in size]
    )
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0, 0, np.pi)  # 180° around Z
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    
    return resampler.Execute(img)


def evaluate_orientation_quality(img):
    """
    Evaluate orientation quality using center of mass and symmetry.
    Higher score = better orientation.
    """
    arr = sitk.GetArrayFromImage(img)
    body_mask = arr > -500
    
    if not body_mask.any():
        return 0.0
    
    # Center of mass
    coords = np.argwhere(body_mask)
    com = coords.mean(axis=0)
    volume_center = np.array(arr.shape) / 2.0
    distance = np.linalg.norm(com - volume_center) / np.linalg.norm(volume_center)
    
    # Left/right symmetry (axial plane)
    projection = body_mask.sum(axis=0)
    mid_x = projection.shape[0] // 2
    left_half = projection[:mid_x, :].sum()
    right_half = projection[mid_x:, :].sum()
    
    if left_half + right_half == 0:
        symmetry = 0
    else:
        symmetry = 1.0 - abs(left_half - right_half) / (left_half + right_half)
    
    score = (1.0 - distance) * 0.5 + symmetry * 0.5
    return score


def optimize_rotation(img, desc):
    """Test original and 180° rotated, return better one"""
    print(f"    Testing rotation...")
    
    score_original = evaluate_orientation_quality(img)
    print(f"      Original: {score_original:.3f}")
    
    img_rotated = rotate_volume_180_z(img)
    score_rotated = evaluate_orientation_quality(img_rotated)
    print(f"      Rotated 180°: {score_rotated:.3f}")
    
    if score_rotated > score_original:
        print(f"      → Using ROTATED")
        return img_rotated
    else:
        print(f"      → Using ORIGINAL")
        return img


# === PART 3: AUTOMATIC ALIGNMENT OPTIMIZATION ===

def pad_array_to_512(arr):
    """
    Pad array in X and Y dimensions to 512x512 if smaller.
    Does not stretch - adds empty voxels (-1000 HU) instead.
    """
    current_shape = arr.shape  # (Z, Y, X) for numpy arrays
    
    if current_shape[1] >= 512 and current_shape[2] >= 512:
        return arr
    
    target_y = max(512, current_shape[1])
    target_x = max(512, current_shape[2])
    
    # Create padded array filled with air
    padded = np.full((current_shape[0], target_y, target_x), -1000, dtype=arr.dtype)
    
    # Center the original data
    y_start = (target_y - current_shape[1]) // 2
    x_start = (target_x - current_shape[2]) // 2
    
    padded[:, y_start:y_start+current_shape[1], x_start:x_start+current_shape[2]] = arr
    
    return padded


def calculate_overlap_quality(img1, img2, z_offset):
    """
    Calculate quality of overlap between two volumes at given Z offset.
    
    Hybrid scoring:
    1. Prefer good anatomical overlap (high correlation) over touching
    2. If no overlap, reward touching as fallback
    3. Penalize bad overlaps (air inside body)
    
    Returns:
    - score: Higher is better
    - overlap_voxels: Number of overlapping voxels
    """
    # Apply Z offset to img2
    img2_translated = apply_z_translation(img2, z_offset)
    
    # Get arrays
    arr1 = sitk.GetArrayFromImage(img1)
    arr2 = sitk.GetArrayFromImage(img2_translated)
    
    # Pad both arrays to 512x512 in X-Y plane if needed
    arr1_padded = pad_array_to_512(arr1)
    arr2_padded = pad_array_to_512(arr2)
    
    # Get physical geometries for Z-axis overlap calculation
    origin1 = img1.GetOrigin()
    origin2 = img2_translated.GetOrigin()
    spacing1 = img1.GetSpacing()
    spacing2 = img2_translated.GetSpacing()
    size1 = img1.GetSize()
    size2 = img2_translated.GetSize()
    
    # Calculate Z ranges in physical space
    z1_min = origin1[2]
    z1_max = origin1[2] + (size1[2] - 1) * spacing1[2]
    z2_min = origin2[2]
    z2_max = origin2[2] + (size2[2] - 1) * spacing2[2]
    
    # Calculate gap or overlap
    if z2_min > z1_max:
        # Gap between volumes (vol2 is above vol1)
        gap = z2_min - z1_max
        overlap_thickness = 0
        is_touching = True
    elif z1_min > z2_max:
        # Gap between volumes (vol1 is above vol2)
        gap = z1_min - z2_max
        overlap_thickness = 0
        is_touching = True
    else:
        # Overlapping
        gap = 0
        overlap_z_min = max(z1_min, z2_min)
        overlap_z_max = min(z1_max, z2_max)
        overlap_thickness = overlap_z_max - overlap_z_min
        is_touching = False
    
    # CASE 1: No overlap at all
    if is_touching:
        if gap < 50.0:  # Very close touching (within 50mm)
            # This is OK as a fallback, but give low score
            gap_score = max(0.0, 1.0 - (gap / 50.0))
            return gap_score * 0.3, 0  # Low score for touching (0.3 max)
        else:
            # Too far apart
            return 0.0, 0
    
    # CASE 2: Overlapping volumes - calculate detailed score
    if overlap_thickness <= 0:
        return 0.0, 0
    
    # Convert to index space
    z1_start = max(0, int((overlap_z_min - z1_min) / spacing1[2]))
    z1_end = min(arr1_padded.shape[0], int((overlap_z_max - z1_min) / spacing1[2]) + 1)
    z2_start = max(0, int((overlap_z_min - z2_min) / spacing2[2]))
    z2_end = min(arr2_padded.shape[0], int((overlap_z_max - z2_min) / spacing2[2]) + 1)
    
    if z1_end <= z1_start or z2_end <= z2_start:
        return 0.0, 0
    
    # Extract overlap regions
    overlap1 = arr1_padded[z1_start:z1_end, :, :]
    overlap2 = arr2_padded[z2_start:z2_end, :, :]
    
    # Ensure same Z depth (take minimum)
    min_z_slices = min(overlap1.shape[0], overlap2.shape[0])
    overlap1 = overlap1[:min_z_slices, :, :]
    overlap2 = overlap2[:min_z_slices, :, :]
    
    # Now both should have same shape
    if overlap1.shape != overlap2.shape:
        # Final safeguard: crop to common shape
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(overlap1.shape, overlap2.shape))
        overlap1 = overlap1[:min_shape[0], :min_shape[1], :min_shape[2]]
        overlap2 = overlap2[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Create masks (body vs air)
    mask1 = overlap1 > -500
    mask2 = overlap2 > -500
    
    # Find common body region
    common_mask = np.logical_and(mask1, mask2)
    overlap_voxels = np.sum(common_mask)
    
    # Check if there's meaningful anatomical overlap
    if overlap_voxels < 1000:
        # Very little actual body overlap - treat as touching
        return 0.2, overlap_voxels  # Low score
    
    # Calculate normalized cross-correlation in overlap region
    vals1 = overlap1[common_mask].astype(np.float32)
    vals2 = overlap2[common_mask].astype(np.float32)
    
    if len(vals1) == 0:
        return 0.2, 0
    
    # Normalize
    vals1_norm = (vals1 - vals1.mean()) / (vals1.std() + 1e-6)
    vals2_norm = (vals2 - vals2.mean()) / (vals2.std() + 1e-6)
    
    # Correlation (ranges from -1 to 1)
    correlation = np.mean(vals1_norm * vals2_norm)
    
    # Strong penalty for air inside body
    air1_in_body2 = np.logical_and(overlap1 <= -500, mask2)
    air2_in_body1 = np.logical_and(overlap2 <= -500, mask1)
    air_penalty = (np.sum(air1_in_body2) + np.sum(air2_in_body1)) / max(overlap_voxels, 1)
    
    # Overlap thickness analysis
    # Good overlap: 20-150mm (matches edge regions of anatomy)
    # Bad overlap: >300mm (one volume completely inside another)
    if overlap_thickness < 20.0:
        thickness_score = overlap_thickness / 20.0
    elif overlap_thickness <= 150.0:
        thickness_score = 1.0  # Sweet spot
    elif overlap_thickness <= 300.0:
        thickness_score = 1.0 - ((overlap_thickness - 150.0) / 150.0) * 0.5
    else:
        # Huge overlap - probably wrong
        thickness_score = max(0.0, 0.5 - ((overlap_thickness - 300.0) / 500.0))
    
    # Calculate proportion of overlap relative to smaller volume
    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    smaller_volume = min(total1, total2)
    
    if smaller_volume > 0:
        overlap_ratio = overlap_voxels / smaller_volume
        # Penalize if one volume is mostly inside the other (>60% overlap)
        if overlap_ratio > 0.6:
            overlap_penalty = (overlap_ratio - 0.6) * 2.0
        else:
            overlap_penalty = 0.0
    else:
        overlap_penalty = 0.0
    
    # Combined score:
    # 1. High correlation is good (0 to 1)
    # 2. Low air penalty is good (0 to 1 penalty)
    # 3. Good thickness is important (0 to 1 multiplier)
    # 4. Avoid stacking one volume inside another
    
    base_score = max(0.0, correlation)  # Only positive correlations
    quality_score = base_score * thickness_score - (2.0 * air_penalty) - overlap_penalty
    
    # Boost score if correlation is really good
    if correlation > 0.7 and air_penalty < 0.2:
        quality_score *= 1.5  # Reward great anatomical matches
    
    return quality_score, overlap_voxels


def find_optimal_z_offset(img1, img2, initial_offset=0.0, max_iterations=50):
    """
    Find optimal Z offset for img2 relative to img1 using adaptive search.
    
    Returns:
    - best_offset: Optimal Z translation in mm
    - best_score: Quality score at optimal offset
    """
    print(f"      Searching for optimal Z alignment...")
    
    # Get physical extents to determine intelligent search range
    origin1 = img1.GetOrigin()
    origin2 = img2.GetOrigin()
    size1 = img1.GetSize()
    size2 = img2.GetSize()
    spacing1 = img1.GetSpacing()
    spacing2 = img2.GetSpacing()
    
    z1_extent = (size1[2] - 1) * spacing1[2]
    z2_extent = (size2[2] - 1) * spacing2[2]
    
    # Determine search range based on volume extents
    # Search enough to try volume above, below, and overlapping
    search_range = max(2000.0, z1_extent + z2_extent)  # At least 2000mm search
    
    # Initial search with large steps
    step_size = 100.0  # mm - larger steps for exhaustive search
    best_offset = initial_offset
    best_score, best_overlap = calculate_overlap_quality(img1, img2, best_offset)
    
    print(f"        Initial offset: {initial_offset:.1f}mm, score: {best_score:.4f}")
    print(f"        Search range: ±{search_range:.0f}mm with {step_size:.0f}mm steps")
    
    # Coarse search - EXHAUSTIVE across large range
    offsets_to_try = np.arange(initial_offset - search_range, 
                                initial_offset + search_range + step_size, 
                                step_size)
    
    print(f"        Testing {len(offsets_to_try)} positions...")
    for i, offset in enumerate(offsets_to_try):
        score, overlap = calculate_overlap_quality(img1, img2, offset)
        if score > best_score or (score > 0.5 and overlap > best_overlap):
            best_score = score
            best_offset = offset
            best_overlap = overlap
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"          Progress: {i+1}/{len(offsets_to_try)}, best so far: {best_offset:.1f}mm (score={best_score:.4f})")
    
    print(f"        Coarse search: offset={best_offset:.1f}mm, score={best_score:.4f}, overlap={best_overlap}")
    
    # Medium refinement
    step_size = 20.0
    search_window = 200.0
    for offset in np.arange(best_offset - search_window, best_offset + search_window, step_size):
        score, overlap = calculate_overlap_quality(img1, img2, offset)
        if score > best_score:
            best_score = score
            best_offset = offset
            best_overlap = overlap
    
    print(f"        Medium refinement: offset={best_offset:.1f}mm, score={best_score:.4f}")
    
    # Fine search with adaptive step size
    step_size = 5.0
    iterations_without_improvement = 0
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try offsets around current best
        for direction in [-1, 1]:
            test_offset = best_offset + direction * step_size
            score, overlap = calculate_overlap_quality(img1, img2, test_offset)
            
            if score > best_score:
                best_score = score
                best_offset = test_offset
                best_overlap = overlap
                improved = True
                iterations_without_improvement = 0
                break
        
        if improved:
            # Keep current step size
            pass
        else:
            iterations_without_improvement += 1
            # Reduce step size if not improving
            if iterations_without_improvement >= 3:
                step_size *= 0.5
                iterations_without_improvement = 0
                
                if step_size < 0.5:  # Stop if step too small
                    break
    
    print(f"        Final: offset={best_offset:.1f}mm, score={best_score:.4f}")
    
    return best_offset, best_score


def apply_z_translation(img, z_offset):
    """Apply Z translation to a volume"""
    if abs(z_offset) < 0.01:
        return img
    
    new_origin = list(img.GetOrigin())
    new_origin[2] += z_offset
    
    translated = sitk.Image(img)
    translated.SetOrigin(new_origin)
    
    return translated


def merge_volumes_if_aligned(img1, img2, z_offset_threshold=10.0):
    """
    Check if two volumes are essentially the same (duplicates).
    If Z offset is very small and correlation is high, they're duplicates.
    """
    offset, score = find_optimal_z_offset(img1, img2, initial_offset=0.0, max_iterations=20)
    
    # If offset is tiny and correlation is high, they're duplicates
    if abs(offset) < z_offset_threshold and score > 0.8:
        print(f"        → Volumes are duplicates (offset={offset:.1f}mm, score={score:.3f})")
        return True, img1  # Return first volume as merged result
    
    return False, None


def align_all_volumes(images):
    """
    INCREMENTAL alignment strategy: Build composite step-by-step.
    
    Instead of aligning all volumes at once, we:
    1. Start with first volume as base
    2. For each additional volume:
       - Find optimal alignment with current composite
       - Stitch them together into new composite
    3. Continue until all volumes are incorporated
    
    This ensures each addition is optimal relative to what's already built.
    """
    print("\nStep 2c: Incremental Volume Alignment...")
    
    n_volumes = len(images)
    print(f"  Starting with {n_volumes} volumes")
    print(f"  Strategy: Build composite incrementally\n")
    
    if n_volumes == 0:
        return []
    if n_volumes == 1:
        return images
    
    # Start with first volume as base composite
    composite = images[0]
    origin = composite.GetOrigin()
    size = composite.GetSize()
    spacing = composite.GetSpacing()
    z_min = origin[2]
    z_max = origin[2] + (size[2] - 1) * spacing[2]
    
    print(f"  Step 1/{ n_volumes}: Base composite established")
    print(f"    Slices: {size[2]}")
    print(f"    Z extent: {z_min:.1f} to {z_max:.1f} mm")
    
    # Incrementally add each remaining volume
    for step_idx, next_vol in enumerate(images[1:], start=2):
        print(f"\n  Step {step_idx}/{n_volumes}: Adding new volume")
        
        # Get next volume info
        next_origin = next_vol.GetOrigin()
        next_size = next_vol.GetSize()
        next_spacing = next_vol.GetSpacing()
        
        print(f"    Next volume: {next_size[2]} slices, Z: {next_origin[2]:.1f} to {next_origin[2] + (next_size[2]-1)*next_spacing[2]:.1f} mm")
        
        # Find optimal alignment with current composite
        print(f"    Finding optimal alignment with composite...")
        best_offset, best_score = find_optimal_z_offset(composite, next_vol)
        print(f"    → Best offset: {best_offset:+.1f}mm, score: {best_score:.4f}")
        
        # Apply optimal translation
        next_vol_aligned = apply_z_translation(next_vol, best_offset)
        
        # Stitch together
        print(f"    Stitching composite + new volume...")
        composite = stitch_volumes([composite, next_vol_aligned])
        
        # Report new composite stats
        comp_origin = composite.GetOrigin()
        comp_size = composite.GetSize()
        comp_spacing = composite.GetSpacing()
        comp_z_min = comp_origin[2]
        comp_z_max = comp_origin[2] + (comp_size[2] - 1) * comp_spacing[2]
        
        print(f"    ✓ New composite: {comp_size[2]} slices")
        print(f"    Z extent: {comp_z_min:.1f} to {comp_z_max:.1f} mm")
    
    print(f"\n  ✓ Successfully merged all {n_volumes} volumes into single composite!")
    
    return [composite]  # Return as single-element list for compatibility


# === PART 4: DISTANCE-WEIGHTED BLENDING (From 3D Slicer) ===

def create_distance_map(binary_array):
    """
    Create distance map using SimpleITK.
    From 3D Slicer's createDistanceMapArray logic.
    """
    try:
        sitk_img = sitk.GetImageFromArray(binary_array.astype('uint8'))
        # Use SignedMaurerDistanceMap which is more stable for large volumes
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_img, 
            insideIsPositive=True, 
            squaredDistance=False, 
            useImageSpacing=False
        )
        return sitk.GetArrayFromImage(distance_map)
    except Exception as e:
        print(f"      Warning: Distance map calculation failed, using fallback")
        # Fallback: simple erosion-based distance approximation
        from scipy import ndimage
        if binary_array.any():
            dist = ndimage.distance_transform_edt(binary_array)
        else:
            dist = np.zeros_like(binary_array, dtype=np.float32)
        return dist


def calculate_blend_weights(vol_mask, all_masks, idx):
    """
    Calculate blend weights for a volume.
    From 3D Slicer's getBlendWeightArray logic.
    
    Logic:
    - Weight = 1 in regions unique to this volume
    - Weight = 0 outside this volume
    - Weight = d0/(d0+d1) in overlap regions
      where d0 = distance from outside edge
            d1 = distance from inside edge
    """
    n_images = len(all_masks)
    
    # Find intersection with other volumes
    vol_intersections = np.zeros_like(vol_mask, dtype=bool)
    for other_idx in range(n_images):
        if other_idx != idx:
            intersection = np.logical_and(vol_mask, all_masks[other_idx])
            vol_intersections = np.logical_or(vol_intersections, intersection)
    
    # Three regions:
    vol_alone = np.logical_and(vol_mask, np.logical_not(vol_intersections))
    not_vol = np.logical_not(vol_mask)
    
    # Create distance maps
    d1 = create_distance_map(vol_alone)  # Distance from inside
    d0 = create_distance_map(not_vol)    # Distance from outside
    
    # Calculate weights
    weights = np.zeros(vol_mask.shape, dtype=np.float32)
    weights[vol_alone] = 1.0
    weights[not_vol] = 0.0
    
    # Blend in overlap regions
    if vol_intersections.any():
        d0_masked = d0[vol_intersections]
        d1_masked = d1[vol_intersections]
        weights[vol_intersections] = d0_masked / (d0_masked + d1_masked)
    
    return weights


def create_union_canvas(images):
    """
    Create a canvas that encompasses all input images.
    From 3D Slicer's automatic ROI creation logic.
    """
    print("\nStep 3: Creating Union Canvas...")
    
    # Find bounding box in physical space
    min_phys = list(images[0].GetOrigin())
    max_phys = list(images[0].GetOrigin())
    
    ref_spacing = images[0].GetSpacing()
    
    for img in images:
        sz = img.GetSize()
        # Check all 8 corners
        corners = [
            (0,0,0), (sz[0]-1,0,0), (0,sz[1]-1,0), (0,0,sz[2]-1),
            (sz[0]-1,sz[1]-1,0), (sz[0]-1,0,sz[2]-1), (0,sz[1]-1,sz[2]-1),
            (sz[0]-1,sz[1]-1,sz[2]-1)
        ]
        
        for corner in corners:
            pt = img.TransformIndexToPhysicalPoint(corner)
            for i in range(3):
                if pt[i] < min_phys[i]:
                    min_phys[i] = pt[i]
                if pt[i] > max_phys[i]:
                    max_phys[i] = pt[i]
    
    print(f"  Bounds X: {min_phys[0]:.1f} to {max_phys[0]:.1f} mm")
    print(f"  Bounds Y: {min_phys[1]:.1f} to {max_phys[1]:.1f} mm")
    print(f"  Bounds Z: {min_phys[2]:.1f} to {max_phys[2]:.1f} mm")
    
    # Calculate canvas size
    new_size = [
        int(np.ceil((max_phys[i] - min_phys[i]) / ref_spacing[i]))
        for i in range(3)
    ]
    
    print(f"  Canvas size: {new_size[0]} × {new_size[1]} × {new_size[2]} voxels")
    print(f"  Total voxels: {np.prod(new_size):,}")
    
    # Create canvas
    canvas = sitk.Image(new_size, sitk.sitkFloat32)
    canvas.SetOrigin(min_phys)
    canvas.SetSpacing(ref_spacing)
    canvas.SetDirection((1,0,0, 0,1,0, 0,0,1))
    
    return canvas


def stitch_volumes(images):
    """
    Stitch multiple volumes using distance-weighted blending.
    Main stitching logic from 3D Slicer's blend_volumes.
    """
    print("\nStep 4: Stitching Volumes with Distance-Weighted Blending...")
    
    n_images = len(images)
    
    # Create union canvas
    canvas = create_union_canvas(images)
    canvas_shape = sitk.GetArrayFromImage(canvas).shape
    
    # Prepare resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(canvas)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(sitk.sitkLinear)
    
    # Arrays for accumulation
    vol_arrays = np.zeros((n_images, *canvas_shape), dtype=np.float32)
    vol_masks = np.zeros((n_images, *canvas_shape), dtype=bool)
    
    # Resample all volumes onto canvas
    print("  Resampling volumes onto canvas...")
    for i, img in enumerate(images):
        print(f"    Volume {i+1}/{n_images}...")
        resampled = resampler.Execute(img)
        resampled = sitk.Cast(resampled, sitk.sitkFloat32)
        
        arr = sitk.GetArrayFromImage(resampled)
        vol_arrays[i] = arr
        
        # Create mask (where data exists, > -900 HU)
        vol_masks[i] = arr > -900
    
    # Calculate blend weights for each volume
    print("  Calculating blend weights...")
    weights_all = np.zeros((n_images, *canvas_shape), dtype=np.float32)
    
    for i in range(n_images):
        print(f"    Computing weights for volume {i+1}/{n_images}...")
        weights_all[i] = calculate_blend_weights(vol_masks[i], vol_masks, i)
    
    # Apply weighted average
    print("  Blending volumes...")
    
    # Handle threshold (ignore air)
    threshold = -900
    for i in range(n_images):
        weights_all[i][vol_arrays[i] <= threshold] = 0
    
    # Calculate weight sum
    weight_sum = np.sum(weights_all, axis=0)
    
    # Where sum of weights is zero, use default value
    default_value = -1000
    zero_weight_mask = weight_sum == 0
    
    # For weighted average, temporarily set weights to 1 where sum is zero
    # (will be replaced with default value anyway)
    weights_temp = weights_all.copy()
    weights_temp[0][zero_weight_mask] = 1.0
    
    # Perform weighted average
    blended = np.average(vol_arrays, weights=weights_temp, axis=0)
    
    # Replace zero-weight regions with default value
    blended[zero_weight_mask] = default_value
    
    # Convert to final volume
    print("  Creating final volume...")
    final_img = sitk.GetImageFromArray(blended.astype(np.int16))
    final_img.CopyInformation(canvas)
    
    print(f"  ✓ Stitching complete!")
    print(f"    HU range: [{np.min(blended):.0f}, {np.max(blended):.0f}]")
    
    return final_img


# === PART 4: VTK VISUALIZATION ===

def sitk_to_vtk(sitk_img):
    """Convert SimpleITK image to VTK ImageData"""
    img_np = sitk.GetArrayFromImage(sitk_img)
    
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=img_np.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    dims = sitk_img.GetSize()
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims[0], dims[1], dims[2])
    vtk_img.SetSpacing(sitk_img.GetSpacing())
    vtk_img.SetOrigin(sitk_img.GetOrigin())
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    return vtk_img


def render_stitched_volume(sitk_vol, title):
    """Render the stitched volume using VTK"""
    print("\nStep 5: Rendering 3D Volume...")
    
    vtk_vol = sitk_to_vtk(sitk_vol)
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_vol)
    
    # Transfer functions
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1000, 0.0)
    opacity.AddPoint(-500, 0.0)
    opacity.AddPoint(-100, 0.02)
    opacity.AddPoint(50, 0.15)
    opacity.AddPoint(200, 0.5)
    opacity.AddPoint(500, 0.9)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-500, 0.3, 0.2, 0.1)
    color.AddRGBPoint(-100, 0.8, 0.6, 0.5)
    color.AddRGBPoint(50, 0.9, 0.4, 0.3)
    color.AddRGBPoint(200, 1.0, 0.95, 0.9)
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
    renderer.SetBackground(0.1, 0.1, 0.15)
    renderer.ResetCamera()
    
    # Add orientation axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(100, 100, 100)
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9, 0.5, 0.1)
    widget.SetOrientationMarker(axes)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    
    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1400, 1400)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    widget.SetInteractor(interactor)
    widget.EnabledOn()
    widget.InteractiveOn()
    
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    print("  Opening interactive viewer...")
    print("  (Use mouse to rotate, zoom, and explore)")
    
    render_window.Render()
    interactor.Start()


# === MAIN PIPELINE ===

def main():
    """Main execution pipeline"""
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Step 1: Scan DICOM files
    series_groups, series_metadata = scan_dicom_folder(dataset_path)
    
    if len(series_groups) == 0:
        print("✗ No DICOM series found")
        return
    
    # Display summary
    print("Step 2: Series Summary")
    print(f"{'#':<4} | {'Type':<10} | {'Description':<50} | {'Slices':<7}")
    print("-" * 100)
    
    series_list = []
    for i, (uid, files) in enumerate(sorted(series_groups.items(), 
                                           key=lambda x: len(x[1]), reverse=True), 1):
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
    
    # Filter for axial series with sufficient slices
    axial_series = [s for s in series_list 
                   if s['type'] == 'AXIAL' and s['num_slices'] >= 20]
    
    if len(axial_series) == 0:
        print("\n✗ No axial series found with sufficient slices")
        return
    
    print(f"\n  → Found {len(axial_series)} axial series to stitch")
    
    # Step 2: Load all axial series
    print("\nStep 2b: Loading Axial Series...")
    loaded_images = []
    
    for series_info in axial_series:
        img = load_series_to_sitk(
            series_info['uid'],
            series_info['metadata'],
            series_info['files']
        )
        
        if img is not None:
            # Optimize rotation
            img_optimized = optimize_rotation(img, series_info['metadata']['SeriesDescription'])
            loaded_images.append(img_optimized)
    
    if len(loaded_images) == 0:
        print("\n✗ Failed to load any series")
        return
    
    print(f"\n  ✓ Successfully loaded {len(loaded_images)} volumes")
    
    # Step 2c: Align all volumes intelligently
    aligned_images = align_all_volumes(loaded_images)
    
    if len(aligned_images) == 0:
        print("\n✗ No volumes remaining after alignment")
        return
    
    print(f"\n  ✓ Successfully aligned {len(aligned_images)} volumes")
    
    if len(aligned_images) == 1:
        print("\n  Only 1 volume - rendering directly (no stitching needed)")
        final_volume = aligned_images[0]
    else:
        # Step 3-4: Stitch volumes
        final_volume = stitch_volumes(aligned_images)
    
    # Step 5: Render
    title = f"{DATASET} - Complete Stitched Body"
    render_stitched_volume(final_volume, title)
    
    print(f"\n{'='*100}")
    print(f"✓ COMPLETE!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
