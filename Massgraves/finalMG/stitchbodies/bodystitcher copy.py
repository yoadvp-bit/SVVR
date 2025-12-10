#!/usr/bin/env python3
"""
IMPROVED BODY STITCHER
======================
Enhanced version of complete_body_stitcher.py with:
1. XY alignment optimization (center of mass + fine search)
2. Multi-scale Z search to avoid local minima
3. Better overlap scoring with surface matching
4. Automatic loading from nifti-ordered folder (1, 2, 3, etc.)

Based on: Massgraves/scripts5/complete_body_stitcher.py
"""

import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to nifti-ordered folder (auto-detect relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALMG_DIR = os.path.dirname(SCRIPT_DIR)  # Parent of stitchbodies = finalMG
NIFTI_ORDERED_BASE = os.path.join(FINALMG_DIR, 'nifti-ordered')

# Bodies to process (names of patient folders in nifti-ordered)
PATIENT_NAMES = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']

OUTPUT_DIR = os.path.join(FINALMG_DIR, 'nifti-stitched')

# IMPORTANT: Set numbering convention
# If you numbered 1=HEAD, 2=MIDDLE, 3=FEET, use REVERSE_NUMBERING=False
# If you numbered 1=FEET, 2=MIDDLE, 3=HEAD, use REVERSE_NUMBERING=True
REVERSE_NUMBERING = True  # Try True if body comes out upside down


# ============================================================================
# FILE LOADING
# ============================================================================

def load_ordered_volumes(patient_name):
    """
    Load ordered NIfTI volumes for a patient from nifti-ordered folder.
    Files are named like: 1_axial_xxx.nii, 2_axial_xxx.nii, etc.
    (Format created by chooseLabelAxials.py: {order_num}_{original_filename})
    Order goes from head (1) to toes (highest number).
    
    Returns:
        dict: {1: path, 2: path, 3: path, ...} sorted by order number
        None if patient folder doesn't exist or has no ordered files
    """
    patient_dir = os.path.join(NIFTI_ORDERED_BASE, patient_name)
    
    if not os.path.exists(patient_dir):
        print(f"  ⚠️  Patient folder not found: {patient_dir}")
        return None
    
    # Find all NIfTI files
    nii_files = glob.glob(os.path.join(patient_dir, '*.nii'))
    
    if not nii_files:
        print(f"  ⚠️  No NIfTI files found in: {patient_dir}")
        return None
    
    # Parse order numbers and create mapping
    volume_map = {}
    skipped_files = []
    for filepath in nii_files:
        filename = os.path.basename(filepath)
        
        # Check if filename starts with a number followed by underscore
        # Format: {order_num}_{original_name}.nii
        if '_' not in filename:
            skipped_files.append(filename)
            continue
            
        try:
            # Extract order number (first part before underscore)
            prefix = filename.split('_')[0]
            order_num = int(prefix)
            
            # Store the path mapped to order number
            if order_num in volume_map:
                print(f"  ⚠️  Warning: Duplicate order number {order_num} found!")
                print(f"      Existing: {os.path.basename(volume_map[order_num])}")
                print(f"      New: {filename}")
                print(f"      Using first file found.")
            else:
                volume_map[order_num] = filepath
                
        except (ValueError, IndexError):
            skipped_files.append(filename)
            continue
    
    if not volume_map:
        print(f"  ⚠️  No valid ordered files found in: {patient_dir}")
        if skipped_files:
            print(f"      Skipped files (missing numeric prefix): {skipped_files}")
        return None
    
    # Sort by order number
    sorted_volumes = dict(sorted(volume_map.items()))
    
    print(f"  ✓ Found {len(sorted_volumes)} ordered volumes:")
    for order_num, path in sorted_volumes.items():
        filename = os.path.basename(path)
        print(f"    {order_num}: {filename}")
    
    if skipped_files:
        print(f"  ℹ️  Skipped {len(skipped_files)} file(s) without numeric prefix")
    
    return sorted_volumes


# ============================================================================
# BED ARTIFACT REMOVAL (Noise Elimination)
# ============================================================================

def remove_bed_artifact(img, bed_threshold=-200):
    """
    Remove CT scanner bed artifact from volume.
    The bed appears as separate connected component below/beside the patient.
    
    Strategy:
    - For each axial slice, find the largest connected component (the patient)
    - Remove everything else (bed is typically a smaller/separate component)
    
    Args:
        img: SimpleITK Image
        bed_threshold: HU threshold to detect solid structures
    
    Returns:
        Cleaned SimpleITK Image with bed removed
    """
    from scipy.ndimage import label
    
    arr = sitk.GetArrayFromImage(img)
    solid_mask = arr > bed_threshold
    cleaned = np.copy(arr)
    
    print(f"      Removing bed artifact (threshold={bed_threshold} HU)...")
    removed_voxels = 0
    
    # Process each axial slice
    for z in range(arr.shape[0]):
        slice_mask = solid_mask[z]
        
        if not slice_mask.any():
            continue
        
        # Label connected components
        labeled, num_features = label(slice_mask)
        
        if num_features <= 1:
            continue  # Only one component, assume it's patient
        
        # Find largest component (patient body)
        component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
        largest_idx = np.argmax(component_sizes) + 1
        patient_mask = (labeled == largest_idx)
        
        # Remove everything else
        removed_mask = slice_mask & (~patient_mask)
        cleaned[z][removed_mask] = -1024
        removed_voxels += removed_mask.sum()
    
    if removed_voxels > 0:
        print(f"      → Removed {removed_voxels:,} voxels (bed/artifacts)")
    
    # Create output image
    cleaned_img = sitk.GetImageFromArray(cleaned)
    cleaned_img.CopyInformation(img)
    
    return cleaned_img


# ============================================================================
# NEW: XY ALIGNMENT FUNCTIONS
# ============================================================================

def get_body_center_of_mass_xy(img, threshold=-300):
    """
    Calculate center of mass in XY plane (ignoring Z).
    Only considers voxels above threshold (body tissue).
    
    Returns: (x_center_mm, y_center_mm) in physical coordinates
    """
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    
    # Create body mask
    body_mask = arr > threshold
    
    if not body_mask.any():
        # Fallback to image center
        size = img.GetSize()
        return (origin[0] + size[0] * spacing[0] / 2,
                origin[1] + size[1] * spacing[1] / 2)
    
    # Get coordinates of body voxels
    # arr shape is (Z, Y, X)
    coords = np.argwhere(body_mask)
    
    # Calculate mean position
    mean_z = coords[:, 0].mean()
    mean_y = coords[:, 1].mean()
    mean_x = coords[:, 2].mean()
    
    # Convert to physical coordinates
    x_phys = origin[0] + mean_x * spacing[0]
    y_phys = origin[1] + mean_y * spacing[1]
    
    return (x_phys, y_phys)


def translate_volume_xy(img, dx_mm, dy_mm):
    """
    Translate volume in XY plane by given amount (in mm).
    Uses resampling to handle sub-voxel shifts.
    """
    if abs(dx_mm) < 0.1 and abs(dy_mm) < 0.1:
        return img
    
    # Create translation transform
    transform = sitk.TranslationTransform(3)
    transform.SetOffset([-dx_mm, -dy_mm, 0])  # Negative because we're moving the grid
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    
    return resampler.Execute(img)


def align_xy_centers(torso_img, legs_img):
    """
    Align legs XY center of mass to match torso XY center of mass.
    Returns translated legs image.
    """
    print("    Aligning XY centers of mass...")
    
    torso_com = get_body_center_of_mass_xy(torso_img)
    legs_com = get_body_center_of_mass_xy(legs_img)
    
    dx = torso_com[0] - legs_com[0]
    dy = torso_com[1] - legs_com[1]
    
    print(f"      Torso COM: ({torso_com[0]:.1f}, {torso_com[1]:.1f}) mm")
    print(f"      Legs COM:  ({legs_com[0]:.1f}, {legs_com[1]:.1f}) mm")
    print(f"      Shift needed: ({dx:.1f}, {dy:.1f}) mm")
    
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        print("      → Already aligned, no shift needed")
        return legs_img
    
    aligned_legs = translate_volume_xy(legs_img, dx, dy)
    print(f"      → Applied XY shift")
    
    return aligned_legs


def fine_tune_xy_alignment(torso_img, legs_img, search_range_mm=30, step_mm=5):
    """
    Fine-tune XY alignment by searching for best overlap score.
    Tests different XY shifts and finds the one with best overlap quality.
    """
    print("    Fine-tuning XY alignment...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    spacing = torso_img.GetSpacing()
    
    # Convert mm to voxels
    search_range_vox = int(search_range_mm / spacing[0])
    step_vox = max(1, int(step_mm / spacing[0]))
    
    # Get overlap region (bottom of torso, top of legs)
    # Use last 20% of torso and first 20% of legs
    torso_slices = int(len(torso_arr) * 0.2)
    legs_slices = int(len(legs_arr) * 0.2)
    
    # Use same number of slices for both
    n_slices = min(torso_slices, legs_slices)
    
    torso_overlap = torso_arr[-n_slices:]
    legs_overlap = legs_arr[:n_slices]
    
    # Create body masks
    torso_mask = torso_overlap > -300
    legs_mask = legs_overlap > -300
    
    best_score = -1
    best_dx = 0
    best_dy = 0
    
    # Search grid
    for dx in range(-search_range_vox, search_range_vox + 1, step_vox):
        for dy in range(-search_range_vox, search_range_vox + 1, step_vox):
            # Shift legs mask
            shifted_legs_mask = np.roll(np.roll(legs_mask, dx, axis=2), dy, axis=1)
            
            # Calculate overlap score (IoU-like)
            intersection = np.logical_and(torso_mask, shifted_legs_mask).sum()
            union = np.logical_or(torso_mask, shifted_legs_mask).sum()
            
            if union > 0:
                score = intersection / union
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
    
    # Convert best shift back to mm
    dx_mm = best_dx * spacing[0]
    dy_mm = best_dy * spacing[1]
    
    print(f"      Best XY shift: ({dx_mm:.1f}, {dy_mm:.1f}) mm, score: {best_score:.4f}")
    
    if abs(dx_mm) < 1.0 and abs(dy_mm) < 1.0:
        return legs_img
    
    return translate_volume_xy(legs_img, dx_mm, dy_mm)


# ============================================================================
# IMPROVED: MULTI-SCALE Z SEARCH (Avoids Local Minima)
# ============================================================================

def calculate_slice_similarity(slice1, slice2, threshold=-300, use_soft_tissue=False):
    """
    Calculate similarity between two slices with adaptive scoring.
    
    Args:
        slice1, slice2: 2D numpy arrays
        threshold: Body detection threshold
        use_soft_tissue: If True, use soft tissue mode (for bone gaps)
    
    Returns:
        Combined score (higher is better)
    """
    from scipy.ndimage import sobel
    
    # Create body masks
    mask1 = slice1 > threshold
    mask2 = slice2 > threshold
    
    if not mask1.any() or not mask2.any():
        return 0.0
    
    # 1. Body IoU
    body_intersection = np.logical_and(mask1, mask2).sum()
    body_union = np.logical_or(mask1, mask2).sum()
    body_iou = body_intersection / body_union if body_union > 0 else 0
    
    # 2. Bone IoU (for normal alignment)
    bone_mask1 = slice1 > 200
    bone_mask2 = slice2 > 200
    bone_intersection = np.logical_and(bone_mask1, bone_mask2).sum()
    bone_union = np.logical_or(bone_mask1, bone_mask2).sum()
    bone_iou = bone_intersection / bone_union if bone_union > 10 else 0
    
    # 3. Correlation of intensities in overlap region
    common = np.logical_and(mask1, mask2)
    if common.sum() > 100:
        vals1 = slice1[common].astype(np.float32)
        vals2 = slice2[common].astype(np.float32)
        
        # Normalize
        vals1_norm = (vals1 - vals1.mean()) / (vals1.std() + 1e-6)
        vals2_norm = (vals2 - vals2.mean()) / (vals2.std() + 1e-6)
        
        correlation = np.mean(vals1_norm * vals2_norm)
        correlation = max(0, correlation)  # Only positive
    else:
        correlation = 0
    
    # 4. Gradient matching (NEW)
    grad1_x = sobel(slice1.astype(np.float32), axis=1)
    grad1_y = sobel(slice1.astype(np.float32), axis=0)
    grad2_x = sobel(slice2.astype(np.float32), axis=1)
    grad2_y = sobel(slice2.astype(np.float32), axis=0)
    
    grad_mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    grad_mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    if common.sum() > 100:
        grad1_common = grad_mag1[common]
        grad2_common = grad_mag2[common]
        
        if grad1_common.std() > 1e-6 and grad2_common.std() > 1e-6:
            grad1_norm = (grad1_common - grad1_common.mean()) / grad1_common.std()
            grad2_norm = (grad2_common - grad2_common.mean()) / grad2_common.std()
            grad_corr = np.mean(grad1_norm * grad2_norm)
            grad_corr = max(0, grad_corr)
        else:
            grad_corr = 0
    else:
        grad_corr = 0
    
    # ADAPTIVE WEIGHTING
    if use_soft_tissue:
        # Soft tissue mode (bone gap): focus on body shape + gradients
        score = 0.50 * body_iou + 0.30 * correlation + 0.20 * grad_corr
    else:
        # Bone mode (normal): balance bone + body + correlation + gradients
        score = 0.30 * bone_iou + 0.30 * body_iou + 0.30 * correlation + 0.10 * grad_corr
    
    return score


def detect_bone_gap(overlap_arr, bone_threshold=200):
    """
    Detect if overlap region has insufficient bone (e.g., rib-pelvis gap).
    
    Args:
        overlap_arr: 3D numpy array (Z, Y, X) of overlap volume
        bone_threshold: HU threshold for bone detection
    
    Returns:
        True if bone gap detected (low bone fraction)
    """
    bone_mask = overlap_arr > bone_threshold
    total_body_voxels = (overlap_arr > -300).sum()
    
    if total_body_voxels < 1000:
        return False  # Insufficient data
    
    bone_fraction = bone_mask.sum() / total_body_voxels
    
    # Bone gap if <5% bone in overlap
    is_gap = bone_fraction < 0.05
    
    if is_gap:
        print(f"      → Bone gap detected (bone fraction: {bone_fraction*100:.1f}%)")
    
    return is_gap


def multi_scale_z_search(torso_img, legs_img, verbose=True):
    """
    Multi-scale search for optimal Z overlap with adaptive scoring.
    
    Returns:
        (overlap_slices, score, confidence)
        - overlap_slices: Number of slices to overlap
        - score: Similarity score
        - confidence: "HIGH" (>=0.5), "MEDIUM" (>=0.3), or "LOW" (<0.3)
    """
    print("    Multi-scale Z overlap search with adaptive scoring...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    
    torso_len = len(torso_arr)
    legs_len = len(legs_arr)
    
    # Maximum overlap is 50% of smaller volume
    max_overlap = min(torso_len, legs_len) // 2
    min_overlap = 10  # At least 10 slices overlap
    
    if verbose:
        print(f"      Torso: {torso_len} slices, Legs: {legs_len} slices")
        print(f"      Search range: {min_overlap} to {max_overlap} slices overlap")
    
    # ========== STAGE 1: COARSE SEARCH ==========
    coarse_step = max(5, max_overlap // 20)  # ~20 positions
    coarse_scores = []
    
    if verbose:
        print(f"\n      [Stage 1] Coarse search (step={coarse_step})...")
    
    for overlap in range(min_overlap, max_overlap + 1, coarse_step):
        # Get overlap slices
        torso_bottom = torso_arr[-overlap:]
        legs_top = legs_arr[:overlap]
        
        # Detect if this is a bone gap region
        overlap_volume = np.concatenate([torso_bottom, legs_top], axis=0)
        use_soft_tissue = detect_bone_gap(overlap_volume)
        
        # Calculate average similarity across overlap region
        sample_step = max(1, overlap // 10)
        scores = []
        for i in range(0, overlap, sample_step):
            score = calculate_slice_similarity(
                torso_bottom[i], 
                legs_top[i], 
                use_soft_tissue=use_soft_tissue
            )
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        coarse_scores.append((overlap, avg_score))
    
    # Sort by score and keep top 5 candidates
    coarse_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = coarse_scores[:5]
    
    if verbose:
        print(f"      Top 5 candidates: {[(o, f'{s:.3f}') for o, s in top_candidates]}")
    
    # ========== STAGE 2: FINE SEARCH AROUND CANDIDATES ==========
    fine_step = max(1, coarse_step // 5)
    best_overlap = top_candidates[0][0]
    best_score = top_candidates[0][1]
    
    if verbose:
        print(f"\n      [Stage 2] Fine search (step={fine_step})...")
    
    for candidate_overlap, candidate_score in top_candidates:
        # Search ±coarse_step around this candidate
        search_start = max(min_overlap, candidate_overlap - coarse_step)
        search_end = min(max_overlap, candidate_overlap + coarse_step)
        
        for overlap in range(search_start, search_end + 1, fine_step):
            torso_bottom = torso_arr[-overlap:]
            legs_top = legs_arr[:overlap]
            
            # Detect bone gap
            overlap_volume = np.concatenate([torso_bottom, legs_top], axis=0)
            use_soft_tissue = detect_bone_gap(overlap_volume)
            
            # More thorough scoring at fine level
            scores = []
            for i in range(0, overlap, max(1, overlap // 20)):
                score = calculate_slice_similarity(
                    torso_bottom[i], 
                    legs_top[i],
                    use_soft_tissue=use_soft_tissue
                )
                scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            
            if avg_score > best_score:
                best_score = avg_score
                best_overlap = overlap
    
    if verbose:
        print(f"      Fine search best: overlap={best_overlap}, score={best_score:.4f}")
    
    # ========== STAGE 3: ULTRA-FINE (1 slice precision) ==========
    if verbose:
        print(f"\n      [Stage 3] Ultra-fine search...")
    
    search_start = max(min_overlap, best_overlap - fine_step)
    search_end = min(max_overlap, best_overlap + fine_step)
    
    for overlap in range(search_start, search_end + 1):
        torso_bottom = torso_arr[-overlap:]
        legs_top = legs_arr[:overlap]
        
        # Detect bone gap
        overlap_volume = np.concatenate([torso_bottom, legs_top], axis=0)
        use_soft_tissue = detect_bone_gap(overlap_volume)
        
        # Full scoring
        scores = []
        for i in range(overlap):
            score = calculate_slice_similarity(
                torso_bottom[i], 
                legs_top[i],
                use_soft_tissue=use_soft_tissue
            )
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score > best_score:
            best_score = avg_score
            best_overlap = overlap
    
    # Determine confidence level
    if best_score >= 0.5:
        confidence = "HIGH"
    elif best_score >= 0.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    if verbose:
        print(f"      Final: overlap={best_overlap} slices, score={best_score:.4f}, confidence={confidence}")
    
    if confidence == "LOW":
        print(f"      ⚠️  WARNING: Low confidence alignment (score={best_score:.3f}). Results may be inaccurate.")
    elif confidence == "MEDIUM":
        print(f"      ⚠️  CAUTION: Medium confidence alignment (score={best_score:.3f}). Manual verification recommended.")
    
    return best_overlap, best_score, confidence


# ============================================================================
# STITCHING WITH BLENDING
# ============================================================================

def stitch_with_linear_blend(torso_img, legs_img, overlap_slices):
    """
    Stitch torso and legs with linear blending in overlap region.
    """
    print(f"    Stitching with {overlap_slices} slice overlap...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    
    # Calculate output size
    total_slices = len(torso_arr) + len(legs_arr) - overlap_slices
    
    # Create output array
    output = np.full((total_slices, torso_arr.shape[1], torso_arr.shape[2]), 
                     -1024, dtype=np.int16)
    
    # Copy non-overlapping torso (upper part)
    output[:len(torso_arr) - overlap_slices] = torso_arr[:-overlap_slices]
    
    # Blend overlap region
    overlap_start = len(torso_arr) - overlap_slices
    for i in range(overlap_slices):
        # Linear weight: 0 at start (all torso) to 1 at end (all legs)
        w_legs = i / overlap_slices
        w_torso = 1.0 - w_legs
        
        torso_slice = torso_arr[-(overlap_slices - i)].astype(np.float32)
        legs_slice = legs_arr[i].astype(np.float32)
        
        # Only blend where both have tissue
        blended = w_torso * torso_slice + w_legs * legs_slice
        output[overlap_start + i] = blended.astype(np.int16)
    
    # Copy non-overlapping legs (lower part)
    output[len(torso_arr):] = legs_arr[overlap_slices:]
    
    # Create output image
    output_img = sitk.GetImageFromArray(output)
    output_img.SetSpacing(torso_img.GetSpacing())
    output_img.SetOrigin(torso_img.GetOrigin())
    output_img.SetDirection(torso_img.GetDirection())
    
    print(f"      Output: {output.shape[0]} slices")
    
    return output_img


# ============================================================================
# ROTATION OPTIMIZATION
# ============================================================================

def rotate_180_z(img):
    """Rotate volume 180° around Z axis"""
    size = img.GetSize()
    center = img.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2.0 for sz in size]
    )
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0, 0, np.pi)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    
    return resampler.Execute(img)


def check_rotation_needed(torso_img, legs_img):
    """
    Check if legs need 180° rotation by comparing overlap quality.
    """
    print("    Checking rotation...")
    
    # Test original orientation
    _, score_original = multi_scale_z_search(torso_img, legs_img, verbose=False)
    
    # Test 180° rotated
    legs_rotated = rotate_180_z(legs_img)
    _, score_rotated = multi_scale_z_search(torso_img, legs_rotated, verbose=False)
    
    print(f"      Original: {score_original:.4f}, Rotated 180°: {score_rotated:.4f}")
    
    if score_rotated > score_original * 1.1:  # 10% better threshold
        print(f"      → Using ROTATED legs")
        return legs_rotated, True
    else:
        print(f"      → Using ORIGINAL legs")
        return legs_img, False


# ============================================================================
# MAIN STITCHING PIPELINE
# ============================================================================

# ============================================================================
# MAIN STITCHING PIPELINE
# ============================================================================

def stitch_body(name, volume_paths):
    """
    Complete pipeline to stitch a single body from ordered volumes.
    Includes bed artifact removal and confidence validation.
    
    Logic:
    - Lowest number = torso (reference, top part)
    - Middle number = pelvis (if 3 parts exist)
    - Highest number = legs (bottom part)
    
    If REVERSE_NUMBERING=True, this is flipped.
    """
    print("=" * 80)
    print(f"STITCHING: {name}")
    print("=" * 80)
    print()
    
    if not volume_paths or len(volume_paths) < 2:
        raise ValueError(f"Need at least 2 volumes to stitch, got {len(volume_paths)}")
    
    # Get sorted order numbers
    order_nums = sorted(volume_paths.keys())
    
    # Map numbers to body parts based on REVERSE_NUMBERING setting
    if REVERSE_NUMBERING:
        # 1=feet, 2=middle, 3=head → reverse it
        torso_num = order_nums[-1]  # Highest = torso (top)
        legs_num = order_nums[0]   # Lowest = legs (bottom)
        pelvis_num = order_nums[-2] if len(order_nums) == 3 else None
        print(f"  Using REVERSED numbering: {order_nums[-1]}(torso/top) → {order_nums[0]}(legs/bottom)")
    else:
        # 1=head, 2=middle, 3=feet → use as-is
        torso_num = order_nums[0]  # Lowest = torso (top)
        legs_num = order_nums[-1]  # Highest = legs (bottom)
        pelvis_num = order_nums[1] if len(order_nums) == 3 else None
        print(f"  Using STANDARD numbering: {order_nums[0]}(torso/top) → {order_nums[-1]}(legs/bottom)")
    
    # Load volumes
    print("\n  Loading volumes...")
    torso = sitk.ReadImage(volume_paths[torso_num])
    print(f"    Volume {torso_num} (torso/top): {torso.GetSize()}, {torso.GetSize()[2]} slices")
    
    legs = sitk.ReadImage(volume_paths[legs_num])
    print(f"    Volume {legs_num} (legs/bottom): {legs.GetSize()}, {legs.GetSize()[2]} slices")
    
    pelvis = None
    if pelvis_num is not None:
        pelvis = sitk.ReadImage(volume_paths[pelvis_num])
        print(f"    Volume {pelvis_num} (pelvis/middle): {pelvis.GetSize()}, {pelvis.GetSize()[2]} slices")
    
    # ========== BED ARTIFACT REMOVAL ==========
    print("\n  [Preprocessing] Removing bed artifacts...")
    torso = remove_bed_artifact(torso)
    legs = remove_bed_artifact(legs)
    if pelvis is not None:
        pelvis = remove_bed_artifact(pelvis)
    
    # If pelvis exists, first stitch torso + pelvis
    if pelvis is not None:
        print("\n  [Step 1] Stitching Torso + Pelvis...")
        
        # Check rotation
        pelvis, _ = check_rotation_needed(torso, pelvis)
        
        # XY alignment
        pelvis = align_xy_centers(torso, pelvis)
        pelvis = fine_tune_xy_alignment(torso, pelvis)
        
        # Z search (with confidence)
        overlap, score, confidence = multi_scale_z_search(torso, pelvis)
        
        # Stitch
        upper_body = stitch_with_linear_blend(torso, pelvis, overlap)
        print(f"    → Upper body: {upper_body.GetSize()[2]} slices\n")
        
        # Now stitch upper_body + legs
        print("  [Step 2] Stitching Upper Body + Legs...")
        torso = upper_body
    else:
        print("\n  Stitching Torso + Legs...")
    
    # Check rotation
    legs, was_rotated = check_rotation_needed(torso, legs)
    
    # XY alignment
    print("\n  XY Alignment:")
    legs = align_xy_centers(torso, legs)
    legs = fine_tune_xy_alignment(torso, legs)
    
    # Multi-scale Z search (with confidence)
    print("\n  Z Overlap Search:")
    overlap, score, confidence = multi_scale_z_search(torso, legs)
    
    # Stitch
    print("\n  Final Stitching:")
    result = stitch_with_linear_blend(torso, legs, overlap)
    
    print()
    print(f"  ✓ Final size: {result.GetSize()}")
    print(f"  ✓ Alignment confidence: {confidence} (score={score:.3f})")
    
    if confidence == "LOW":
        print(f"\n  ⚠️  WARNING: Low confidence stitching for {name}")
        print(f"      Manual inspection strongly recommended!")
    
    return result


def visualize_result(img, name):
    """Visualize stitched result with VTK"""
    print(f"\n  Visualizing {name}...")
    
    # Convert to VTK
    arr = sitk.GetArrayFromImage(img)
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(), deep=True, array_type=vtk.VTK_SHORT
    )
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(img.GetSize()[0], img.GetSize()[1], img.GetSize()[2])
    vtk_img.SetSpacing(img.GetSpacing())
    vtk_img.SetOrigin(img.GetOrigin())
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    # Volume rendering
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    
    # Transfer functions
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-300, 0.0)
    opacity.AddPoint(-100, 0.1)
    opacity.AddPoint(50, 0.2)
    opacity.AddPoint(300, 0.6)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-300, 0.4, 0.2, 0.2)
    color.AddRGBPoint(-100, 0.8, 0.5, 0.4)
    color.AddRGBPoint(50, 0.9, 0.6, 0.5)
    color.AddRGBPoint(300, 1.0, 0.9, 0.85)
    
    vol_prop = vtk.vtkVolumeProperty()
    vol_prop.SetColor(color)
    vol_prop.SetScalarOpacity(opacity)
    vol_prop.ShadeOn()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(vol_prop)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.15)
    renderer.ResetCamera()
    
    # Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1200, 900)
    window.SetWindowName(f"Improved Stitch: {name}")
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print("  Controls: Left-drag=Rotate, Right-drag=Zoom, 'q'=Next")
    
    window.Render()
    interactor.Start()


# ============================================================================
# MAIN
# ============================================================================

def main(visualize=True):
    """
    Main stitching pipeline.
    
    Args:
        visualize: If True, show VTK visualization windows. If False, skip visualization.
    """
    print("=" * 80)
    print("IMPROVED BODY STITCHER")
    print("With XY Alignment + Multi-Scale Z Search")
    print("Loading from nifti-ordered folder")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each patient
    results = {}
    for patient_name in PATIENT_NAMES:
        print(f"\n{'=' * 80}")
        print(f"PATIENT: {patient_name}")
        print('=' * 80)
        
        try:
            # Load ordered volumes
            print("\n  Loading ordered volumes from nifti-ordered folder...")
            volume_paths = load_ordered_volumes(patient_name)
            
            if volume_paths is None:
                print(f"  ⚠️  Skipping {patient_name} - no ordered data found")
                continue
            
            # Stitch the body
            result = stitch_body(patient_name, volume_paths)
            results[patient_name] = result
            
            # Save result
            output_path = os.path.join(OUTPUT_DIR, f'{patient_name}_stitched.nii')
            sitk.WriteImage(result, output_path)
            print(f"\n  💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"\n  ❌ Error processing {patient_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    
    if results:
        print("Successfully stitched:")
        for name in results:
            print(f"   ✓ {name}_stitched.nii")
    else:
        print("⚠️  No bodies were successfully stitched.")
        print("     Make sure patients have been processed with chooseLabelAxials.py first.")
        return
    
    # Visualize results (optional)
    if visualize and results:
        print("\nVisualizing results (press 'q' to move to next)...")
        for name, img in results.items():
            visualize_result(img, name)
    else:
        print("\n💾 All files saved. Skipping visualization.")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    # Check for --no-viz flag
    skip_viz = '--no-viz' in sys.argv or '--no-visualize' in sys.argv
    main(visualize=not skip_viz)