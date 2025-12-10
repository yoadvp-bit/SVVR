#!/usr/bin/env python3
"""
BODY STITCHER WITH SMALL ROTATION + EXTENSIVE Z SEARCH
=======================================================
Copy of bodystitcher_smallrotation.py with:
- More extensive Z-axis search (more candidates, finer steps)
- Everything else IDENTICAL

Changes from bodystitcher_smallrotation.py:
- Increased coarse search granularity
- More candidates kept from coarse search (top 10 instead of 5)
- Finer fine-search steps
- Larger ultra-fine search range
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

OUTPUT_DIR = os.path.join(FINALMG_DIR, 'nifti-stitched-smallrot-extensivez')

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
# NEW: SMALL ROTATION FUNCTIONS
# ============================================================================

def rotate_z(img, angle_degrees):
    """
    Rotate volume around Z axis by given angle (in degrees).
    """
    size = img.GetSize()
    center = img.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2.0 for sz in size]
    )
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0, 0, np.deg2rad(angle_degrees))
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    
    return resampler.Execute(img)


# ============================================================================
# IMPROVED: EXTENSIVE Z SEARCH WITH SMALL ROTATION CHECK
# ============================================================================

def calculate_slice_similarity(slice1, slice2, threshold=-300):
    """
    Calculate similarity between two slices using multiple metrics.
    Returns combined score (higher is better).
    """
    # Create body masks
    mask1 = slice1 > threshold
    mask2 = slice2 > threshold
    
    if not mask1.any() or not mask2.any():
        return 0.0
    
    # 1. IoU of body regions
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0
    
    # 2. Correlation of intensities in overlap region
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
    
    # 3. Contour similarity (body boundary matching)
    # Simple: compare the area of body regions
    area1 = mask1.sum()
    area2 = mask2.sum()
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    # Combined score
    score = 0.4 * iou + 0.4 * correlation + 0.2 * area_ratio
    
    return score


def multi_scale_z_search_with_rotation(torso_img, legs_img, verbose=True):
    """
    EXTENSIVE multi-scale search for optimal Z overlap with small rotation optimization.
    
    MODIFIED FOR MORE EXTENSIVE SEARCH:
    - Finer coarse search step (more positions tested)
    - Keep top 10 candidates instead of 5
    - Finer fine-search steps
    - Larger ultra-fine search range
    
    Strategy:
    1. Coarse search: Test MANY Z positions with smaller steps
    2. Keep top 10 candidates
    3. Fine search around each candidate WITH rotation optimization
    4. Ultra-fine search with larger range
    5. Return global best overlap AND best rotation angle
    """
    print("    EXTENSIVE multi-scale Z overlap search WITH small rotation check...")
    
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
    
    # ========== STAGE 1: COARSE SEARCH (MORE EXTENSIVE) ==========
    # CHANGED: Smaller step = more positions tested
    coarse_step = max(3, max_overlap // 30)  # ~30 positions instead of ~20
    coarse_scores = []
    
    if verbose:
        print(f"\n      [Stage 1] EXTENSIVE coarse search (step={coarse_step})...")
    
    for overlap in range(min_overlap, max_overlap + 1, coarse_step):
        # Get overlap slices
        torso_bottom = torso_arr[-overlap:]
        legs_top = legs_arr[:overlap]
        
        # Calculate average similarity across overlap region
        # Sample every few slices for speed
        sample_step = max(1, overlap // 10)
        scores = []
        for i in range(0, overlap, sample_step):
            score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        coarse_scores.append((overlap, avg_score))
    
    # CHANGED: Keep top 10 candidates instead of 5
    coarse_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = coarse_scores[:10]
    
    if verbose:
        print(f"      Top 10 candidates: {[(o, f'{s:.3f}') for o, s in top_candidates[:5]]}...")
    
    # ========== STAGE 2: FINE SEARCH WITH ROTATION ==========
    # CHANGED: Finer step size
    fine_step = max(1, coarse_step // 3)  # Divide by 3 instead of 5
    best_overlap = top_candidates[0][0]
    best_score = top_candidates[0][1]
    best_rotation = 0.0
    
    if verbose:
        print(f"\n      [Stage 2] Fine search WITH rotation (step={fine_step})...")
    
    # Rotation angles to test (±20° in 5° steps)
    rotation_angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    
    for candidate_overlap, candidate_score in top_candidates:
        # Search ±coarse_step around this candidate
        search_start = max(min_overlap, candidate_overlap - coarse_step)
        search_end = min(max_overlap, candidate_overlap + coarse_step)
        
        for overlap in range(search_start, search_end + 1, fine_step):
            # Test different rotations for this overlap
            for angle in rotation_angles:
                # Rotate legs
                if angle == 0:
                    legs_rotated = legs_img
                else:
                    legs_rotated = rotate_z(legs_img, angle)
                
                legs_arr_rot = sitk.GetArrayFromImage(legs_rotated)
                torso_bottom = torso_arr[-overlap:]
                legs_top = legs_arr_rot[:overlap]
                
                # More thorough scoring at fine level
                scores = []
                for i in range(0, overlap, max(1, overlap // 20)):
                    score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
                    scores.append(score)
                
                avg_score = np.mean(scores) if scores else 0
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_overlap = overlap
                    best_rotation = angle
    
    if verbose:
        print(f"      Fine search best: overlap={best_overlap}, rotation={best_rotation}°, score={best_score:.4f}")
    
    # ========== STAGE 3: ULTRA-FINE (1 slice + 1° precision) ==========
    # CHANGED: Larger search range
    if verbose:
        print(f"\n      [Stage 3] Ultra-fine search (EXTENDED range)...")
    
    # CHANGED: Search ±(fine_step * 2) instead of ±fine_step
    search_start = max(min_overlap, best_overlap - fine_step * 2)
    search_end = min(max_overlap, best_overlap + fine_step * 2)
    
    # Fine rotation angles around best rotation (±2° in 1° steps)
    fine_rotation_angles = [best_rotation - 2, best_rotation - 1, best_rotation, 
                            best_rotation + 1, best_rotation + 2]
    # Clamp to ±20°
    fine_rotation_angles = [a for a in fine_rotation_angles if -20 <= a <= 20]
    
    for overlap in range(search_start, search_end + 1):
        for angle in fine_rotation_angles:
            # Rotate legs
            if angle == 0:
                legs_rotated = legs_img
            else:
                legs_rotated = rotate_z(legs_img, angle)
            
            legs_arr_rot = sitk.GetArrayFromImage(legs_rotated)
            torso_bottom = torso_arr[-overlap:]
            legs_top = legs_arr_rot[:overlap]
            
            # Full scoring
            scores = []
            for i in range(overlap):
                score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
                scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            
            if avg_score > best_score:
                best_score = avg_score
                best_overlap = overlap
                best_rotation = angle
    
    if verbose:
        print(f"      Final: overlap={best_overlap} slices, rotation={best_rotation}°, score={best_score:.4f}")
    
    return best_overlap, best_rotation, best_score


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
# MAIN STITCHING PIPELINE
# ============================================================================

def stitch_body(name, volume_paths):
    """
    Complete pipeline to stitch a single body from ordered volumes.
    
    MODIFIED: No 180° rotation check, but WITH small rotation optimization + EXTENSIVE Z search.
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
    
    # If pelvis exists, first stitch torso + pelvis
    if pelvis is not None:
        print("\n  [Step 1] Stitching Torso + Pelvis...")
        
        # NO 180° rotation check - REMOVED
        
        # XY alignment
        pelvis = align_xy_centers(torso, pelvis)
        pelvis = fine_tune_xy_alignment(torso, pelvis)
        
        # EXTENSIVE Z search WITH small rotation
        overlap, rotation, score = multi_scale_z_search_with_rotation(torso, pelvis)
        
        # Apply rotation if needed
        if abs(rotation) > 0.5:
            print(f"    Applying {rotation}° rotation to pelvis...")
            pelvis = rotate_z(pelvis, rotation)
        
        # Stitch
        upper_body = stitch_with_linear_blend(torso, pelvis, overlap)
        print(f"    → Upper body: {upper_body.GetSize()[2]} slices\n")
        
        # Now stitch upper_body + legs
        print("  [Step 2] Stitching Upper Body + Legs...")
        torso = upper_body
    else:
        print("\n  Stitching Torso + Legs...")
    
    # NO 180° rotation check - REMOVED
    
    # XY alignment
    print("\n  XY Alignment:")
    legs = align_xy_centers(torso, legs)
    legs = fine_tune_xy_alignment(torso, legs)
    
    # EXTENSIVE Multi-scale Z search WITH small rotation
    print("\n  Z Overlap Search WITH Small Rotation (EXTENSIVE):")
    overlap, rotation, score = multi_scale_z_search_with_rotation(torso, legs)
    
    # Apply rotation if needed
    if abs(rotation) > 0.5:
        print(f"\n  Applying {rotation}° rotation to legs...")
        legs = rotate_z(legs, rotation)
    
    # Stitch
    print("\n  Final Stitching:")
    result = stitch_with_linear_blend(torso, legs, overlap)
    
    print()
    print(f"  ✓ Final size: {result.GetSize()}")
    
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
    window.SetWindowName(f"Stitch with Small Rotation + Extensive Z: {name}")
    
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
    print("BODY STITCHER WITH SMALL ROTATION + EXTENSIVE Z SEARCH")
    print("NO 180° rotation + Small rotation (±20°) + EXTENSIVE Z search")
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
