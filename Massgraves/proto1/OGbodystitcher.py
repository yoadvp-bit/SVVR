#!/usr/bin/env python3
"""
IMPROVED BODY STITCHER
======================
Enhanced version of complete_body_stitcher.py with:
1. XY alignment optimization (center of mass + fine search)
2. Multi-scale Z search to avoid local minima
3. Better overlap scoring with surface matching

Based on: Massgraves/scripts5/complete_body_stitcher.py
"""

import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


# ============================================================================
# CONFIGURATION
# ============================================================================

# Bodies to process
BODIES = {
    'Maria': {
        'torso': '../exported_axial_volumes/Maria/series_005_CAP_w-o__5.0__B31f_214slices.nii',
        'legs': '../exported_axial_volumes/Maria/series_003_CAP_w-o__5.0__B31f_157slices.nii',
    },
    'Jan': {
        'torso': '../exported_axial_volumes/Jan/series_016_CAP_w-o__5.0__B31f_253slices.nii',
        'legs': '../exported_axial_volumes/Jan/series_008_CAP_w-o__5.0__B31f_196slices.nii',
    },
    'Jarek': {
        'torso': '../exported_axial_volumes/Jarek/series_013_Abdomen__1.0__B20f_968slices.nii',
        'pelvis': '../exported_axial_volumes/Jarek/series_009_Abdomen__5.0__B30f_129slices.nii',
        'legs': '../exported_axial_volumes/Jarek/series_005_Thorax__5.0__B31f_194slices.nii',
    },
    'Gerda': {
        'torso': '../exported_axial_volumes/Gerda/series_019_CAP_w-o__5.0__B31f_262slices.nii',
        'legs': '../exported_axial_volumes/Gerda/series_011_CAP_w-o__4.0__B31f_214slices.nii',
    },
    'Joost': {
        'torso': '../exported_axial_volumes/Joost/series_011_CAP_w-o__5.0__B31f_240slices.nii',
        'legs': '../exported_axial_volumes/Joost/series_003_CAP_w-o__5.0__B31f_184slices.nii',
    },
    'Loes': {
        'torso': '../exported_axial_volumes/Loes/series_008_CAP_w-o__5.0__B31f_237slices.nii',
        'legs': '../exported_axial_volumes/Loes/series_004_CAP_w-o__5.0__B31f_189slices.nii',
    },
}

OUTPUT_DIR = '../improved_stitched_bodies'


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


def multi_scale_z_search(torso_img, legs_img, verbose=True):
    """
    Multi-scale search for optimal Z overlap to avoid local minima.
    
    Strategy:
    1. Coarse search: Test many Z positions with large steps
    2. Keep top N candidates
    3. Fine search around each candidate
    4. Return global best
    """
    print("    Multi-scale Z overlap search...")
    
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
        
        # Calculate average similarity across overlap region
        # Sample every few slices for speed
        sample_step = max(1, overlap // 10)
        scores = []
        for i in range(0, overlap, sample_step):
            score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
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
            
            # More thorough scoring at fine level
            scores = []
            for i in range(0, overlap, max(1, overlap // 20)):
                score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
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
        
        # Full scoring
        scores = []
        for i in range(overlap):
            score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score > best_score:
            best_score = avg_score
            best_overlap = overlap
    
    if verbose:
        print(f"      Final: overlap={best_overlap} slices, score={best_score:.4f}")
    
    return best_overlap, best_score


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

def stitch_body(name, volume_paths):
    """
    Complete pipeline to stitch a single body.
    """
    print("=" * 80)
    print(f"STITCHING: {name}")
    print("=" * 80)
    print()
    
    # Load volumes
    print("  Loading volumes...")
    torso = sitk.ReadImage(volume_paths['torso'])
    print(f"    Torso: {torso.GetSize()}")
    
    legs = sitk.ReadImage(volume_paths['legs'])
    print(f"    Legs: {legs.GetSize()}")
    
    pelvis = None
    if 'pelvis' in volume_paths:
        pelvis = sitk.ReadImage(volume_paths['pelvis'])
        print(f"    Pelvis: {pelvis.GetSize()}")
    
    # If pelvis exists, first stitch torso + pelvis
    if pelvis is not None:
        print("\n  [Step 1] Stitching Torso + Pelvis...")
        
        # Check rotation
        pelvis, _ = check_rotation_needed(torso, pelvis)
        
        # XY alignment
        pelvis = align_xy_centers(torso, pelvis)
        pelvis = fine_tune_xy_alignment(torso, pelvis)
        
        # Z search
        overlap, score = multi_scale_z_search(torso, pelvis)
        
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
    
    # XY alignment (NEW!)
    print("\n  XY Alignment:")
    legs = align_xy_centers(torso, legs)
    legs = fine_tune_xy_alignment(torso, legs)
    
    # Multi-scale Z search (IMPROVED!)
    print("\n  Z Overlap Search:")
    overlap, score = multi_scale_z_search(torso, legs)
    
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

def main():
    print("=" * 80)
    print("IMPROVED BODY STITCHER")
    print("With XY Alignment + Multi-Scale Z Search")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each body
    results = {}
    for name, paths in BODIES.items():
        try:
            result = stitch_body(name, paths)
            results[name] = result
            
            # Save result
            output_path = os.path.join(OUTPUT_DIR, f'{name}_improved.nii')
            sitk.WriteImage(result, output_path)
            print(f"\n  💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"\n  ❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("Files:")
    for name in results:
        print(f"   - {name}_improved.nii")
    
    # Visualize results
    print("\nVisualizing results (press 'q' to move to next)...")
    for name, img in results.items():
        visualize_result(img, name)
    
    print("\nDone!")


if __name__ == '__main__':
    main()