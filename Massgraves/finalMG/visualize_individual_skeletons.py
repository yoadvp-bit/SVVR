#!/usr/bin/env python3
"""
SKELETON BODY STITCHER
======================
EXACT COPY of bodystitcher.py but visualizes SKELETONS instead of soft tissue.
Does NOT save files - ONLY visualizes.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NIFTI_ORDERED_BASE = os.path.join(SCRIPT_DIR, 'nifti-ordered')

PATIENT_NAMES = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']

REVERSE_NUMBERING = True

# SKELETON SETTINGS
BONE_THRESHOLD_HU = 700  # Higher threshold for cleaner bone


# ============================================================================
# FILE LOADING (COPIED FROM BODYSTITCHER)
# ============================================================================

def load_ordered_volumes(patient_name):
    patient_dir = os.path.join(NIFTI_ORDERED_BASE, patient_name)
    
    if not os.path.exists(patient_dir):
        print(f"  ⚠️  Patient folder not found: {patient_dir}")
        return None
    
    nii_files = glob.glob(os.path.join(patient_dir, '*.nii'))
    
    if not nii_files:
        print(f"  ⚠️  No NIfTI files found in: {patient_dir}")
        return None
    
    volume_map = {}
    skipped_files = []
    for filepath in nii_files:
        filename = os.path.basename(filepath)
        
        if '_' not in filename:
            skipped_files.append(filename)
            continue
            
        try:
            prefix = filename.split('_')[0]
            order_num = int(prefix)
            
            if order_num in volume_map:
                print(f"  ⚠️  Warning: Duplicate order number {order_num} found!")
                print(f"      Using first file found.")
            else:
                volume_map[order_num] = filepath
                
        except (ValueError, IndexError):
            skipped_files.append(filename)
            continue
    
    if not volume_map:
        print(f"  ⚠️  No valid ordered files found in: {patient_dir}")
        return None
    
    sorted_volumes = dict(sorted(volume_map.items()))
    
    print(f"  ✓ Found {len(sorted_volumes)} ordered volumes:")
    for order_num, path in sorted_volumes.items():
        filename = os.path.basename(path)
        print(f"    {order_num}: {filename}")
    
    return sorted_volumes


# ============================================================================
# XY ALIGNMENT FUNCTIONS (COPIED FROM BODYSTITCHER)
# ============================================================================

def get_body_center_of_mass_xy(img, threshold=-300):
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    
    body_mask = arr > threshold
    
    if not body_mask.any():
        size = img.GetSize()
        return (origin[0] + size[0] * spacing[0] / 2,
                origin[1] + size[1] * spacing[1] / 2)
    
    coords = np.argwhere(body_mask)
    mean_y = coords[:, 1].mean()
    mean_x = coords[:, 2].mean()
    
    x_phys = origin[0] + mean_x * spacing[0]
    y_phys = origin[1] + mean_y * spacing[1]
    
    return (x_phys, y_phys)


def translate_volume_xy(img, dx_mm, dy_mm):
    if abs(dx_mm) < 0.1 and abs(dy_mm) < 0.1:
        return img
    
    transform = sitk.TranslationTransform(3)
    transform.SetOffset([-dx_mm, -dy_mm, 0])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    
    return resampler.Execute(img)


def align_xy_centers(torso_img, legs_img):
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
    print("    Fine-tuning XY alignment...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    spacing = torso_img.GetSpacing()
    
    search_range_vox = int(search_range_mm / spacing[0])
    step_vox = max(1, int(step_mm / spacing[0]))
    
    torso_slices = int(len(torso_arr) * 0.2)
    legs_slices = int(len(legs_arr) * 0.2)
    
    n_slices = min(torso_slices, legs_slices)
    
    torso_overlap = torso_arr[-n_slices:]
    legs_overlap = legs_arr[:n_slices]
    
    torso_mask = torso_overlap > -300
    legs_mask = legs_overlap > -300
    
    best_score = -1
    best_dx = 0
    best_dy = 0
    
    for dx in range(-search_range_vox, search_range_vox + 1, step_vox):
        for dy in range(-search_range_vox, search_range_vox + 1, step_vox):
            shifted_legs_mask = np.roll(np.roll(legs_mask, dx, axis=2), dy, axis=1)
            
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
    
    dx_mm = best_dx * spacing[0]
    dy_mm = best_dy * spacing[1]
    
    print(f"      Best XY shift: ({dx_mm:.1f}, {dy_mm:.1f}) mm, score: {best_score:.4f}")
    
    if abs(dx_mm) < 1.0 and abs(dy_mm) < 1.0:
        return legs_img
    
    return translate_volume_xy(legs_img, dx_mm, dy_mm)


# ============================================================================
# MULTI-SCALE Z SEARCH (COPIED FROM BODYSTITCHER)
# ============================================================================

def calculate_slice_similarity(slice1, slice2, threshold=-300):
    mask1 = slice1 > threshold
    mask2 = slice2 > threshold
    
    if not mask1.any() or not mask2.any():
        return 0.0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0
    
    common = np.logical_and(mask1, mask2)
    if common.sum() > 100:
        vals1 = slice1[common].astype(np.float32)
        vals2 = slice2[common].astype(np.float32)
        
        vals1_norm = (vals1 - vals1.mean()) / (vals1.std() + 1e-6)
        vals2_norm = (vals2 - vals2.mean()) / (vals2.std() + 1e-6)
        
        correlation = np.mean(vals1_norm * vals2_norm)
        correlation = max(0, correlation)
    else:
        correlation = 0
    
    area1 = mask1.sum()
    area2 = mask2.sum()
    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    score = 0.4 * iou + 0.4 * correlation + 0.2 * area_ratio
    
    return score


def multi_scale_z_search(torso_img, legs_img, verbose=True):
    print("    Multi-scale Z overlap search...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    
    torso_len = len(torso_arr)
    legs_len = len(legs_arr)
    
    max_overlap = min(torso_len, legs_len) // 2
    min_overlap = 10
    
    if verbose:
        print(f"      Torso: {torso_len} slices, Legs: {legs_len} slices")
        print(f"      Search range: {min_overlap} to {max_overlap} slices overlap")
    
    # STAGE 1: COARSE SEARCH
    coarse_step = max(5, max_overlap // 20)
    coarse_scores = []
    
    if verbose:
        print(f"\n      [Stage 1] Coarse search (step={coarse_step})...")
    
    for overlap in range(min_overlap, max_overlap + 1, coarse_step):
        torso_bottom = torso_arr[-overlap:]
        legs_top = legs_arr[:overlap]
        
        sample_step = max(1, overlap // 10)
        scores = []
        for i in range(0, overlap, sample_step):
            score = calculate_slice_similarity(torso_bottom[i], legs_top[i])
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        coarse_scores.append((overlap, avg_score))
    
    coarse_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = coarse_scores[:5]
    
    if verbose:
        print(f"      Top 5 candidates: {[(o, f'{s:.3f}') for o, s in top_candidates]}")
    
    # STAGE 2: FINE SEARCH
    fine_step = max(1, coarse_step // 5)
    best_overlap = top_candidates[0][0]
    best_score = top_candidates[0][1]
    
    if verbose:
        print(f"\n      [Stage 2] Fine search (step={fine_step})...")
    
    for candidate_overlap, candidate_score in top_candidates:
        search_start = max(min_overlap, candidate_overlap - coarse_step)
        search_end = min(max_overlap, candidate_overlap + coarse_step)
        
        for overlap in range(search_start, search_end + 1, fine_step):
            torso_bottom = torso_arr[-overlap:]
            legs_top = legs_arr[:overlap]
            
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
    
    # STAGE 3: ULTRA-FINE
    if verbose:
        print(f"\n      [Stage 3] Ultra-fine search...")
    
    search_start = max(min_overlap, best_overlap - fine_step)
    search_end = min(max_overlap, best_overlap + fine_step)
    
    for overlap in range(search_start, search_end + 1):
        torso_bottom = torso_arr[-overlap:]
        legs_top = legs_arr[:overlap]
        
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
# STITCHING (COPIED FROM BODYSTITCHER)
# ============================================================================

def stitch_with_linear_blend(torso_img, legs_img, overlap_slices):
    print(f"    Stitching with {overlap_slices} slice overlap...")
    
    torso_arr = sitk.GetArrayFromImage(torso_img)
    legs_arr = sitk.GetArrayFromImage(legs_img)
    
    total_slices = len(torso_arr) + len(legs_arr) - overlap_slices
    
    output = np.full((total_slices, torso_arr.shape[1], torso_arr.shape[2]), 
                     -1024, dtype=np.int16)
    
    output[:len(torso_arr) - overlap_slices] = torso_arr[:-overlap_slices]
    
    overlap_start = len(torso_arr) - overlap_slices
    for i in range(overlap_slices):
        w_legs = i / overlap_slices
        w_torso = 1.0 - w_legs
        
        torso_slice = torso_arr[-(overlap_slices - i)].astype(np.float32)
        legs_slice = legs_arr[i].astype(np.float32)
        
        blended = w_torso * torso_slice + w_legs * legs_slice
        output[overlap_start + i] = blended.astype(np.int16)
    
    output[len(torso_arr):] = legs_arr[overlap_slices:]
    
    output_img = sitk.GetImageFromArray(output)
    output_img.SetSpacing(torso_img.GetSpacing())
    output_img.SetOrigin(torso_img.GetOrigin())
    output_img.SetDirection(torso_img.GetDirection())
    
    print(f"      Output: {output.shape[0]} slices")
    
    return output_img


# ============================================================================
# ROTATION (COPIED FROM BODYSTITCHER)
# ============================================================================

def rotate_180_z(img):
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
    print("    Checking rotation...")
    
    _, score_original = multi_scale_z_search(torso_img, legs_img, verbose=False)
    
    legs_rotated = rotate_180_z(legs_img)
    _, score_rotated = multi_scale_z_search(torso_img, legs_rotated, verbose=False)
    
    print(f"      Original: {score_original:.4f}, Rotated 180°: {score_rotated:.4f}")
    
    if score_rotated > score_original * 1.1:
        print(f"      → Using ROTATED legs")
        return legs_rotated, True
    else:
        print(f"      → Using ORIGINAL legs")
        return legs_img, False


# ============================================================================
# MAIN STITCHING PIPELINE (COPIED FROM BODYSTITCHER)
# ============================================================================

def stitch_body(name, volume_paths):
    print("=" * 80)
    print(f"STITCHING: {name}")
    print("=" * 80)
    print()
    
    if not volume_paths or len(volume_paths) < 2:
        raise ValueError(f"Need at least 2 volumes to stitch, got {len(volume_paths)}")
    
    order_nums = sorted(volume_paths.keys())
    
    if REVERSE_NUMBERING:
        torso_num = order_nums[-1]
        legs_num = order_nums[0]
        pelvis_num = order_nums[-2] if len(order_nums) == 3 else None
        print(f"  Using REVERSED numbering: {order_nums[-1]}(torso/top) → {order_nums[0]}(legs/bottom)")
    else:
        torso_num = order_nums[0]
        legs_num = order_nums[-1]
        pelvis_num = order_nums[1] if len(order_nums) == 3 else None
        print(f"  Using STANDARD numbering: {order_nums[0]}(torso/top) → {order_nums[-1]}(legs/bottom)")
    
    print("\n  Loading volumes...")
    torso = sitk.ReadImage(volume_paths[torso_num])
    print(f"    Volume {torso_num} (torso/top): {torso.GetSize()}, {torso.GetSize()[2]} slices")
    
    legs = sitk.ReadImage(volume_paths[legs_num])
    print(f"    Volume {legs_num} (legs/bottom): {legs.GetSize()}, {legs.GetSize()[2]} slices")
    
    pelvis = None
    if pelvis_num is not None:
        pelvis = sitk.ReadImage(volume_paths[pelvis_num])
        print(f"    Volume {pelvis_num} (pelvis/middle): {pelvis.GetSize()}, {pelvis.GetSize()[2]} slices")
    
    if pelvis is not None:
        print("\n  [Step 1] Stitching Torso + Pelvis...")
        
        pelvis, _ = check_rotation_needed(torso, pelvis)
        pelvis = align_xy_centers(torso, pelvis)
        pelvis = fine_tune_xy_alignment(torso, pelvis)
        
        overlap, score = multi_scale_z_search(torso, pelvis)
        
        upper_body = stitch_with_linear_blend(torso, pelvis, overlap)
        print(f"    → Upper body: {upper_body.GetSize()[2]} slices\n")
        
        print("  [Step 2] Stitching Upper Body + Legs...")
        torso = upper_body
    else:
        print("\n  Stitching Torso + Legs...")
    
    legs, was_rotated = check_rotation_needed(torso, legs)
    
    print("\n  XY Alignment:")
    legs = align_xy_centers(torso, legs)
    legs = fine_tune_xy_alignment(torso, legs)
    
    print("\n  Z Overlap Search:")
    overlap, score = multi_scale_z_search(torso, legs)
    
    print("\n  Final Stitching:")
    result = stitch_with_linear_blend(torso, legs, overlap)
    
    print()
    print(f"  ✓ Final size: {result.GetSize()}")
    
    return result


# ============================================================================
# SKELETON VISUALIZATION (NEW - REPLACES SOFT TISSUE)
# ============================================================================

def visualize_skeleton(img, name):
    """Visualize SKELETON ONLY using bone threshold"""
    print(f"\n  Creating skeleton visualization for {name}...")
    
    # Extract bone data
    arr = sitk.GetArrayFromImage(img)
    bone_mask = (arr > BONE_THRESHOLD_HU).astype(np.uint8)
    
    bone_count = np.sum(bone_mask)
    print(f"  Initial bone voxels: {bone_count:,} ({100*bone_count/arr.size:.2f}%)")
    
    if bone_count == 0:
        print(f"  ⚠️  No bone found with threshold {BONE_THRESHOLD_HU} HU")
        return
    
    # CLEAN UP NOISE - morphological operations
    print(f"  Cleaning up noise...")
    
    # Convert to SimpleITK image for morphological ops
    bone_sitk = sitk.GetImageFromArray(bone_mask)
    bone_sitk.CopyInformation(img)
    
    # Binary closing to fill small gaps
    bone_sitk = sitk.BinaryMorphologicalClosing(bone_sitk, [3, 3, 3])
    
    # Remove small disconnected components
    connected = sitk.ConnectedComponent(bone_sitk)
    relabeled = sitk.RelabelComponent(connected, minimumObjectSize=1000)
    bone_sitk = relabeled > 0
    
    # Binary opening to smooth edges
    bone_sitk = sitk.BinaryMorphologicalOpening(bone_sitk, [2, 2, 2])
    
    # Convert back to numpy
    bone_mask = sitk.GetArrayViewFromImage(bone_sitk).astype(np.uint8)
    
    bone_count = np.sum(bone_mask)
    print(f"  After cleanup: {bone_count:,} voxels ({100*bone_count/arr.size:.2f}%)")
    
    if bone_count == 0:
        print(f"  ⚠️  No bone remaining after cleanup")
        return
    
    # Convert to VTK
    vtk_data = numpy_support.numpy_to_vtk(
        bone_mask.ravel(order='F'), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(img.GetSize()[0], img.GetSize()[1], img.GetSize()[2])
    vtk_img.SetSpacing(img.GetSpacing())
    vtk_img.SetOrigin(img.GetOrigin())
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    # Marching cubes
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_img)
    marching_cubes.SetValue(0, 0.5)
    marching_cubes.Update()
    
    # Smooth MORE aggressively
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(marching_cubes.GetOutputPort())
    smoother.SetNumberOfIterations(30)  # More smoothing
    smoother.SetRelaxationFactor(0.15)  # Stronger smoothing
    smoother.Update()
    
    # Decimate to reduce polygon count if needed
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(smoother.GetOutputPort())
    decimate.SetTargetReduction(0.3)  # Reduce by 30%
    decimate.PreserveTopologyOn()
    decimate.Update()
    
    # Mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(decimate.GetOutputPort())
    mapper.ScalarVisibilityOff()
    
    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.95, 0.95, 0.85)  # Bone color
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.15)
    
    # Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1200, 900)
    window.SetWindowName(f"SKELETON - {name} (HU > {BONE_THRESHOLD_HU})")
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    renderer.ResetCamera()
    
    print("  Controls: Left-drag=Rotate, Right-drag=Zoom, 'q'=Next")
    
    window.Render()
    interactor.Start()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("SKELETON VISUALIZATION (USING BODYSTITCHER ALGORITHM)")
    print("=" * 80)
    print(f"Bone threshold: {BONE_THRESHOLD_HU} HU")
    print("NO FILES WILL BE SAVED - VISUALIZATION ONLY")
    print()
    
    results = {}
    for patient_name in PATIENT_NAMES:
        print(f"\n{'=' * 80}")
        print(f"PATIENT: {patient_name}")
        print('=' * 80)
        
        try:
            print("\n  Loading ordered volumes from nifti-ordered folder...")
            volume_paths = load_ordered_volumes(patient_name)
            
            if volume_paths is None:
                print(f"  ⚠️  Skipping {patient_name} - no ordered data found")
                continue
            
            result = stitch_body(patient_name, volume_paths)
            results[patient_name] = result
            
        except Exception as e:
            print(f"\n  ❌ Error processing {patient_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("STITCHING COMPLETE!")
    print("=" * 80)
    
    if results:
        print("\nNow visualizing SKELETONS...")
        for name, img in results.items():
            visualize_skeleton(img, name)
    else:
        print("⚠️  No bodies were successfully stitched.")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
