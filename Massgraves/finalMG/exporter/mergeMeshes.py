#!/usr/bin/env python3
"""
EXPORT COMBINED SKELETONS
Combines merged-bodies and nifti-stitched volumes side-by-side for each patient.
Both volumes are already in skeleton form (bone only).
Outputs to nifti-final folder.
"""

import SimpleITK as sitk
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALMG_DIR = os.path.dirname(SCRIPT_DIR)

MERGED_DIR = os.path.join(FINALMG_DIR, "merged-bodies")
STITCHED_DIR = os.path.join(FINALMG_DIR, "nifti-stitched")
OUTPUT_DIR = os.path.join(FINALMG_DIR, "nifti-final")

# Padding between the two models (in mm)
PADDING_MM = 100.0

# Patient names
PATIENTS = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_volume(filepath):
    """Load a NIfTI volume and return image + metadata."""
    if not os.path.exists(filepath):
        return None
    
    img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(img)
    
    return {
        'image': img,
        'array': arr,
        'size': img.GetSize(),  # (X, Y, Z)
        'spacing': img.GetSpacing(),
        'origin': img.GetOrigin(),
        'shape': arr.shape  # (Z, Y, X)
    }


def flip_x_axis(arr):
    """Flip volume along X axis (left-right mirror)."""
    return np.flip(arr, axis=2)  # axis 2 is X in (Z, Y, X) array


def combine_side_by_side(vol1, vol2, padding_mm, flip_second=False):
    """
    Combine two volumes side-by-side with padding.
    
    Args:
        vol1: First volume dict (will be on the left)
        vol2: Second volume dict (will be on the right)
        padding_mm: Padding between volumes in mm
        flip_second: If True, flip the second volume for same orientation
    
    Returns:
        Combined SimpleITK Image
    """
    arr1 = vol1['array']
    arr2 = vol2['array']
    spacing1 = vol1['spacing']
    spacing2 = vol2['spacing']
    
    # Use spacing from the first volume as reference
    ref_spacing = spacing1
    
    # Flip second volume if needed (to face same direction)
    if flip_second:
        arr2 = flip_x_axis(arr2)
        print("    Flipped second volume for same orientation")
    
    # Get dimensions (Z, Y, X)
    z1, y1, x1 = arr1.shape
    z2, y2, x2 = arr2.shape
    
    # Calculate padding in pixels
    padding_x = int(np.ceil(padding_mm / ref_spacing[0]))
    
    # Calculate output dimensions
    out_z = max(z1, z2)
    out_y = max(y1, y2)
    out_x = x1 + padding_x + x2
    
    print(f"    Vol1: {x1}x{y1}x{z1}, Vol2: {x2}x{y2}x{z2}")
    print(f"    Padding: {padding_mm}mm = {padding_x} pixels")
    print(f"    Output: {out_x}x{out_y}x{out_z}")
    
    # Create output array filled with air
    output = np.full((out_z, out_y, out_x), -1024, dtype=np.int16)
    
    # Calculate vertical centering offsets
    z_offset1 = (out_z - z1) // 2
    y_offset1 = (out_y - y1) // 2
    
    z_offset2 = (out_z - z2) // 2
    y_offset2 = (out_y - y2) // 2
    
    # Place first volume on the left
    output[
        z_offset1:z_offset1 + z1,
        y_offset1:y_offset1 + y1,
        0:x1
    ] = arr1
    
    # Place second volume on the right (after padding)
    x_start2 = x1 + padding_x
    output[
        z_offset2:z_offset2 + z2,
        y_offset2:y_offset2 + y2,
        x_start2:x_start2 + x2
    ] = arr2
    
    # Create output image
    output_img = sitk.GetImageFromArray(output)
    output_img.SetSpacing(ref_spacing)
    output_img.SetOrigin((0, 0, 0))
    
    return output_img


def process_patient(patient_name):
    """Process a single patient - combine merged and stitched volumes."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {patient_name}")
    print(f"{'='*60}")
    
    # File paths
    merged_path = os.path.join(MERGED_DIR, f"{patient_name}_merged.nii")
    stitched_path = os.path.join(STITCHED_DIR, f"{patient_name}_stitched.nii")
    output_path = os.path.join(OUTPUT_DIR, f"{patient_name}_final.nii")
    
    # Load volumes
    print(f"\n  Loading merged body...")
    merged = load_volume(merged_path)
    if merged is None:
        print(f"    ⚠️  Not found: {merged_path}")
        return False
    print(f"    Size: {merged['size']}, Spacing: {merged['spacing']}")
    
    print(f"\n  Loading stitched body...")
    stitched = load_volume(stitched_path)
    if stitched is None:
        print(f"    ⚠️  Not found: {stitched_path}")
        return False
    print(f"    Size: {stitched['size']}, Spacing: {stitched['spacing']}")
    
    # Combine side-by-side
    # Merged on left, Stitched on right
    # Flip stitched if needed so both face same direction
    print(f"\n  Combining side-by-side (padding={PADDING_MM}mm)...")
    combined = combine_side_by_side(
        merged, 
        stitched, 
        padding_mm=PADDING_MM,
        flip_second=False  # Set to True if orientations differ
    )
    
    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sitk.WriteImage(combined, output_path)
    
    print(f"\n  ✓ Saved: {output_path}")
    print(f"  ✓ Final size: {combined.GetSize()}")
    
    return True


def main():
    """Process all patients."""
    print("=" * 60)
    print("EXPORT COMBINED SKELETONS")
    print("=" * 60)
    print(f"\nMerged bodies: {MERGED_DIR}")
    print(f"Stitched bodies: {STITCHED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Padding: {PADDING_MM}mm")
    
    # Process each patient
    success_count = 0
    for patient in PATIENTS:
        if process_patient(patient):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success_count}/{len(PATIENTS)} patients processed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()