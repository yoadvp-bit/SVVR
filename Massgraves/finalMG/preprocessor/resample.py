#!/usr/bin/env python3
"""
RESAMPLE AXIAL VOLUMES TO CONSISTENT SPACING

This script ensures all axial volumes for each patient have the same voxel spacing.
This is critical for proper stitching - without consistent spacing, body parts
will appear stretched or compressed when merged.

STRATEGY:
- For each patient, find the FINEST (smallest) XY spacing among all their volumes
- Resample all volumes to this common spacing
- This preserves detail (no information loss from upsampling fine data)
- Physical dimensions of the body remain unchanged

INPUT:  nifti-ordered/{PatientName}/*.nii
OUTPUT: nifti-resampled/{PatientName}/*.nii

The resampled files keep their original names.
"""

import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALMG_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(FINALMG_DIR, 'nifti-ordered')
OUTPUT_DIR = os.path.join(FINALMG_DIR, 'nifti-resampled')

# Patients to process
PATIENT_NAMES = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']


# ============================================================================
# RESAMPLING FUNCTIONS
# ============================================================================

def get_volume_info(filepath):
    """Get spacing and size info from a NIfTI file without loading full array"""
    img = sitk.ReadImage(str(filepath))
    return {
        'path': filepath,
        'filename': os.path.basename(filepath),
        'size': img.GetSize(),
        'spacing': img.GetSpacing(),
        'origin': img.GetOrigin(),
        'direction': img.GetDirection()
    }


def resample_volume(img, target_spacing, target_xy_size=None, interpolator=sitk.sitkLinear):
    """
    Resample a volume to a new spacing and optionally a target XY size.
    
    If target_xy_size is provided, the output will have exactly that XY dimension,
    with the image centered and padded/cropped as needed.
    
    Args:
        img: SimpleITK image
        target_spacing: tuple (x, y, z) of desired spacing in mm
        target_xy_size: tuple (x, y) of desired output size, or None to preserve physical extent
        interpolator: SimpleITK interpolation method
        
    Returns:
        Resampled SimpleITK image
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    original_origin = img.GetOrigin()
    
    if target_xy_size is not None:
        # Use specified XY size, calculate Z to preserve physical extent
        new_size = [
            target_xy_size[0],
            target_xy_size[1],
            int(round(original_size[2] * original_spacing[2] / target_spacing[2]))
        ]
        
        # Calculate new origin to center the data
        # Physical center of original image
        orig_center_x = original_origin[0] + (original_size[0] * original_spacing[0]) / 2
        orig_center_y = original_origin[1] + (original_size[1] * original_spacing[1]) / 2
        
        # New origin so that center stays the same
        new_origin = [
            orig_center_x - (new_size[0] * target_spacing[0]) / 2,
            orig_center_y - (new_size[1] * target_spacing[1]) / 2,
            original_origin[2]
        ]
    else:
        # Calculate new size to maintain physical extent
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(3)
        ]
        new_origin = original_origin
    
    # Ensure at least 1 voxel in each dimension
    new_size = [max(1, s) for s in new_size]
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(-1024)  # Air value for CT
    
    # Use identity transform (no rotation/translation, just resampling)
    resampler.SetTransform(sitk.Transform())
    
    return resampler.Execute(img)


def find_target_spacing(volume_infos):
    """
    Determine the target spacing for resampling.
    
    Strategy: Use the FINEST (smallest) spacing for X and Y.
    For Z, use the most common Z spacing (usually all the same).
    
    This ensures no loss of detail - we upsample coarser volumes rather than
    downsampling fine ones.
    """
    x_spacings = [info['spacing'][0] for info in volume_infos]
    y_spacings = [info['spacing'][1] for info in volume_infos]
    z_spacings = [info['spacing'][2] for info in volume_infos]
    
    # Use finest (smallest) XY spacing
    target_x = min(x_spacings)
    target_y = min(y_spacings)
    
    # For Z, use the most common value (mode)
    # If all different, use the smallest
    from collections import Counter
    z_counts = Counter([round(z, 2) for z in z_spacings])
    most_common_z = z_counts.most_common(1)[0][0]
    
    # Find actual z value closest to the rounded most common
    target_z = min(z_spacings, key=lambda z: abs(round(z, 2) - most_common_z))
    
    return (target_x, target_y, target_z)


def find_target_xy_size(volume_infos, target_spacing):
    """
    Determine the target XY grid size that can contain all volumes.
    
    Strategy: Calculate what size each volume would be at the target spacing,
    then use the maximum to ensure all data fits.
    """
    max_x = 0
    max_y = 0
    
    for info in volume_infos:
        # Physical extent of this volume
        phys_x = info['size'][0] * info['spacing'][0]
        phys_y = info['size'][1] * info['spacing'][1]
        
        # Size at target spacing
        size_x = int(np.ceil(phys_x / target_spacing[0]))
        size_y = int(np.ceil(phys_y / target_spacing[1]))
        
        max_x = max(max_x, size_x)
        max_y = max(max_y, size_y)
    
    # Round up to nearest multiple of 16 for efficiency (optional)
    # max_x = ((max_x + 15) // 16) * 16
    # max_y = ((max_y + 15) // 16) * 16
    
    return (max_x, max_y)


def process_patient(patient_name):
    """Process all axial volumes for a single patient"""
    print("=" * 70)
    print(f"PROCESSING: {patient_name}")
    print("=" * 70)
    
    patient_input_dir = os.path.join(INPUT_DIR, patient_name)
    patient_output_dir = os.path.join(OUTPUT_DIR, patient_name)
    
    if not os.path.exists(patient_input_dir):
        print(f"  ⚠️  Input folder not found: {patient_input_dir}")
        return False
    
    # Find all NIfTI files
    nii_files = sorted(glob.glob(os.path.join(patient_input_dir, '*.nii')))
    
    if not nii_files:
        print(f"  ⚠️  No .nii files found in {patient_input_dir}")
        return False
    
    print(f"\n  Found {len(nii_files)} volumes:")
    
    # Gather info from all volumes
    volume_infos = []
    for filepath in nii_files:
        info = get_volume_info(filepath)
        volume_infos.append(info)
        s = info['spacing']
        print(f"    {info['filename']}: size={info['size']}, spacing=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.2f})")
    
    # Check if resampling is needed
    xy_spacings = set((round(info['spacing'][0], 4), round(info['spacing'][1], 4)) 
                      for info in volume_infos)
    
    if len(xy_spacings) == 1:
        print(f"\n  ✓ All volumes already have consistent XY spacing!")
        # Still copy to output for consistency
    else:
        print(f"\n  ⚠️  Found {len(xy_spacings)} different XY spacings - resampling needed")
    
    # Determine target spacing
    target_spacing = find_target_spacing(volume_infos)
    print(f"\n  Target spacing: ({target_spacing[0]:.4f}, {target_spacing[1]:.4f}, {target_spacing[2]:.2f}) mm")
    
    # Determine target XY size (so all volumes have same grid dimensions)
    target_xy_size = find_target_xy_size(volume_infos, target_spacing)
    print(f"  Target XY size: {target_xy_size[0]} x {target_xy_size[1]}")
    
    # Create output directory
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Process each volume - ALL volumes need resampling to get consistent size
    print(f"\n  Resampling volumes to consistent grid...")
    for info in volume_infos:
        filepath = info['path']
        filename = info['filename']
        original_spacing = info['spacing']
        original_size = info['size']
        
        print(f"    {filename}: resampling...")
        print(f"        from: size={original_size}, spacing=({original_spacing[0]:.4f}, {original_spacing[1]:.4f}, {original_spacing[2]:.2f})")
        
        # Load full image
        img = sitk.ReadImage(str(filepath))
        
        # Resample to target spacing AND target XY size
        resampled = resample_volume(img, target_spacing, target_xy_size)
        
        # Report size change
        new_size = resampled.GetSize()
        print(f"        to:   size={new_size}, spacing=({target_spacing[0]:.4f}, {target_spacing[1]:.4f}, {target_spacing[2]:.2f})")
        
        # Save
        output_path = os.path.join(patient_output_dir, filename)
        sitk.WriteImage(resampled, output_path)
    
    print(f"\n  ✓ Saved {len(volume_infos)} volumes to {patient_output_dir}")
    return True


def verify_results():
    """Verify that all resampled volumes have consistent spacing AND size"""
    print("\n")
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    for patient_name in PATIENT_NAMES:
        patient_dir = os.path.join(OUTPUT_DIR, patient_name)
        
        if not os.path.exists(patient_dir):
            continue
        
        nii_files = sorted(glob.glob(os.path.join(patient_dir, '*.nii')))
        
        if not nii_files:
            continue
        
        print(f"\n  {patient_name}:")
        
        spacings = []
        sizes = []
        for filepath in nii_files:
            img = sitk.ReadImage(filepath)
            s = img.GetSpacing()
            sz = img.GetSize()
            spacings.append(s)
            sizes.append(sz)
            print(f"    {os.path.basename(filepath)}: size={sz}, spacing=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.2f})")
        
        # Check spacing consistency
        xy_spacing_set = set((round(s[0], 4), round(s[1], 4)) for s in spacings)
        if len(xy_spacing_set) > 1:
            print(f"    ❌ INCONSISTENT XY spacing!")
            all_good = False
        
        # Check XY size consistency
        xy_size_set = set((sz[0], sz[1]) for sz in sizes)
        if len(xy_size_set) > 1:
            print(f"    ❌ INCONSISTENT XY size!")
            all_good = False
        
        if len(xy_spacing_set) == 1 and len(xy_size_set) == 1:
            print(f"    ✓ Consistent spacing and XY size")
    
    return all_good


def main():
    print("=" * 70)
    print("RESAMPLE AXIAL VOLUMES TO CONSISTENT SPACING")
    print("=" * 70)
    print()
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each patient
    success_count = 0
    for patient_name in PATIENT_NAMES:
        print()
        if process_patient(patient_name):
            success_count += 1
    
    # Verify results
    all_consistent = verify_results()
    
    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Processed: {success_count}/{len(PATIENT_NAMES)} patients")
    print(f"  Output:    {OUTPUT_DIR}")
    if all_consistent:
        print(f"  Status:    ✓ All spacings consistent")
    else:
        print(f"  Status:    ❌ Some inconsistencies remain")
    print()


if __name__ == '__main__':
    main()
