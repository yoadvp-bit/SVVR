#!/usr/bin/env python3
"""
Merge ordered NIfTI files into complete body volumes.

Logic:
- Files are named: {order}_{original_name}.nii
- Order 1 = head/torso (top, highest Z)
- Order 2 = pelvis/middle
- Order 3 = legs (bottom, lowest Z)
- Same order numbers = side by side (same Z range)

NO overlap/blending - just place volumes at correct Z positions.
"""

import os
import re
import SimpleITK as sitk
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

NIFTI_ORDERED_DIR = "/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/finalMG/nifti-ordered"
OUTPUT_DIR = "/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/finalMG/merged-bodies"

# Padding between body parts (in mm)
PADDING_MM = 20.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_filename(filename):
    """
    Parse ordered filename to extract order number and original name.
    Format: {order}_{original_name}.nii
    
    Returns:
        (order_number, original_name) or (None, None) if parsing fails
    """
    match = re.match(r'^(\d+)_(.+)$', filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def get_volume_height_mm(img):
    """Get the physical height (Z extent) of a volume in mm."""
    spacing = img.GetSpacing()
    size = img.GetSize()
    return size[2] * spacing[2]


def load_patient_volumes(patient_dir):
    """
    Load all volumes for a patient and group by order number.
    
    Returns:
        dict: {order_number: [(filepath, image, height_mm), ...]}
    """
    volumes_by_order = defaultdict(list)
    
    for filename in os.listdir(patient_dir):
        if not filename.endswith('.nii'):
            continue
        
        order_num, original_name = parse_filename(filename)
        if order_num is None:
            print(f"    Warning: Could not parse filename: {filename}")
            continue
        
        filepath = os.path.join(patient_dir, filename)
        img = sitk.ReadImage(filepath)
        height_mm = get_volume_height_mm(img)
        
        volumes_by_order[order_num].append({
            'filepath': filepath,
            'filename': filename,
            'image': img,
            'height_mm': height_mm,
            'size': img.GetSize(),
            'spacing': img.GetSpacing(),
            'original_name': original_name
        })
        
        print(f"    Loaded: {filename} (order={order_num}, height={height_mm:.1f}mm, slices={img.GetSize()[2]})")
    
    return volumes_by_order


def merge_volumes_vertical(volumes_by_order):
    """
    Merge volumes vertically (head to toe) based on order numbers.
    
    Order 1 = top (highest Z position) - HEAD/TORSO
    Order 2 = below order 1 - PELVIS/MIDDLE  
    Order 3 = below order 2 - LEGS/FEET
    etc.
    
    In NIfTI/SimpleITK, Z=0 is typically at the BOTTOM, so:
    - Order 1 should be placed at HIGHEST Z indices
    - Order 2 should be placed BELOW (lower Z indices)
    
    Volumes with same order number are placed SIDE-BY-SIDE (horizontally).
    
    Returns:
        SimpleITK Image: merged volume
    """
    if not volumes_by_order:
        raise ValueError("No volumes to merge")
    
    # Get sorted order numbers (1 = top/head, 2 = below, etc.)
    order_nums = sorted(volumes_by_order.keys())
    print(f"\n  Found {len(order_nums)} vertical levels: {order_nums}")
    
    # Determine reference spacing (use first volume)
    ref_vol = volumes_by_order[order_nums[0]][0]
    ref_spacing = ref_vol['spacing']
    
    print(f"  Reference spacing: {ref_spacing}")
    
    # Calculate padding in slices
    padding_slices = int(np.ceil(PADDING_MM / ref_spacing[2]))
    padding_x_pixels = int(np.ceil(PADDING_MM / ref_spacing[0]))  # Horizontal padding
    print(f"  Padding between groups: {PADDING_MM}mm (Z: {padding_slices} slices, X: {padding_x_pixels} pixels)")
    
    # Calculate dimensions needed
    level_slices = {}
    level_heights = {}
    level_total_width = {}  # Total X width needed for side-by-side volumes
    max_y = 0
    
    for order_num in order_nums:
        vols = volumes_by_order[order_num]
        
        # Z: use maximum slices
        max_slices = max(v['size'][2] for v in vols)
        max_height = max(v['height_mm'] for v in vols)
        level_slices[order_num] = max_slices
        level_heights[order_num] = max_height
        
        # X: sum of all widths + padding between them
        total_width = sum(v['size'][0] for v in vols)
        if len(vols) > 1:
            total_width += (len(vols) - 1) * padding_x_pixels
        level_total_width[order_num] = total_width
        
        # Y: track maximum
        for v in vols:
            max_y = max(max_y, v['size'][1])
        
        print(f"  Level {order_num}: {len(vols)} volume(s), slices={max_slices}, total_width={total_width}px")
        for v in vols:
            print(f"    - {v['filename']}: {v['size'][0]}x{v['size'][1]}x{v['size'][2]}")
    
    # Calculate total dimensions
    total_slices = sum(level_slices.values()) + (len(order_nums) - 1) * padding_slices
    total_width = max(level_total_width.values())
    total_height = max_y
    
    print(f"\n  Output dimensions: {total_width} x {total_height} x {total_slices}")
    
    # Create output array filled with air
    output_arr = np.full(
        (total_slices, total_height, total_width),
        -1024,  # Air HU
        dtype=np.int16
    )
    
    # Place volumes from BOTTOM to TOP in the array
    current_z = 0  # Start from bottom of output array
    
    for order_num in reversed(order_nums):  # Process feet first, then pelvis, then head
        vols_at_level = volumes_by_order[order_num]
        level_slice_count = level_slices[order_num]
        level_width = level_total_width[order_num]
        
        print(f"\n  Placing level {order_num} at Z slices {current_z}-{current_z + level_slice_count}")
        
        # Calculate starting X position to center the group
        group_x_start = (total_width - level_width) // 2
        current_x = group_x_start
        
        for vol_info in vols_at_level:
            img = vol_info['image']
            arr = sitk.GetArrayFromImage(img)
            
            # Get volume dimensions
            vol_slices = arr.shape[0]
            vol_height_y = arr.shape[1]
            vol_width_x = arr.shape[2]
            
            # X position: current position in the side-by-side layout
            x_offset = current_x
            
            # Y position: center vertically
            y_offset = max(0, (total_height - vol_height_y) // 2)
            
            # Handle size differences
            copy_height = min(vol_height_y, total_height - y_offset)
            copy_width = min(vol_width_x, total_width - x_offset)
            actual_slices = min(vol_slices, total_slices - current_z)
            
            print(f"    {vol_info['filename']}: placing at X={x_offset}, Y={y_offset}, Z={current_z}")
            print(f"      Size: {copy_width}x{copy_height}x{actual_slices}")
            
            output_arr[
                current_z:current_z + actual_slices,
                y_offset:y_offset + copy_height,
                x_offset:x_offset + copy_width
            ] = arr[:actual_slices, :copy_height, :copy_width]
            
            # Move X position for next volume at same level
            current_x += vol_width_x + padding_x_pixels
        
        # Move up for next level
        current_z += level_slice_count
        
        # Add padding after this level (except after the last one placed, which is order 1)
        if order_num != order_nums[0]:
            print(f"    Adding {padding_slices} Z padding slices")
            current_z += padding_slices
    
    # Create output SimpleITK image
    output_img = sitk.GetImageFromArray(output_arr)
    output_img.SetSpacing(ref_spacing)
    output_img.SetOrigin((0, 0, 0))
    
    return output_img


def merge_patient(patient_name, patient_dir, output_dir):
    """Process a single patient."""
    print(f"\n{'='*60}")
    print(f"MERGING: {patient_name}")
    print(f"{'='*60}")
    
    # Load all volumes
    volumes_by_order = load_patient_volumes(patient_dir)
    
    if not volumes_by_order:
        print(f"  No valid volumes found for {patient_name}")
        return None
    
    # Merge volumes
    merged = merge_volumes_vertical(volumes_by_order)
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{patient_name}_merged.nii")
    
    sitk.WriteImage(merged, output_path)
    print(f"\n  ✓ Saved: {output_path}")
    print(f"  ✓ Final size: {merged.GetSize()} ({merged.GetSize()[2]} slices)")
    
    return merged


def main():
    """Process all patients in nifti-ordered directory."""
    print("=" * 60)
    print("MERGE ORDERED NIFTI FILES")
    print("=" * 60)
    print(f"\nInput: {NIFTI_ORDERED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Find all patient directories
    patients = []
    for name in os.listdir(NIFTI_ORDERED_DIR):
        patient_dir = os.path.join(NIFTI_ORDERED_DIR, name)
        if os.path.isdir(patient_dir):
            patients.append((name, patient_dir))
    
    if not patients:
        print("\nNo patient directories found!")
        return
    
    print(f"\nFound {len(patients)} patients: {[p[0] for p in patients]}")
    
    # Process each patient
    for patient_name, patient_dir in sorted(patients):
        merge_patient(patient_name, patient_dir, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
