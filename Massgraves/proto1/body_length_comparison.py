#!/usr/bin/env python3
"""
BODY LENGTH COMPARISON
======================
Calculates and compares the lengths of stitched bodies vs their individual components.
Reads from:
- Stitched: ../finalMG/nifti-stitched/
- Components: ../finalMG/nifti-ordered/
"""

import SimpleITK as sitk
import os
import glob
import sys

def get_z_length(filepath):
    """Calculate physical length in Z direction (mm)"""
    try:
        img = sitk.ReadImage(filepath)
        size = img.GetSize()      # (X, Y, Z)
        spacing = img.GetSpacing() # (sx, sy, sz)
        length_mm = size[2] * spacing[2]
        return length_mm, size[2], spacing[2]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, 0

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    massgraves_root = os.path.dirname(script_dir)
    stitched_dir = os.path.join(massgraves_root, 'finalMG', 'nifti-stitched')
    ordered_dir = os.path.join(massgraves_root, 'finalMG', 'nifti-ordered')
    
    # Bodies to process
    bodies = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']
    
    output_lines = []
    header = f"{'BODY':<10} | {'TYPE':<20} | {'FILENAME':<30} | {'SLICES':<8} | {'SPACING':<8} | {'LENGTH (mm)':<12}"
    output_lines.append("=" * 100)
    output_lines.append(header)
    output_lines.append("=" * 100)
    
    print(header)
    print("-" * 100)

    for name in bodies:
        # 1. Get Stitched Body Info
        stitched_path = os.path.join(stitched_dir, f"{name}_stitched.nii")
        stitched_len = 0
        if os.path.exists(stitched_path):
            s_len, s_slices, s_spacing = get_z_length(stitched_path)
            stitched_len = s_len
            line = f"{name:<10} | {'STITCHED':<20} | {os.path.basename(stitched_path):<30} | {s_slices:<8} | {s_spacing:<8.2f} | {s_len:<12.2f}"
            output_lines.append(line)
            print(line)
        else:
            output_lines.append(f"{name:<10} | {'STITCHED':<20} | {'NOT FOUND':<30} | {'-':<8} | {'-':<8} | {'-':<12}")
            print(f"{name:<10} | {'STITCHED':<20} | {'NOT FOUND':<30} | {'-':<8} | {'-':<8} | {'-':<12}")

        # 2. Get Components Info
        body_comp_dir = os.path.join(ordered_dir, name)
        if os.path.exists(body_comp_dir):
            # Get all .nii files
            comp_files = sorted(glob.glob(os.path.join(body_comp_dir, "*.nii")))
            
            total_comp_len = 0
            for comp_path in comp_files:
                c_len, c_slices, c_spacing = get_z_length(comp_path)
                total_comp_len += c_len
                line = f"{'':<10} | {'COMPONENT':<20} | {os.path.basename(comp_path):<30} | {c_slices:<8} | {c_spacing:<8.2f} | {c_len:<12.2f}"
                output_lines.append(line)
                print(line)
            
            # Summary for this body
            diff = total_comp_len - stitched_len
            summary_line = f"{'':<10} | {'SUMMARY':<20} | {'Total Components':<30} | {'-':<8} | {'-':<8} | {total_comp_len:<12.2f}"
            diff_line = f"{'':<10} | {'DIFFERENCE':<20} | {'(Sum - Stitched)':<30} | {'-':<8} | {'-':<8} | {diff:<12.2f}"
            
            output_lines.append("-" * 100)
            output_lines.append(summary_line)
            output_lines.append(diff_line)
            output_lines.append("=" * 100)
            
            print("-" * 100)
            print(summary_line)
            print(diff_line)
            print("=" * 100)
        else:
            print(f"  No components found in {body_comp_dir}")
            output_lines.append(f"  No components found in {body_comp_dir}")
            output_lines.append("=" * 100)

    # Save to file
    output_file = os.path.join(script_dir, "body_length_comparison.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    main()
