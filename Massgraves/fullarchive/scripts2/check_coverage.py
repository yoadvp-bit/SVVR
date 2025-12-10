#!/usr/bin/env python3
"""
Check if gaps in visualization are due to missing data or rendering issues
"""
import numpy as np
from PIL import Image

# Based on the output, these are the Z-positions of the series:
# Series 10: Z=[-941.0, -297.7] mm - Abdomen  1.0  B20f  (LEGS)
# Series 9:  Z=[-941.0, -296.0] mm - Abdomen  5.0  B30f  (LEGS duplicate)
# Series 13: Z=[-412.4, 265.2]  mm - Abdomen  1.0  B20f  (PELVIS + ABDOMEN)
# Series 12: Z=[-410.5, 269.5]  mm - Abdomen  5.0  B30f  (PELVIS + ABDOMEN duplicate)
# Series 6:  Z=[-367.5, 602.5]  mm - Thorax  5.0  B80f  (THORAX + HEAD)
# Series 5:  Z=[-367.5, 602.5]  mm - Thorax  5.0  B31f  (THORAX + HEAD duplicate)

print("="*80)
print("COVERAGE ANALYSIS - checking for physical gaps in DICOM data")
print("="*80)

series_ranges = [
    ("Series 10 (LEGS)", -941.0, -297.7),
    ("Series 9 (LEGS dup)", -941.0, -296.0),
    ("Series 13 (PELVIS+ABD)", -412.4, 265.2),
    ("Series 12 (PELVIS+ABD dup)", -410.5, 269.5),
    ("Series 6 (THORAX+HEAD)", -367.5, 602.5),
    ("Series 5 (THORAX+HEAD dup)", -367.5, 602.5),
]

print("\n1. Series Z-ranges:")
for name, z_min, z_max in series_ranges:
    extent = z_max - z_min
    print(f"  {name:30} Z=[{z_min:7.1f}, {z_max:7.1f}] mm  |  extent: {extent:6.1f} mm")

print("\n2. Coverage analysis (sorted by Z):")
sorted_ranges = sorted(series_ranges, key=lambda x: x[1])

total_extent = 602.5 - (-941.0)
print(f"  Total Z-range: [{-941.0:.1f}, {602.5:.1f}] mm = {total_extent:.1f} mm")

# Check for gaps
print("\n3. Checking for physical gaps between series:")
gaps = []

for i in range(len(sorted_ranges) - 1):
    current_name, current_min, current_max = sorted_ranges[i]
    next_name, next_min, next_max = sorted_ranges[i+1]
    
    # Check if there's a gap
    if next_min > current_max:
        gap_size = next_min - current_max
        gaps.append((current_max, next_min, gap_size, current_name, next_name))
        print(f"  ⚠ GAP: Z=[{current_max:.1f}, {next_min:.1f}] mm ({gap_size:.1f} mm)")
        print(f"       Between: {current_name} → {next_name}")
    elif next_min < current_max:
        overlap = current_max - next_min
        print(f"  ✓ OVERLAP: {overlap:.1f} mm between {current_name} and {next_name}")
    else:
        print(f"  ✓ ADJACENT: {current_name} and {next_name}")

# Merge overlapping series to find true coverage
print("\n4. True anatomical coverage (merging overlaps):")

# Unique series (removing duplicates)
unique_ranges = [
    ("LEGS", -941.0, -296.0),  # Series 9/10 merged
    ("PELVIS+ABDOMEN", -412.4, 269.5),  # Series 12/13 merged  
    ("THORAX+HEAD", -367.5, 602.5),  # Series 5/6 merged
]

print("  Unique series after merging duplicates:")
for name, z_min, z_max in unique_ranges:
    extent = z_max - z_min
    print(f"    {name:20} Z=[{z_min:7.1f}, {z_max:7.1f}] mm  |  extent: {extent:6.1f} mm")

# Find actual gaps
print("\n5. ACTUAL GAPS in anatomical coverage:")
actual_gaps = []

# Gap 1: Between LEGS and PELVIS+ABDOMEN
if -296.0 < -412.4:  # legs end before pelvis starts? NO, pelvis overlaps
    # Pelvis starts at -412.4, legs end at -296.0
    # So there's NO gap, pelvis starts BEFORE legs end (overlap of 116.4mm)
    print(f"  ✓ LEGS → PELVIS: NO GAP (overlap {-296.0 - (-412.4):.1f} mm)")
else:
    print(f"  ✓ LEGS → PELVIS+ABDOMEN: OVERLAP of {-296.0 - (-412.4):.1f} mm")

# Gap 2: Between PELVIS+ABDOMEN and THORAX+HEAD  
if 269.5 < -367.5:  # abdomen ends before thorax starts? NO
    print(f"  Should not happen")
else:
    print(f"  ✓ PELVIS+ABDOMEN → THORAX+HEAD: OVERLAP of {269.5 - (-367.5):.1f} mm")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("✓ All series OVERLAP - there are NO physical gaps in the DICOM data!")
print("✓ The body should appear continuous from legs (-941mm) to head (+602mm)")
print("\n⚠ If visualization shows gaps, it's a rendering/placement BUG, not missing data")
print("="*80)
