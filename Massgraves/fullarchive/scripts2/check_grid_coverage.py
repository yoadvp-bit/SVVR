#!/usr/bin/env python3
"""
Load the unified volume and check if it has data in the "gap" regions
"""
import os
import pydicom
import numpy as np
from collections import defaultdict

# Recreate the unified volume to inspect it
dicom_base = "/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
dataset_name = "DICOM-Jarek"
dicom_path = os.path.join(dicom_base, dataset_name)

def load_series_volume_simple(series_files):
    """Quick load of a series"""
    series_files.sort(key=lambda f: pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2])
    
    first_dcm = pydicom.dcmread(series_files[0])
    slices = []
    for f in series_files:
        dcm = pydicom.dcmread(f)
        img = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept
        slices.append(img)
    
    volume = np.stack(slices, axis=0).astype(np.int16)
    origin = np.array(first_dcm.ImagePositionPatient, dtype=np.float64)
    spacing = np.array([first_dcm.PixelSpacing[0], first_dcm.PixelSpacing[1], 
                       float(first_dcm.SliceThickness if hasattr(first_dcm, 'SliceThickness') else 1.0)])
    
    return volume, origin, spacing

# Scan and group files
series_dict = defaultdict(list)
count = 0
for root, dirs, files in os.walk(dicom_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
            if hasattr(dcm, 'SeriesInstanceUID') and hasattr(dcm, 'ImagePositionPatient'):
                # Filter for primary axial series
                if (hasattr(dcm, 'ImageType') and 'PRIMARY' in dcm.ImageType and 
                    hasattr(dcm, 'ImageOrientationPatient')):
                    orient = dcm.ImageOrientationPatient
                    if abs(orient[4]) < 0.1 and abs(orient[2]) < 0.1:  # Axial
                        series_dict[dcm.SeriesInstanceUID].append(filepath)
                        count += 1
                        if count % 500 == 0:
                            print(f"Scanned {count} files...")
        except:
            pass

print(f"\n✓ Found {len(series_dict)} primary axial series")

# Load and analyze
series_info = []
for series_uid, files in series_dict.items():
    if len(files) > 50:  # Significant series only
        volume, origin, spacing = load_series_volume_simple(files)
        shape = volume.shape
        z_min = origin[2]
        z_max = origin[2] + shape[0] * spacing[2]
        
        series_info.append({
            'volume': volume,
            'origin': origin,
            'spacing': spacing,
            'z_min': z_min,
            'z_max': z_max,
            'num_slices': len(files)
        })
        print(f"  Loaded series: {len(files)} slices, Z=[{z_min:.1f}, {z_max:.1f}]")

# Sort by Z
series_info.sort(key=lambda s: s['z_min'])

print(f"\n" + "="*80)
print("CHECKING GRID PLACEMENT")
print("="*80)

# Create unified grid
global_z_min = min(s['z_min'] for s in series_info)
global_z_max = max(s['z_max'] for s in series_info)
spacing_z = series_info[0]['spacing'][2]

unified_z_size = int(np.ceil((global_z_max - global_z_min) / spacing_z))

print(f"\nUnified grid Z-dimension: {unified_z_size} slices")
print(f"Z-range: [{global_z_min:.1f}, {global_z_max:.1f}] mm")
print(f"Z-spacing: {spacing_z:.2f} mm")

# Track which Z-indices have data
z_coverage = np.zeros(unified_z_size, dtype=bool)

for i, s in enumerate(series_info):
    z_idx_start = int(np.round((s['z_min'] - global_z_min) / spacing_z))
    z_idx_end = z_idx_start + s['volume'].shape[0]
    
    print(f"\nSeries {i+1}: Z=[{s['z_min']:.1f}, {s['z_max']:.1f}]")
    print(f"  Grid indices: Z=[{z_idx_start}:{z_idx_end}] ({z_idx_end - z_idx_start} slices)")
    print(f"  Volume shape: {s['volume'].shape}")
    
    # Mark coverage
    z_coverage[z_idx_start:z_idx_end] = True

# Find gaps in coverage
print(f"\n" + "="*80)
print("Z-COVERAGE ANALYSIS")
print("="*80)

print(f"\nTotal Z-slices with data: {np.sum(z_coverage)} / {unified_z_size} ({np.sum(z_coverage)/unified_z_size*100:.1f}%)")

# Find gaps
gaps = []
in_gap = False
gap_start = 0

for i in range(unified_z_size):
    if not z_coverage[i]:
        if not in_gap:
            gap_start = i
            in_gap = True
    else:
        if in_gap:
            gap_size = i - gap_start
            if gap_size > 5:  # Significant gap
                gap_z_min = global_z_min + gap_start * spacing_z
                gap_z_max = global_z_min + i * spacing_z
                gaps.append((gap_start, i, gap_size, gap_z_min, gap_z_max))
            in_gap = False

if gaps:
    print(f"\n⚠ Found {len(gaps)} gap(s) in Z-coverage:")
    for gap_start, gap_end, gap_size, gap_z_min, gap_z_max in gaps:
        print(f"  Gap: Z-indices [{gap_start}:{gap_end}] ({gap_size} slices)")
        print(f"       Physical Z=[{gap_z_min:.1f}, {gap_z_max:.1f}] mm ({gap_z_max - gap_z_min:.1f} mm)")
else:
    print("\n✓ NO GAPS - Complete Z-coverage!")

print("\n" + "="*80)
