#!/usr/bin/env python3
"""
Verify that Z-sorting is working correctly by examining the actual Z-ranges
"""
import os
import pydicom
from collections import defaultdict

# Path to DICOM data
dicom_base = "/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
dataset_name = "DICOM-Jarek"
dicom_path = os.path.join(dicom_base, dataset_name)

print(f"Analyzing Z-positions for {dataset_name}...\n")

# Group files by SeriesInstanceUID
series_dict = defaultdict(list)

for root, dirs, files in os.walk(dicom_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
            
            if not hasattr(dcm, 'SeriesInstanceUID'):
                continue
            
            series_uid = dcm.SeriesInstanceUID
            
            # Get position
            if hasattr(dcm, 'ImagePositionPatient'):
                pos = dcm.ImagePositionPatient
                series_dict[series_uid].append({
                    'z': float(pos[2]),
                    'series_num': dcm.SeriesNumber if hasattr(dcm, 'SeriesNumber') else 'Unknown',
                    'desc': dcm.SeriesDescription if hasattr(dcm, 'SeriesDescription') else 'Unknown'
                })
        except:
            pass

# Analyze each series
series_info = []
for series_uid, positions in series_dict.items():
    if len(positions) > 0:
        z_values = [p['z'] for p in positions]
        z_min = min(z_values)
        z_max = max(z_values)
        series_num = positions[0]['series_num']
        desc = positions[0]['desc']
        
        series_info.append({
            'series_uid': series_uid,
            'series_num': series_num,
            'desc': desc,
            'z_min': z_min,
            'z_max': z_max,
            'slices': len(positions)
        })

# Sort by Z_min (inferior to superior)
series_info.sort(key=lambda x: x['z_min'])

print("Series sorted by Z-position (inferior → superior / feet → head):\n")
for i, info in enumerate(series_info, 1):
    z_range = f"[{info['z_min']:.1f}, {info['z_max']:.1f}]"
    print(f"{i}. Series {info['series_num']:>3}: Z={z_range:>24} mm ({info['slices']:>4} slices) - {info['desc']}")

print("\n" + "="*80)
print("Expected anatomical order (inferior → superior):")
print("  1. Legs/Feet (most negative Z)")
print("  2. Pelvis")
print("  3. Abdomen")
print("  4. Thorax")
print("  5. Head (most positive Z)")
print("="*80)
