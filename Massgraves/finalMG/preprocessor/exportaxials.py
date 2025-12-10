#!/usr/bin/env python3
"""
EXPORT AXIAL DICOM TO NIFTI

Automatically finds and exports all axial CT series from patient DICOM folders to NIfTI format.

SETUP:
  - Expects a 'data' folder containing patient directories named 'DICOM-{PatientName}'
  - The 'data' folder should be in the parent, grandparent, or great-grandparent directory
  - No command-line arguments needed - just run with the play button in VS Code
  
  (Series are numbered sequentially; descriptions are unreliable and omitted)

FILTERING:
  - Only axial orientation series are exported
  - Series with fewer than 20 slices are skipped
  - Slices are automatically sorted by Z position

USAGE:
  python exportaxials.py
  
  Processes all DICOM-* folders found in the data directory automatically.
"""

import SimpleITK as sitk
import os
import pydicom
from collections import defaultdict


def find_data_folder():
    """Search for 'data' folder in parent directories (up to 3 levels)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for _ in range(4):  # Check current, parent, grandparent, great-grandparent
        data_path = os.path.join(current_dir, 'data')
        if os.path.isdir(data_path):
            return data_path
        current_dir = os.path.dirname(current_dir)
    
    return None


def scan_and_export_axial_series(dicom_root, output_dir):
    """Find all axial series and export to NIfTI"""
    
    print("=" * 70)
    print("AXIAL SERIES EXPORT")
    print("=" * 70)
    print(f"\nScanning: {dicom_root}")
    print(f"Output directory: {output_dir}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Scan all DICOM files and group by SeriesInstanceUID
    print("Step 1: Scanning DICOM files...")
    series_map = defaultdict(list)
    
    for root, dirs, files in os.walk(dicom_root):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                # Check if axial
                if hasattr(dcm, 'ImageOrientationPatient'):
                    orientation = dcm.ImageOrientationPatient
                    # Axial: [1,0,0,0,1,0] or close to it
                    if abs(orientation[0] - 1.0) < 0.1 and abs(orientation[4] - 1.0) < 0.1:
                        uid = dcm.SeriesInstanceUID
                        z_pos = float(dcm.ImagePositionPatient[2])
                        
                        series_map[uid].append({
                            'path': filepath,
                            'z': z_pos
                        })
            except:
                continue
    
    print(f"  → Found {len(series_map)} axial series")
    
    # Step 2: Process each series
    print(f"\nStep 2: Exporting series to NIfTI...\n")
    
    exported_count = 0
    axial_number = 1
    
    for uid, slices in series_map.items():
        # Sort by z position
        slices.sort(key=lambda x: x['z'])
        
        # Skip very small series (< 20 slices)
        if len(slices) < 20:
            print(f"  ⊘ Skipping series with only {len(slices)} slices (minimum: 20)")
            continue
        
        # Get Z range for info
        z_min = min(s['z'] for s in slices)
        z_max = max(s['z'] for s in slices)
        
        print(f"  → Processing axial_{axial_number:03d}: {len(slices):4d} slices | Z range: [{z_min:7.1f}, {z_max:7.1f}] mm")
        
        # Load volume
        file_paths = [s['path'] for s in slices]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        
        try:
            volume = reader.Execute()
            
            # Create filename
            output_filename = f"axial_{axial_number:03d}.nii"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as NIfTI
            sitk.WriteImage(volume, output_path)
            
            print(f"    ✓ Saved: {output_filename}")
            exported_count += 1
            axial_number += 1
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            axial_number += 1
    
    print(f"\n{'=' * 70}")
    print(f"✓ COMPLETE: Exported {exported_count} volumes to {output_dir}/")
    print(f"{'=' * 70}\n")


def process_all_patients(data_dir, output_base_dir):
    """Process all DICOM-* patient folders in the data directory"""
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING: ALL PATIENTS")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Output base directory: {output_base_dir}\n")
    
    # Find all DICOM-* folders
    patient_folders = []
    for item in os.listdir(data_dir):
        if item.startswith('DICOM-') and os.path.isdir(os.path.join(data_dir, item)):
            patient_folders.append(item)
    
    patient_folders.sort()
    
    if not patient_folders:
        print("⚠ No DICOM-* folders found in data directory")
        return
    
    print(f"Found {len(patient_folders)} patient(s): {', '.join([p.replace('DICOM-', '') for p in patient_folders])}\n")
    
    # Process each patient
    for i, folder in enumerate(patient_folders, 1):
        patient_name = folder.replace('DICOM-', '')
        
        print(f"\n[{i}/{len(patient_folders)}] Processing: {patient_name}")
        print("-" * 70)
        
        dicom_path = os.path.join(data_dir, folder)
        output_path = os.path.join(output_base_dir, f'NIfTI-{patient_name}')
        
        scan_and_export_axial_series(dicom_path, output_path)
    
    print("\n" + "=" * 70)
    print("✓ ALL PATIENTS PROCESSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # Find data folder automatically
    data_folder = find_data_folder()
    
    if data_folder is None:
        print("✗ ERROR: Could not find 'data' folder in parent directories")
        print("  Please ensure a 'data' folder exists containing DICOM-* patient folders")
        exit(1)
    
    # Set output directory to parent of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_base_dir = os.path.join(parent_dir, 'nifti-raw')
    
    # Process all patients
    process_all_patients(data_folder, output_base_dir)