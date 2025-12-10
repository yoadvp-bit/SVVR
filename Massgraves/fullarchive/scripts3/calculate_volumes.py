"""
Calculate Volume of Each DICOM Series Group
============================================

Calculates the physical volume (in cm³ and liters) for each DICOM series
by counting voxels above tissue threshold and multiplying by voxel volume.
"""

import os
import numpy as np
import pydicom
from collections import defaultdict


# Configuration
DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Maria"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek", "DICOM-Gerda", "DICOM-Joop", "DICOM-Loes"

# Thresholds for volume calculation (Hounsfield Units)
AIR_THRESHOLD = -500  # Below this is air/background
BONE_THRESHOLD = 200  # Above this is bone


def scan_dicom_folder(folder_path):
    """Group files by SeriesInstanceUID"""
    print(f"\nScanning DICOM folder: {os.path.basename(folder_path)}")
    
    series_groups = defaultdict(list)
    series_metadata = {}
    files_scanned = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            files_scanned += 1
            
            if files_scanned % 500 == 0:
                print(f"  Scanned {files_scanned} files...")
            
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                series_uid = dcm.SeriesInstanceUID
                
                series_groups[series_uid].append(filepath)
                
                # Store metadata from first file
                if series_uid not in series_metadata:
                    series_metadata[series_uid] = {
                        'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'SeriesNumber': int(getattr(dcm, 'SeriesNumber', 0)),
                        'PixelSpacing': list(dcm.PixelSpacing),
                        'SliceThickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                        'Rows': int(dcm.Rows),
                        'Columns': int(dcm.Columns),
                        'RescaleSlope': float(getattr(dcm, 'RescaleSlope', 1.0)),
                        'RescaleIntercept': float(getattr(dcm, 'RescaleIntercept', 0.0)),
                    }
                
            except Exception:
                continue
    
    print(f"  ✓ Scanned {files_scanned} files")
    print(f"  ✓ Found {len(series_groups)} unique series\n")
    
    return series_groups, series_metadata


def calculate_series_volume(series_files, series_metadata):
    """
    Calculate volume of a series
    
    Returns:
    - total_voxels: number of voxels in volume
    - tissue_voxels: number of voxels containing tissue (HU > AIR_THRESHOLD)
    - bone_voxels: number of voxels containing bone (HU > BONE_THRESHOLD)
    - voxel_volume_mm3: volume of one voxel in mm³
    - total_volume_cm3: total physical volume in cm³
    - tissue_volume_cm3: tissue volume in cm³
    - bone_volume_cm3: bone volume in cm³
    """
    pixel_spacing = series_metadata['PixelSpacing']  # [row, col] spacing in mm
    
    # Calculate actual Z-spacing from slice positions if possible
    if len(series_files) > 1:
        try:
            dcm1 = pydicom.dcmread(series_files[0], stop_before_pixels=True)
            dcm2 = pydicom.dcmread(series_files[1], stop_before_pixels=True)
            pos1 = np.array(dcm1.ImagePositionPatient)
            pos2 = np.array(dcm2.ImagePositionPatient)
            z_spacing = np.linalg.norm(pos2 - pos1)
        except:
            z_spacing = series_metadata['SliceThickness']
    else:
        z_spacing = series_metadata['SliceThickness']
    
    # Voxel volume in mm³
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * z_spacing
    
    # Load all slices and count voxels
    rescale_slope = series_metadata['RescaleSlope']
    rescale_intercept = series_metadata['RescaleIntercept']
    
    total_voxels = 0
    tissue_voxels = 0
    bone_voxels = 0
    
    for i, filepath in enumerate(series_files):
        try:
            dcm = pydicom.dcmread(filepath)
            pixel_data = dcm.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units
            hu_data = (pixel_data * rescale_slope) + rescale_intercept
            
            # Count voxels
            total_voxels += hu_data.size
            tissue_voxels += np.sum(hu_data > AIR_THRESHOLD)
            bone_voxels += np.sum(hu_data > BONE_THRESHOLD)
            
        except Exception as e:
            print(f"    Warning: Could not read slice {i+1}: {e}")
            continue
    
    # Calculate volumes in cm³ (1 cm³ = 1000 mm³)
    total_volume_cm3 = (total_voxels * voxel_volume_mm3) / 1000.0
    tissue_volume_cm3 = (tissue_voxels * voxel_volume_mm3) / 1000.0
    bone_volume_cm3 = (bone_voxels * voxel_volume_mm3) / 1000.0
    
    return {
        'total_voxels': total_voxels,
        'tissue_voxels': tissue_voxels,
        'bone_voxels': bone_voxels,
        'voxel_volume_mm3': voxel_volume_mm3,
        'total_volume_cm3': total_volume_cm3,
        'tissue_volume_cm3': tissue_volume_cm3,
        'bone_volume_cm3': bone_volume_cm3,
        'z_spacing': z_spacing
    }


def main():
    """Calculate volumes for all series"""
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Scan folder
    series_groups, series_metadata = scan_dicom_folder(dataset_path)
    
    if len(series_groups) == 0:
        print("✗ No DICOM series found")
        return
    
    # Calculate volumes for each series
    print("="*120)
    print("VOLUME CALCULATION RESULTS")
    print("="*120)
    print(f"\nDataset: {DATASET}")
    print(f"Thresholds: Air < {AIR_THRESHOLD} HU, Bone > {BONE_THRESHOLD} HU\n")
    
    print(f"{'#':<4} | {'Description':<40} | {'Slices':<7} | {'Voxel mm³':<12} | {'Total cm³':<12} | {'Tissue cm³':<12} | {'Bone cm³':<12}")
    print("-"*120)
    
    results = []
    
    for i, (uid, files) in enumerate(sorted(series_groups.items(), 
                                            key=lambda x: series_metadata[x[0]]['SeriesNumber']), 1):
        metadata = series_metadata[uid]
        desc = metadata['SeriesDescription'][:40]
        num_slices = len(files)
        
        print(f"{i:<4} | {desc:<40} | {num_slices:<7} | ", end='', flush=True)
        
        # Calculate volume
        vol_info = calculate_series_volume(files, metadata)
        
        print(f"{vol_info['voxel_volume_mm3']:>11.3f} | "
              f"{vol_info['total_volume_cm3']:>11.1f} | "
              f"{vol_info['tissue_volume_cm3']:>11.1f} | "
              f"{vol_info['bone_volume_cm3']:>11.1f}")
        
        results.append({
            'number': i,
            'series_uid': uid,
            'description': metadata['SeriesDescription'],
            'num_slices': num_slices,
            **vol_info
        })
    
    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    total_tissue_volume = sum(r['tissue_volume_cm3'] for r in results)
    total_bone_volume = sum(r['bone_volume_cm3'] for r in results)
    
    print(f"\nTotal number of series: {len(results)}")
    print(f"Total tissue volume (all series): {total_tissue_volume:,.1f} cm³ = {total_tissue_volume/1000:.3f} liters")
    print(f"Total bone volume (all series): {total_bone_volume:,.1f} cm³ = {total_bone_volume/1000:.3f} liters")
    
    # Find largest series
    largest = max(results, key=lambda r: r['tissue_volume_cm3'])
    print(f"\nLargest series by tissue volume:")
    print(f"  #{largest['number']}: {largest['description']}")
    print(f"  {largest['tissue_volume_cm3']:,.1f} cm³ ({largest['num_slices']} slices)")
    
    # Detailed breakdown
    print(f"\n" + "="*120)
    print("DETAILED BREAKDOWN BY SERIES")
    print("="*120)
    
    for r in results:
        print(f"\nSeries #{r['number']}: {r['description']}")
        print(f"  Slices: {r['num_slices']}")
        print(f"  Voxel dimensions: {r['voxel_volume_mm3']:.3f} mm³")
        print(f"  Z-spacing: {r['z_spacing']:.3f} mm")
        print(f"  Total voxels: {r['total_voxels']:,}")
        print(f"  Tissue voxels (HU > {AIR_THRESHOLD}): {r['tissue_voxels']:,} ({r['tissue_voxels']/r['total_voxels']*100:.1f}%)")
        print(f"  Bone voxels (HU > {BONE_THRESHOLD}): {r['bone_voxels']:,} ({r['bone_voxels']/r['total_voxels']*100:.1f}%)")
        print(f"  Total volume: {r['total_volume_cm3']:,.1f} cm³")
        print(f"  Tissue volume: {r['tissue_volume_cm3']:,.1f} cm³ ({r['tissue_volume_cm3']/1000:.3f} L)")
        print(f"  Bone volume: {r['bone_volume_cm3']:,.1f} cm³ ({r['bone_volume_cm3']/1000:.3f} L)")
    
    print(f"\n" + "="*120)
    print("✓ VOLUME CALCULATION COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()
