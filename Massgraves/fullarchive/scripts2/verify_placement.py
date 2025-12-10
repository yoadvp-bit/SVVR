"""
Visual Verification of Body Part Placement
==========================================

Creates sagittal MIP (Maximum Intensity Projection) views to visually verify
that body parts are in the correct anatomical positions.

This will show us if heads are where heads should be, legs where legs should be, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from collections import defaultdict

DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Jan"

def scan_and_load_series(root_path):
    """Scan and load all primary axial series"""
    print(f"\nScanning {os.path.basename(root_path)}...\n")
    
    series_map = defaultdict(list)
    series_metadata = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Check if axial
                iop = dcm.ImageOrientationPatient
                row_vec = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                col_vec = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                slice_normal = np.cross(row_vec, col_vec)
                
                if not np.allclose(np.abs(slice_normal), [0, 0, 1], atol=0.3):
                    continue  # Skip non-axial
                
                # Check if primary (not derived)
                image_type = getattr(dcm, 'ImageType', [])
                if 'DERIVED' in str(image_type).upper():
                    continue
                
                uid = dcm.SeriesInstanceUID
                pos = [float(x) for x in dcm.ImagePositionPatient]
                series_map[uid].append((pos[2], filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'series_num': int(getattr(dcm, 'SeriesNumber', 0)),
                        'origin': pos,
                    }
            except:
                continue
    
    # Filter to significant series
    valid_series = []
    for uid, files in series_map.items():
        if len(files) < 10: continue
        
        files.sort(key=lambda x: x[0])
        z_positions = [x[0] for x in files]
        z_min, z_max = min(z_positions), max(z_positions)
        
        valid_series.append({
            'uid': uid,
            'desc': series_metadata[uid]['desc'],
            'series_num': series_metadata[uid]['series_num'],
            'files': [x[1] for x in files],
            'z_min': z_min,
            'z_max': z_max,
            'z_center': (z_min + z_max) / 2,
            'num_slices': len(files)
        })
    
    # Sort by Z position (head to feet)
    valid_series.sort(key=lambda x: x['z_max'], reverse=True)
    
    print(f"Found {len(valid_series)} primary axial series:\n")
    for i, s in enumerate(valid_series, 1):
        print(f"{i}. Series {s['series_num']:3d}: {s['desc'][:50]:<50}")
        print(f"   Z-range: [{s['z_min']:7.1f}, {s['z_max']:7.1f}] mm ({s['num_slices']} slices)")
        print(f"   Z-center: {s['z_center']:.1f} mm\n")
    
    return valid_series


def load_series_volume(file_list):
    """Load a series into a 3D numpy array"""
    print(f"  Loading {len(file_list)} slices...")
    
    # Read first slice to get dimensions
    dcm0 = pydicom.dcmread(file_list[0])
    rows, cols = dcm0.Rows, dcm0.Columns
    
    volume = np.zeros((len(file_list), rows, cols), dtype=np.int16)
    
    for i, filepath in enumerate(file_list):
        dcm = pydicom.dcmread(filepath)
        pixel_data = dcm.pixel_array.astype(np.float32)
        
        # Apply HU correction
        slope = float(getattr(dcm, 'RescaleSlope', 1.0))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        hu_data = (pixel_data * slope) + intercept
        
        volume[i, :, :] = hu_data.astype(np.int16)
    
    return volume


def visualize_series_positions(all_series):
    """Create visualization showing where each series is positioned"""
    
    num_series = len(all_series)
    fig, axes = plt.subplots(num_series, 2, figsize=(12, 4*num_series))
    if num_series == 1:
        axes = axes.reshape(1, -1)
    
    for idx, series in enumerate(all_series):
        print(f"\nSeries {idx+1}/{num_series}: {series['desc']}")
        print(f"  Z-range: [{series['z_min']:.1f}, {series['z_max']:.1f}] mm")
        
        # Load volume
        volume = load_series_volume(series['files'])
        
        print(f"  Volume shape: {volume.shape}")
        print(f"  HU range: [{np.min(volume)}, {np.max(volume)}]")
        
        # Create sagittal MIP (side view) - compress along X axis
        sagittal_mip = np.max(volume, axis=2)  # Max over columns (X)
        sagittal_mip = sagittal_mip.T  # Transpose to (Y, Z)
        
        # Create axial slice (top view) - middle slice
        mid_z = volume.shape[0] // 2
        axial_slice = volume[mid_z, :, :]
        
        # Plot sagittal MIP
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(sagittal_mip, cmap='gray', aspect='auto', 
                         vmin=-200, vmax=400)
        ax1.set_title(f'Series {series["series_num"]}: {series["desc"][:40]}\nSagittal MIP (Side View)\nZ=[{series["z_min"]:.0f}, {series["z_max"]:.0f}]mm')
        ax1.set_xlabel('Z (slice direction)')
        ax1.set_ylabel('Y (anterior-posterior)')
        
        # Plot axial slice
        ax2 = axes[idx, 1]
        im2 = ax2.imshow(axial_slice, cmap='gray', aspect='equal',
                         vmin=-200, vmax=400)
        ax2.set_title(f'Axial Slice (Top View)\nZ={series["z_min"] + mid_z * (series["z_max"]-series["z_min"])/series["num_slices"]:.0f}mm')
        ax2.set_xlabel('X (left-right)')
        ax2.set_ylabel('Y (anterior-posterior)')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/scripts2/{DATASET}_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}")
    
    plt.show()


def main():
    dataset_path = os.path.join(DATA_PATH, DATASET)
    
    # Scan and load all series
    all_series = scan_and_load_series(dataset_path)
    
    if len(all_series) == 0:
        print("✗ No primary axial series found")
        return
    
    # Visualize
    print(f"\n{'='*80}")
    print("LOADING VOLUMES FOR VISUALIZATION")
    print(f"{'='*80}")
    visualize_series_positions(all_series)
    
    print(f"\n{'='*80}")
    print("ANATOMICAL ASSESSMENT")
    print(f"{'='*80}\n")
    
    print("Expected Z-ranges for adult human body (approximate):")
    print("  Head:     +600 to +900 mm")
    print("  Thorax:   +200 to +600 mm") 
    print("  Abdomen:    -50 to +200 mm")
    print("  Pelvis:   -200 to  -50 mm")
    print("  Legs:     -900 to -200 mm")
    print()
    
    print("Found series:")
    for i, s in enumerate(all_series, 1):
        z_min, z_max = s['z_min'], s['z_max']
        
        # Classify by Z position
        if z_max > 500:
            region = "HEAD/UPPER THORAX"
        elif z_max > 100:
            region = "THORAX"
        elif z_max > -100:
            region = "ABDOMEN/LOWER THORAX"
        elif z_max > -300:
            region = "PELVIS"
        else:
            region = "LEGS"
        
        print(f"{i}. Z=[{z_min:7.1f}, {z_max:7.1f}] mm → Likely: {region}")
        print(f"   {s['desc']}")
        print()


if __name__ == "__main__":
    main()
