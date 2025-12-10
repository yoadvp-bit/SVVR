"""
Visual Verification Tool for DICOM Body Assembly
Loads each series and creates a simple sagittal MIP (Maximum Intensity Projection)
to visually verify what body parts are in each series.
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from collections import defaultdict

def load_and_visualize_series(root_path, dataset_name):
    """Load all axial series and create visual summaries"""
    
    print(f"\n{'='*80}")
    print(f"VISUAL ANALYSIS: {dataset_name}")
    print(f"{'='*80}\n")
    
    # 1. Find all axial series
    series_map = defaultdict(list)
    series_info = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Check if axial
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if not np.allclose(vec_z, [0, 0, 1], atol=0.3): 
                        continue
                
                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                if uid not in series_info:
                    series_info[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'N/A'),
                        'num_files': 0
                    }
                series_info[uid]['num_files'] += 1
            except:
                continue
    
    # 2. Load and visualize each significant series
    valid_series = []
    for uid, data in series_map.items():
        if len(data) < 20: continue
        
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 100: continue  # Only significant scans
        
        valid_series.append({
            'uid': uid,
            'desc': series_info[uid]['desc'],
            'paths': [x[1] for x in data],
            'z_min': z_min,
            'z_max': z_max,
            'z_center': (z_min + z_max) / 2,
            'num_slices': len(data)
        })
    
    # Sort by Z center
    valid_series.sort(key=lambda x: x['z_center'], reverse=True)
    
    print(f"Found {len(valid_series)} significant axial series\n")
    
    # 3. Load each series and create MIP
    fig, axes = plt.subplots(len(valid_series), 3, figsize=(15, 5*len(valid_series)))
    if len(valid_series) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, series in enumerate(valid_series):
        print(f"Loading series {idx+1}/{len(valid_series)}: {series['desc']}")
        print(f"  Z-range: {series['z_min']:.1f} to {series['z_max']:.1f} mm")
        print(f"  Slices: {series['num_slices']}")
        
        try:
            # Load volume
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(series['paths'])
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)  # Shape: (Z, Y, X)
            
            print(f"  Volume shape: {arr.shape}")
            
            # Create projections
            # Axial (top-down view) - middle slice
            mid_z = arr.shape[0] // 2
            axial_slice = arr[mid_z, :, :]
            
            # Sagittal (side view) - MIP projection
            sagittal_mip = np.max(arr, axis=2)  # Max along X axis
            
            # Coronal (front view) - MIP projection  
            coronal_mip = np.max(arr, axis=1)  # Max along Y axis
            
            # Plot
            axes[idx, 0].imshow(axial_slice, cmap='gray', vmin=-200, vmax=200)
            axes[idx, 0].set_title(f"Axial (middle slice)\n{series['desc'][:30]}")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(sagittal_mip, cmap='gray', vmin=-200, vmax=200, aspect='auto')
            axes[idx, 1].set_title(f"Sagittal MIP (side view)\nZ: {series['z_min']:.0f} to {series['z_max']:.0f}")
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(coronal_mip, cmap='gray', vmin=-200, vmax=200, aspect='auto')
            axes[idx, 2].set_title(f"Coronal MIP (front view)")
            axes[idx, 2].axis('off')
            
            print(f"  ✓ Visualized\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            for ax in axes[idx]:
                ax.text(0.5, 0.5, f"Error loading\n{series['desc']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/scirpts/{dataset_name}_visualization.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {dataset_name}_visualization.png\n")
    plt.close()
    
    return valid_series

if __name__ == "__main__":
    base_path = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
    
    for dataset in ["DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"]:
        dataset_path = os.path.join(base_path, dataset)
        load_and_visualize_series(dataset_path, dataset)
