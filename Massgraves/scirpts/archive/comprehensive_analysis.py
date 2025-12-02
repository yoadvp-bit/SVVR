import os
import pydicom
import numpy as np
from collections import defaultdict
import SimpleITK as sitk

def analyze_all_datasets():
    """Comprehensively analyze all three datasets to understand structure"""
    
    base_path = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
    datasets = ["DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"]
    
    all_results = {}
    
    for dataset_name in datasets:
        dataset_path = os.path.join(base_path, dataset_name)
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*80}\n")
        
        result = analyze_dataset_detailed(dataset_path)
        all_results[dataset_name] = result
        
    return all_results

def analyze_dataset_detailed(root_path):
    """Detailed analysis of a single dataset"""
    
    series_db = defaultdict(lambda: {'metadata': {}, 'slices': []})
    
    # 1. SCAN ALL DICOM FILES
    file_count = 0
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'SeriesInstanceUID' not in dcm: continue
                uid = dcm.SeriesInstanceUID
                
                # Collect metadata once per series
                if not series_db[uid]['slices']:
                    series_db[uid]['metadata'] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'N/A'),
                        'body_part': getattr(dcm, 'BodyPartExamined', 'N/A'),
                        'protocol': getattr(dcm, 'ProtocolName', 'N/A'),
                        'orientation': getattr(dcm, 'ImageOrientationPatient', None),
                        'rows': getattr(dcm, 'Rows', 0),
                        'cols': getattr(dcm, 'Columns', 0),
                        'spacing': getattr(dcm, 'PixelSpacing', [1.0, 1.0]),
                        'slice_thickness': getattr(dcm, 'SliceThickness', 1.0)
                    }
                
                # Collect spatial data
                if 'ImagePositionPatient' in dcm:
                    pos = dcm.ImagePositionPatient
                    series_db[uid]['slices'].append({
                        'z': float(pos[2]),
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'path': filepath
                    })
                    file_count += 1
            except:
                continue

    print(f"Total DICOM files scanned: {file_count}")
    print(f"Unique series found: {len(series_db)}\n")
    
    # 2. ANALYZE EACH SERIES
    axial_series = []
    
    for uid, data in series_db.items():
        meta = data['metadata']
        slices = data['slices']
        
        if len(slices) < 15: continue  # Skip small series
        
        slices.sort(key=lambda x: x['z'])
        
        # Calculate ranges
        z_vals = [s['z'] for s in slices]
        x_vals = [s['x'] for s in slices]
        y_vals = [s['y'] for s in slices]
        
        z_min, z_max = min(z_vals), max(z_vals)
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_span = z_max - z_min
        
        # Determine orientation
        iop = meta['orientation']
        orientation_type = "Unknown"
        is_axial = False
        
        if iop and len(iop) == 6:
            vec_x = np.array(iop[:3])
            vec_y = np.array(iop[3:])
            vec_z = np.cross(vec_x, vec_y)
            
            if np.abs(vec_z[2]) > 0.85:  # Z-axis dominates = Axial
                orientation_type = "AXIAL"
                is_axial = True
            elif np.abs(vec_z[0]) > 0.85:
                orientation_type = "SAGITTAL"
            elif np.abs(vec_z[1]) > 0.85:
                orientation_type = "CORONAL"
            else:
                orientation_type = "OBLIQUE"
        
        # Only work with AXIAL series for body assembly
        if is_axial and z_span > 100:  # Significant coverage
            series_info = {
                'uid': uid,
                'desc': meta['desc'],
                'body_part': meta['body_part'],
                'num_slices': len(slices),
                'z_min': z_min,
                'z_max': z_max,
                'z_center': (z_min + z_max) / 2,
                'z_span': z_span,
                'x_min': x_min,
                'x_max': x_max,
                'x_span': x_max - x_min,
                'y_min': y_min,
                'y_max': y_max,
                'y_span': y_max - y_min,
                'paths': [s['path'] for s in slices],
                'slice_thickness': meta['slice_thickness'],
                'pixel_spacing': meta['spacing']
            }
            axial_series.append(series_info)
    
    # 3. PRINT SUMMARY
    print(f"\n{'='*120}")
    print(f"AXIAL SERIES ONLY (for body reconstruction):")
    print(f"{'='*120}")
    print(f"{'#':<3} | {'DESCRIPTION':<25} | {'SLICES':<7} | {'Z-RANGE (mm)':<25} | {'X-SPAN':<10} | {'Y-SPAN':<10}")
    print(f"{'-'*120}")
    
    for i, series in enumerate(axial_series, 1):
        z_range_str = f"{series['z_min']:.1f} to {series['z_max']:.1f}"
        print(f"{i:<3} | {series['desc'][:25]:<25} | {series['num_slices']:<7} | {z_range_str:<25} | {series['x_span']:.1f}mm{'':<4} | {series['y_span']:.1f}mm")
    
    # 4. DETECT OVERLAPS AND GAPS
    print(f"\n{'='*120}")
    print(f"SPATIAL RELATIONSHIP ANALYSIS:")
    print(f"{'='*120}")
    
    # Sort by Z-center
    sorted_series = sorted(axial_series, key=lambda x: x['z_center'], reverse=True)
    
    for i in range(len(sorted_series) - 1):
        upper = sorted_series[i]
        lower = sorted_series[i+1]
        
        # Check Z relationship
        gap = upper['z_min'] - lower['z_max']
        
        # Check X overlap (do they cover the same body parts horizontally?)
        x_overlap = min(upper['x_max'], lower['x_max']) - max(upper['x_min'], lower['x_min'])
        x_overlap_pct = (x_overlap / min(upper['x_span'], lower['x_span'])) * 100 if min(upper['x_span'], lower['x_span']) > 0 else 0
        
        # Check Y overlap
        y_overlap = min(upper['y_max'], lower['y_max']) - max(upper['y_min'], lower['y_min'])
        y_overlap_pct = (y_overlap / min(upper['y_span'], lower['y_span'])) * 100 if min(upper['y_span'], lower['y_span']) > 0 else 0
        
        print(f"\n[{i+1}] {upper['desc'][:30]}")
        print(f"        Z: {upper['z_min']:.1f} to {upper['z_max']:.1f} (center: {upper['z_center']:.1f})")
        print(f"     vs")
        print(f"[{i+2}] {lower['desc'][:30]}")
        print(f"        Z: {lower['z_min']:.1f} to {lower['z_max']:.1f} (center: {lower['z_center']:.1f})")
        print(f"     -->")
        
        if gap > 0:
            print(f"        Z-GAP: {gap:.1f}mm (series are separated vertically)")
        else:
            print(f"        Z-OVERLAP: {abs(gap):.1f}mm (series overlap vertically)")
        
        print(f"        X-alignment: {x_overlap_pct:.1f}% overlap (horizontal body coverage)")
        print(f"        Y-alignment: {y_overlap_pct:.1f}% overlap (front-back alignment)")
        
        # Suggest if they should be stitched
        if -100 < gap < 100 and x_overlap_pct > 50 and y_overlap_pct > 50:
            print(f"        ✓ GOOD CANDIDATE FOR STITCHING")
        elif x_overlap_pct < 30 or y_overlap_pct < 30:
            print(f"        ✗ POOR X/Y ALIGNMENT - Different body parts?")
        elif abs(gap) > 200:
            print(f"        ✗ LARGE Z-GAP - Missing body section")
    
    return {
        'axial_series': axial_series,
        'sorted_series': sorted_series
    }

if __name__ == "__main__":
    analyze_all_datasets()
