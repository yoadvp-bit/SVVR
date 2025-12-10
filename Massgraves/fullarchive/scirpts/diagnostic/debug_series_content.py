"""Quick debug script to see ALL series in each dataset"""

import pydicom
import os
from collections import defaultdict
import numpy as np

DATASETS = ["DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"]
BASE_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"

for dataset in DATASETS:
    print(f"\n{'='*100}")
    print(f"DATASET: {dataset}")
    print(f"{'='*100}\n")
    
    root_path = os.path.join(BASE_PATH, dataset)
    series_map = defaultdict(list)
    series_metadata = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'body_part': getattr(dcm, 'BodyPartExamined', 'Unknown'),
                        'iop': list(dcm.ImageOrientationPatient) if 'ImageOrientationPatient' in dcm else None
                    }
            except:
                continue
    
    print(f"Total series found: {len(series_map)}\n")
    
    for i, (uid, data) in enumerate(sorted(series_map.items(), key=lambda x: len(x[1]), reverse=True), 1):
        meta = series_metadata[uid]
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        z_span = abs(z_max - z_min)
        
        # Check orientation
        orientation = "UNKNOWN"
        if meta['iop']:
            iop = meta['iop']
            vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
            if np.allclose(vec_z, [0, 0, 1], atol=0.3):
                orientation = "AXIAL"
            elif np.allclose(vec_z, [1, 0, 0], atol=0.3):
                orientation = "SAGITTAL"
            elif np.allclose(vec_z, [0, 1, 0], atol=0.3):
                orientation = "CORONAL"
        
        print(f"  {i:2}. {meta['desc'][:40]:<40} | {len(data):4} slices | Z: {z_min:7.1f} to {z_max:7.1f} ({z_span:5.0f}mm) | {orientation}")
