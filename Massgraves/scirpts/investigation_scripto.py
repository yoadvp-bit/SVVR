import os
import pydicom
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
# Change this to point to a specific patient folder
target_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/DICOM-Jarek" 

def analyze_dicom_structure(root_path):
    print(f"--- ANALYZING DATASET: {os.path.basename(root_path)} ---")
    
    # Store series data: { UID: { 'metadata': ..., 'slices': [] } }
    series_db = defaultdict(lambda: {'metadata': {}, 'slices': []})
    
    # 1. SCAN AND GROUP
    file_count = 0
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                # Read minimal header
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                # We need these basic tags to classify
                if 'SeriesInstanceUID' not in dcm: continue
                uid = dcm.SeriesInstanceUID
                
                # Collect Metadata (Only once per series)
                if not series_db[uid]['slices']:
                    series_db[uid]['metadata'] = {
                        'Desc': getattr(dcm, 'SeriesDescription', 'N/A'),
                        'BodyPart': getattr(dcm, 'BodyPartExamined', 'N/A'),
                        'Protocol': getattr(dcm, 'ProtocolName', 'N/A'),
                        'Modality': getattr(dcm, 'Modality', 'Unknown'),
                        # Orientation is Critical
                        'Orientation': getattr(dcm, 'ImageOrientationPatient', [0,0,0,0,0,0]),
                        'Rows': getattr(dcm, 'Rows', 0),
                        'Cols': getattr(dcm, 'Columns', 0)
                    }
                
                # Collect Spatial Data for every slice
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

    print(f"Scanned {file_count} valid DICOM files.")
    print(f"Found {len(series_db)} unique Series.\n")
    
    # 2. CLASSIFY AND REPORT
    print(f"{'TYPE':<10} | {'DESCRIPTION':<20} | {'ORIENT (Gravity)':<15} | {'Z-RANGE (Height)':<20} | {'X-RANGE (L/R)'}")
    print("-" * 100)
    
    classified_parts = []

    for uid, data in series_db.items():
        meta = data['metadata']
        slices = data['slices']
        
        if len(slices) < 10: continue # Skip noise/scouts
        
        # Sort by Z (Height)
        slices.sort(key=lambda x: x['z'])
        
        # --- CALCULATE METRICS ---
        z_min = slices[0]['z']
        z_max = slices[-1]['z']
        x_min = slices[0]['x']
        
        # 1. ORIENTATION CHECK
        # IOP is a vector of 6 numbers.
        # [1,0,0, 0,1,0] = Standard Axial (Head/Feet is Z)
        # Anything else means the patient is rotated or it's a sagittal scan.
        iop = meta['Orientation']
        orientation_type = "Unknown"
        if hasattr(iop, '__iter__') and len(iop) == 6:
            # Check standard axial: X axis is (1,0,0), Y axis is (0,1,0)
            vec_x = np.array(iop[:3])
            vec_y = np.array(iop[3:])
            vec_z = np.cross(vec_x, vec_y) # The slice normal
            
            if np.abs(vec_z[2]) > 0.9: 
                orientation_type = "AXIAL (Standard)"
            elif np.abs(vec_z[0]) > 0.9:
                orientation_type = "SAGITTAL (Side)"
            elif np.abs(vec_z[1]) > 0.9:
                orientation_type = "CORONAL (Front)"
            else:
                orientation_type = "OBLIQUE (Tilted)"

        # 2. BODY PART GUESS (The "Classifier")
        # We use Z-Height relative to the dataset to guess head vs legs
        # But for now, we just print the raw range.
        
        print(f"{orientation_type[:10]:<10} | {meta['Desc'][:20]:<20} | {orientation_type:<15} | {z_min:.1f} to {z_max:.1f} | {x_min:.1f}")
        
        classified_parts.append({
            'uid': uid,
            'type': orientation_type,
            'z_range': (z_min, z_max),
            'desc': meta['Desc']
        })

    return classified_parts

# Run
analyze_dicom_structure(target_folder)