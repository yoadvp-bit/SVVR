"""Quick debug to see what series we're finding"""
import os
import pydicom
from collections import defaultdict

DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
DATASET = "DICOM-Jarek"

dataset_path = os.path.join(DATA_PATH, DATASET)

series_data = {}

for root, dirs, files in os.walk(dataset_path):
    for filename in files[:50]:  # Just check first 50 files
        if filename.startswith('.'):
            continue
        
        filepath = os.path.join(root, filename)
        
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            
            if 'ImagePositionPatient' not in dcm:
                continue
            
            series_uid = dcm.SeriesInstanceUID
            
            if series_uid not in series_data:
                series_data[series_uid] = {
                    'SeriesNumber': int(getattr(dcm, 'SeriesNumber', 0)),
                    'SeriesDescription': getattr(dcm, 'SeriesDescription', 'UNKNOWN'),
                    'ImageType': getattr(dcm, 'ImageType', []),
                    'ImageOrientationPatient': list(dcm.ImageOrientationPatient),
                }
        except Exception as e:
            continue

print(f"Found {len(series_data)} series")
for uid, data in list(series_data.items())[:3]:
    print(f"\nSeries {data['SeriesNumber']}: {data['SeriesDescription']}")
    print(f"  ImageType: {data['ImageType']}")
    print(f"  ImageOrientationPatient: {data['ImageOrientationPatient']}")
