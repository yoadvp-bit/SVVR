"""
DICOM Hierarchy Inspector
=========================

Examines the complete DICOM hierarchy to understand how 3D Slicer groups data:
- Patient level: PatientID, PatientName
- Study level: StudyInstanceUID, StudyDate, StudyDescription
- Series level: SeriesInstanceUID, SeriesDescription

This reveals how to properly group and merge related scans.
"""

import os
import pydicom
from collections import defaultdict
from datetime import datetime

DATA_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"


def examine_dicom_hierarchy(dataset_name):
    """
    Scan all DICOM files and build complete Patient > Study > Series hierarchy
    """
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    
    print(f"\n{'='*120}")
    print(f"DICOM HIERARCHY ANALYSIS: {dataset_name}")
    print(f"{'='*120}\n")
    
    # Hierarchical structure: Patient > Study > Series
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    files_scanned = 0
    
    print("Scanning files...")
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.startswith('.'):
                continue
            
            filepath = os.path.join(root, filename)
            files_scanned += 1
            
            if files_scanned % 500 == 0:
                print(f"  {files_scanned} files scanned...")
            
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                
                if 'ImagePositionPatient' not in dcm:
                    continue
                
                # Extract hierarchy tags
                patient_id = getattr(dcm, 'PatientID', 'UNKNOWN')
                patient_name = str(getattr(dcm, 'PatientName', 'UNKNOWN'))
                
                study_uid = dcm.StudyInstanceUID
                study_date = getattr(dcm, 'StudyDate', 'UNKNOWN')
                study_description = getattr(dcm, 'StudyDescription', 'UNKNOWN')
                
                series_uid = dcm.SeriesInstanceUID
                series_number = int(getattr(dcm, 'SeriesNumber', 0))
                series_description = getattr(dcm, 'SeriesDescription', 'UNKNOWN')
                
                # Additional metadata
                modality = getattr(dcm, 'Modality', 'UNKNOWN')
                image_type = getattr(dcm, 'ImageType', [])
                
                # Store in hierarchy
                patient_key = f"{patient_name} ({patient_id})"
                
                study_info = hierarchy[patient_key][study_uid]
                if 'metadata' not in study_info:
                    study_info['metadata'] = {
                        'StudyDate': study_date,
                        'StudyDescription': study_description,
                        'StudyUID': study_uid
                    }
                    study_info['series'] = {}
                
                if series_uid not in study_info['series']:
                    study_info['series'][series_uid] = {
                        'metadata': {
                            'SeriesNumber': series_number,
                            'SeriesDescription': series_description,
                            'Modality': modality,
                            'ImageType': image_type,
                            'SeriesUID': series_uid
                        },
                        'files': []
                    }
                
                study_info['series'][series_uid]['files'].append(filepath)
                
            except Exception as e:
                continue
    
    print(f"✓ Scanned {files_scanned} files\n")
    
    return hierarchy


def format_date(date_str):
    """Format DICOM date (YYYYMMDD) to readable format"""
    if len(date_str) == 8:
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
            return dt.strftime('%Y-%m-%d')
        except:
            return date_str
    return date_str


def is_derived_series(image_type):
    """Check if series is derived (reformatted) or original acquisition"""
    if isinstance(image_type, list):
        image_type_str = '\\'.join(image_type)
    else:
        image_type_str = str(image_type)
    
    return 'DERIVED' in image_type_str.upper()


def classify_orientation(series_description):
    """Classify series by orientation from description"""
    desc_upper = series_description.upper()
    
    if 'COR' in desc_upper:
        return 'CORONAL'
    elif 'SAG' in desc_upper:
        return 'SAGITTAL'
    else:
        return 'AXIAL'


def display_hierarchy(hierarchy):
    """
    Display hierarchy in 3D Slicer style
    """
    print("="*120)
    print("PATIENT / STUDY / SERIES HIERARCHY (3D Slicer Format)")
    print("="*120)
    
    total_patients = len(hierarchy)
    total_studies = sum(len(studies) for studies in hierarchy.values())
    total_series = sum(
        len(study_info['series']) 
        for studies in hierarchy.values() 
        for study_info in studies.values()
    )
    
    print(f"\nSummary: {total_patients} Patient(s), {total_studies} Study(ies), {total_series} Series\n")
    
    for patient_name, studies in sorted(hierarchy.items()):
        print(f"\n📁 PATIENT: {patient_name}")
        print(f"   ({len(studies)} study/studies)")
        
        for study_uid, study_info in sorted(studies.items(), key=lambda x: x[1]['metadata']['StudyDate']):
            study_meta = study_info['metadata']
            study_date = format_date(study_meta['StudyDate'])
            study_desc = study_meta['StudyDescription']
            
            print(f"\n   📅 STUDY: {study_desc} ({study_date})")
            print(f"      Study UID: ...{study_uid[-20:]}")
            print(f"      ({len(study_info['series'])} series)")
            
            # Sort series by series number
            sorted_series = sorted(
                study_info['series'].items(),
                key=lambda x: x[1]['metadata']['SeriesNumber']
            )
            
            for series_uid, series_info in sorted_series:
                series_meta = series_info['metadata']
                series_num = series_meta['SeriesNumber']
                series_desc = series_meta['SeriesDescription']
                num_files = len(series_info['files'])
                
                # Classification
                is_derived = is_derived_series(series_meta['ImageType'])
                orientation = classify_orientation(series_desc)
                
                derived_marker = "📐 DERIVED" if is_derived else "🔬 PRIMARY"
                orientation_marker = f"[{orientation}]"
                
                print(f"      {series_num:3d}: {series_desc:<60} | {num_files:4d} slices | {derived_marker} {orientation_marker}")
    
    print(f"\n{'='*120}\n")


def analyze_study_grouping(hierarchy):
    """
    Analyze how studies should be merged/kept separate
    """
    print("="*120)
    print("STUDY GROUPING ANALYSIS")
    print("="*120)
    
    for patient_name, studies in sorted(hierarchy.items()):
        if len(studies) <= 1:
            print(f"\n✓ {patient_name}: Single study - no grouping needed")
            continue
        
        print(f"\n⚠️  {patient_name}: MULTIPLE STUDIES ({len(studies)} studies found)")
        print("     These are separate scan sessions and should NOT be merged!")
        
        for study_uid, study_info in sorted(studies.items(), key=lambda x: x[1]['metadata']['StudyDate']):
            study_meta = study_info['metadata']
            study_date = format_date(study_meta['StudyDate'])
            study_desc = study_meta['StudyDescription']
            
            # Count primary vs derived series
            primary_count = 0
            derived_count = 0
            
            for series_info in study_info['series'].values():
                if is_derived_series(series_info['metadata']['ImageType']):
                    derived_count += 1
                else:
                    primary_count += 1
            
            print(f"     • {study_desc} ({study_date})")
            print(f"       - {primary_count} primary series, {derived_count} derived series")
    
    print(f"\n{'='*120}\n")


def recommend_loading_strategy(hierarchy):
    """
    Recommend which series to load and how to combine them
    """
    print("="*120)
    print("LOADING STRATEGY RECOMMENDATIONS")
    print("="*120)
    
    for patient_name, studies in sorted(hierarchy.items()):
        print(f"\n{patient_name}:")
        
        for study_uid, study_info in sorted(studies.items(), key=lambda x: x[1]['metadata']['StudyDate']):
            study_meta = study_info['metadata']
            study_date = format_date(study_meta['StudyDate'])
            study_desc = study_meta['StudyDescription']
            
            print(f"\n  Study: {study_desc} ({study_date})")
            
            # Find primary axial series
            primary_series = []
            for series_uid, series_info in study_info['series'].items():
                series_meta = series_info['metadata']
                
                if not is_derived_series(series_meta['ImageType']):
                    orientation = classify_orientation(series_meta['SeriesDescription'])
                    if orientation == 'AXIAL' and len(series_info['files']) >= 10:
                        primary_series.append({
                            'number': series_meta['SeriesNumber'],
                            'description': series_meta['SeriesDescription'],
                            'slices': len(series_info['files'])
                        })
            
            if primary_series:
                print(f"  ✓ Load and MERGE these {len(primary_series)} primary axial series:")
                for s in sorted(primary_series, key=lambda x: x['number']):
                    print(f"     - Series {s['number']:3d}: {s['description']} ({s['slices']} slices)")
            else:
                print(f"  ✗ No suitable primary series found")
    
    print(f"\n{'='*120}\n")


def main():
    """
    Analyze all three datasets
    """
    datasets = ["DICOM-Jan", "DICOM-Maria", "DICOM-Jarek"]
    
    for dataset in datasets:
        hierarchy = examine_dicom_hierarchy(dataset)
        display_hierarchy(hierarchy)
        analyze_study_grouping(hierarchy)
        recommend_loading_strategy(hierarchy)
        print("\n" + "="*120)
        print(f"END OF {dataset} ANALYSIS")
        print("="*120 + "\n\n")


if __name__ == "__main__":
    main()
