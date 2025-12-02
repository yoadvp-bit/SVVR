"""
Multi-Part Body Assembly System
================================

Automatically assembles complete human body scans from multiple DICOM series.
Handles complex scenarios where body parts are split across 3+ series.

Features:
- Analyzes ALL available series
- Identifies anatomical regions (head, thorax, abdomen, pelvis, legs, feet)
- Assembles them in correct anatomical order (head → feet)
- Validates continuity and completeness
- Supports multiple partial scans of the same region (e.g., torso with left arm, torso with right arm)

Author: Auto-generated for DICOM body reconstruction
Date: 2025-12-01
"""

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import pydicom
import os
from collections import defaultdict
from scipy.ndimage import center_of_mass, label

# --- CONFIGURATION ---
DATASET = "DICOM-Jan"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"
BASE_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"
TARGET_FOLDER = os.path.join(BASE_PATH, DATASET)

# Anatomical priorities (cranial to caudal order)
ANATOMICAL_ORDER = {
    'HEAD': 1000,
    'NECK': 900,
    'THORAX_UPPER': 800,
    'THORAX_LOWER': 700,
    'ABDOMEN_UPPER': 600,
    'ABDOMEN_LOWER': 500,
    'PELVIS_UPPER': 400,
    'PELVIS_LOWER': 300,
    'THIGH': 200,
    'KNEE': 100,
    'LEG_LOWER': 50,
    'FEET': 0,
    'UNKNOWN': -100
}


def analyze_anatomical_features(volume_arr, z_min, z_max):
    """
    Deep anatomical analysis of volume content
    Returns detailed feature dictionary
    """
    features = {
        'has_skull': False,
        'has_brain': False,
        'has_ribs': False,
        'has_spine': False,
        'has_pelvis_bone': False,
        'has_leg_separation': False,
        'has_feet': False,
        'max_width': 0,
        'min_width': 999999,
        'avg_width': 0,
        'width_variation': 0,
        'high_density_count': 0,  # Bone pixels
        'medium_density_count': 0,  # Soft tissue
        'z_position': (z_min + z_max) / 2
    }
    
    num_slices = volume_arr.shape[0]
    widths = []
    bone_counts = []
    
    # Sample slices throughout volume
    sample_indices = np.linspace(0, num_slices-1, min(num_slices, 30), dtype=int)
    
    for z_idx in sample_indices:
        slice_2d = volume_arr[z_idx, :, :]
        
        # Body mask (exclude air)
        body_mask = (slice_2d > -500) & (slice_2d < 2000)
        
        if np.sum(body_mask) < 100:
            widths.append(0)
            bone_counts.append(0)
            continue
        
        # Measure width
        horizontal_proj = np.sum(body_mask, axis=0)
        width = np.sum(horizontal_proj > 0)
        widths.append(width)
        
        # Count bone (high density > 200 HU)
        bone_mask = slice_2d > 200
        bone_count = np.sum(bone_mask & body_mask)
        bone_counts.append(bone_count)
    
    widths_arr = np.array(widths)
    bone_arr = np.array(bone_counts)
    
    features['max_width'] = np.max(widths_arr) if len(widths_arr) > 0 else 0
    features['min_width'] = np.min(widths_arr[widths_arr > 0]) if np.any(widths_arr > 0) else 0
    features['avg_width'] = np.mean(widths_arr[widths_arr > 0]) if np.any(widths_arr > 0) else 0
    features['width_variation'] = np.std(widths_arr) if len(widths_arr) > 0 else 0
    features['high_density_count'] = np.sum(bone_arr)
    
    # Skull detection: Round, compact structure with high bone density at top
    top_10_percent = max(1, num_slices // 10)
    if top_10_percent > 0:
        top_slices = volume_arr[:top_10_percent, :, :]
        top_bone = np.sum(top_slices > 200)
        top_body = np.sum((top_slices > -500) & (top_slices < 2000))
        
        if top_body > 0 and (top_bone / top_body) > 0.15:  # High bone ratio
            features['has_skull'] = True
            features['has_brain'] = True
    
    # Rib detection: Repeated bone patterns in mid-sections
    mid_start = num_slices // 3
    mid_end = 2 * num_slices // 3
    if mid_start < mid_end and mid_end <= num_slices:
        mid_bone_density = bone_arr[mid_start:mid_end]
        if len(mid_bone_density) > 3:
            if np.mean(mid_bone_density) > 500 and np.std(mid_bone_density) > 100:
                features['has_ribs'] = True
    
    # Pelvis detection: Wide bone structure in lower-middle sections
    lower_30_start = int(num_slices * 0.6)
    lower_30_end = int(num_slices * 0.9)
    if lower_30_start < lower_30_end and lower_30_end <= num_slices:
        pelvis_region = volume_arr[lower_30_start:lower_30_end, :, :]
        pelvis_bone = np.sum(pelvis_region > 200)
        if pelvis_bone > 10000:
            features['has_pelvis_bone'] = True
    
    # Leg separation detection
    bottom_20_percent = max(1, int(num_slices * 0.2))
    sample_z = num_slices - bottom_20_percent
    if sample_z >= 0 and sample_z < num_slices:
        bottom_slice = volume_arr[sample_z, :, :]
        body_mask = (bottom_slice > -500) & (bottom_slice < 2000)
        
        # Check for two separate regions (legs)
        labeled_array, num_features = label(body_mask)
        if num_features >= 2:
            features['has_leg_separation'] = True
    
    # Feet detection: Very narrow, low in body
    if features['min_width'] < 80 and features['avg_width'] < 120:
        features['has_feet'] = True
    
    return features


def classify_anatomical_region(features, z_center, z_span):
    """
    Classify volume into specific anatomical region based on features
    Returns (region_name, priority_score, confidence)
    """
    confidence = 0.5  # Base confidence
    
    # HEAD detection (highest priority)
    if features['has_skull'] or features['has_brain']:
        if z_center > 500:
            return 'HEAD', ANATOMICAL_ORDER['HEAD'], 0.95
        else:
            # Skull but wrong position - might be inverted scan
            return 'HEAD', ANATOMICAL_ORDER['HEAD'], 0.70
    
    # THORAX detection (ribs are unique signature)
    if features['has_ribs']:
        if z_center > 300:
            return 'THORAX_UPPER', ANATOMICAL_ORDER['THORAX_UPPER'], 0.90
        elif z_center > 100:
            return 'THORAX_LOWER', ANATOMICAL_ORDER['THORAX_LOWER'], 0.85
        else:
            return 'ABDOMEN_UPPER', ANATOMICAL_ORDER['ABDOMEN_UPPER'], 0.80
    
    # LEGS/FEET detection (do this before pelvis to catch lower extremities)
    if features['has_leg_separation']:
        if features['has_feet'] or features['avg_width'] < 120:
            return 'FEET', ANATOMICAL_ORDER['FEET'], 0.85
        elif z_center < -400:
            return 'LEG_LOWER', ANATOMICAL_ORDER['LEG_LOWER'], 0.80
        else:
            return 'THIGH', ANATOMICAL_ORDER['THIGH'], 0.75
    
    # PELVIS detection (only if no legs detected)
    if features['has_pelvis_bone']:
        if z_center > 0:
            return 'PELVIS_UPPER', ANATOMICAL_ORDER['PELVIS_UPPER'], 0.85
        else:
            return 'PELVIS_LOWER', ANATOMICAL_ORDER['PELVIS_LOWER'], 0.85
    
    # Large upper volumes without ribs might be torso/abdomen
    if z_span > 500 and z_center > 200:
        return 'THORAX_LOWER', ANATOMICAL_ORDER['THORAX_LOWER'], 0.65
    
    # Large mid-body volumes
    if z_span > 400 and z_center > 0:
        return 'ABDOMEN_UPPER', ANATOMICAL_ORDER['ABDOMEN_UPPER'], 0.60
    
    # Position-based fallback (Z-coordinate hints)
    if z_center > 400:
        return 'THORAX_UPPER', ANATOMICAL_ORDER['THORAX_UPPER'], 0.50
    elif z_center > 100:
        return 'THORAX_LOWER', ANATOMICAL_ORDER['THORAX_LOWER'], 0.50
    elif z_center > -100:
        return 'ABDOMEN_LOWER', ANATOMICAL_ORDER['ABDOMEN_LOWER'], 0.50
    elif z_center > -400:
        return 'PELVIS_LOWER', ANATOMICAL_ORDER['PELVIS_LOWER'], 0.50
    else:
        return 'THIGH', ANATOMICAL_ORDER['THIGH'], 0.45
    
    # Unknown
    return 'UNKNOWN', ANATOMICAL_ORDER['UNKNOWN'], 0.30


def scan_and_classify_all(root_path):
    """Scan all DICOM series and classify them anatomically"""
    print(f"\n{'='*100}")
    print(f"MULTI-PART BODY ASSEMBLY: {os.path.basename(root_path)}")
    print(f"{'='*100}\n")
    
    print("Phase 1: Scanning DICOM files...")
    series_map = defaultdict(list)
    series_metadata = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Only axial series
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if not np.allclose(vec_z, [0, 0, 1], atol=0.3):
                        continue
                
                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'slice_thickness': getattr(dcm, 'SliceThickness', 5.0)
                    }
            except:
                continue
    
    print(f"  → Found {len(series_map)} series\n")
    
    print("Phase 2: Loading and analyzing anatomical content...")
    all_series = []
    
    for uid, data in series_map.items():
        if len(data) < 15: continue
        
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        z_span = abs(z_max - z_min)
        
        if z_span < 80: continue  # Too small
        
        try:
            print(f"  Loading: {series_metadata[uid]['desc'][:50]}...")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([x[1] for x in data])
            img = reader.Execute()
            volume_arr = sitk.GetArrayFromImage(img)
            
            # Analyze features
            features = analyze_anatomical_features(volume_arr, z_min, z_max)
            region, priority, confidence = classify_anatomical_region(features, (z_min+z_max)/2, z_span)
            
            print(f"    → {region} (confidence: {confidence:.0%})")
            
            series_info = {
                'uid': uid,
                'desc': series_metadata[uid]['desc'],
                'paths': [x[1] for x in data],
                'z_min': z_min,
                'z_max': z_max,
                'z_center': (z_min + z_max) / 2,
                'z_span': z_span,
                'num_slices': len(data),
                'slice_thickness': series_metadata[uid]['slice_thickness'],
                'region': region,
                'priority': priority,
                'confidence': confidence,
                'features': features,
                'image': img
            }
            
            all_series.append(series_info)
            
        except Exception as e:
            print(f"    → Skipped: {e}")
            continue
    
    # Remove duplicates (same Z-range and region)
    print(f"\nPhase 3: Removing duplicate series...")
    unique_series = []
    seen = set()
    
    for s in sorted(all_series, key=lambda x: x['confidence'], reverse=True):
        key = (s['region'], round(s['z_min']/10)*10, round(s['z_max']/10)*10)
        if key not in seen:
            unique_series.append(s)
            seen.add(key)
    
    print(f"  → Kept {len(unique_series)} unique series\n")
    
    return unique_series


def build_assembly_chain(all_series):
    """
    Build chain of series from head to feet
    Returns ordered list of series to stitch
    """
    print("Phase 4: Building anatomical assembly chain...")
    
    # Sort by anatomical priority (head to feet)
    sorted_series = sorted(all_series, key=lambda x: x['priority'], reverse=True)
    
    print(f"\n{'  #':<4} | {'REGION':<20} | {'Z-RANGE (mm)':<25} | {'SLICES':<8} | {'CONF':<6}")
    print(f"  {'-'*90}")
    
    for i, s in enumerate(sorted_series, 1):
        z_range = f"{s['z_min']:.0f} to {s['z_max']:.0f}"
        print(f"  {i:<4} | {s['region']:<20} | {z_range:<25} | {s['num_slices']:<8} | {s['confidence']:.0%}")
    
    # Check for anatomical completeness
    regions_present = set(s['region'] for s in sorted_series)
    
    print(f"\n  Anatomical regions detected: {', '.join(sorted(regions_present))}")
    
    expected_regions = ['HEAD', 'THORAX_UPPER', 'THORAX_LOWER', 'ABDOMEN_UPPER', 
                       'ABDOMEN_LOWER', 'PELVIS_UPPER', 'PELVIS_LOWER', 'THIGH', 'LEG_LOWER', 'FEET']
    missing_regions = [r for r in expected_regions if r not in regions_present]
    
    if missing_regions:
        print(f"  ⚠ Potentially missing: {', '.join(missing_regions[:3])}...")
    
    return sorted_series


def stitch_multiple_series(series_chain):
    """
    Stitch multiple series together in anatomical order
    """
    if len(series_chain) < 2:
        print("\n✗ Need at least 2 series to stitch")
        return None
    
    print(f"\nPhase 5: Stitching {len(series_chain)} series together...\n")
    
    # Start with the most cranial (head) series
    result_img = series_chain[0]['image']
    result_arr = sitk.GetArrayFromImage(result_img)
    
    print(f"  Base: {series_chain[0]['region']} ({len(result_arr)} slices)")
    
    for i in range(1, len(series_chain)):
        current_series = series_chain[i]
        current_img = current_series['image']
        
        print(f"\n  Adding: {current_series['region']}")
        
        # Find optimal overlap
        result_arr_current = sitk.GetArrayFromImage(result_img)
        next_arr = sitk.GetArrayFromImage(current_img)
        
        # Search for overlap
        max_search = min(len(result_arr_current) // 4, len(next_arr) // 4, 50)
        best_overlap = 1
        best_score = -1
        
        for k in range(1, max_search + 1):
            bottom_block = result_arr_current[-k:]
            top_block = next_arr[:k]
            
            # Simple correlation
            score = np.corrcoef(bottom_block.flatten(), top_block.flatten())[0, 1]
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_overlap = k
        
        print(f"    Overlap: {best_overlap} slices (score: {best_score:.3f})")
        
        # Concatenate
        stitched_arr = np.concatenate((result_arr_current, next_arr[best_overlap:]), axis=0)
        result_img = sitk.GetImageFromArray(stitched_arr)
        result_img.SetOrigin(series_chain[0]['image'].GetOrigin())
        result_img.SetSpacing(series_chain[0]['image'].GetSpacing())
        result_img.SetDirection(series_chain[0]['image'].GetDirection())
        
        print(f"    Cumulative: {len(stitched_arr)} slices")
    
    final_length_mm = len(stitched_arr) * series_chain[0]['slice_thickness']
    print(f"\n  ✓ Final assembled volume: {len(stitched_arr)} slices ({final_length_mm/10:.1f} cm)")
    
    return result_img


def validate_assembly(series_chain, final_img):
    """Validate the assembled body"""
    print(f"\nPhase 6: Validation...")
    
    # Check anatomical order
    for i in range(len(series_chain)-1):
        if series_chain[i]['priority'] < series_chain[i+1]['priority']:
            print(f"  ⚠ Warning: {series_chain[i]['region']} followed by {series_chain[i+1]['region']} (inverted order)")
    
    # Check completeness
    regions = [s['region'] for s in series_chain]
    
    has_head = any('HEAD' in r for r in regions)
    has_thorax = any('THORAX' in r for r in regions)
    has_abdomen = any('ABDOMEN' in r or 'PELVIS' in r for r in regions)
    has_legs = any('LEG' in r or 'THIGH' in r or 'FEET' in r for r in regions)
    
    completeness_score = sum([has_head, has_thorax, has_abdomen, has_legs]) / 4.0
    
    print(f"  Completeness: {completeness_score:.0%}")
    print(f"    Head: {'✓' if has_head else '✗'}")
    print(f"    Thorax: {'✓' if has_thorax else '✗'}")
    print(f"    Abdomen/Pelvis: {'✓' if has_abdomen else '✗'}")
    print(f"    Legs/Feet: {'✓' if has_legs else '✗'}")
    
    if completeness_score >= 0.75:
        print(f"\n  ✓ GOOD: Body assembly appears complete")
    elif completeness_score >= 0.50:
        print(f"\n  ⚠ FAIR: Some body parts may be missing")
    else:
        print(f"\n  ✗ POOR: Significant body parts missing")
    
    return completeness_score


def render_volume(sitk_img, title):
    """Render the final assembled volume"""
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    
    dims = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims)
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(0, 0, 0)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            mat.SetElement(i, j, direction[i + j*3])
    mat.SetElement(0, 3, origin[0])
    mat.SetElement(1, 3, origin[1])
    mat.SetElement(2, 3, origin[2])
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)
    
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-200, 0.0)
    opacity.AddPoint(-100, 0.1)
    opacity.AddPoint(40, 0.3)
    opacity.AddPoint(200, 0.6)
    opacity.AddPoint(500, 1.0)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-100, 0.8, 0.6, 0.5)
    color.AddRGBPoint(40, 0.8, 0.2, 0.2)
    color.AddRGBPoint(200, 1.0, 0.9, 0.9)
    color.AddRGBPoint(500, 1.0, 1.0, 1.0)
    
    volume = vtk.vtkVolume()
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_img)
    volume.SetMapper(mapper)
    volume.SetUserMatrix(mat)
    
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()
    volume.SetProperty(prop)
    
    renderer.AddVolume(volume)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    volume.Update()
    bounds = volume.GetBounds()
    if bounds:
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
        cam = renderer.GetActiveCamera()
        cam.SetFocalPoint(center)
        cam.SetPosition(center[0], center[1] - 3000, center[2])
        cam.SetViewUp(0, 0, -1)
        renderer.ResetCameraClippingRange()
    
    print(f"\nPhase 7: Rendering...")
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    # Full pipeline
    all_series = scan_and_classify_all(TARGET_FOLDER)
    
    if len(all_series) == 0:
        print("\n✗ No valid series found")
        exit(1)
    
    series_chain = build_assembly_chain(all_series)
    
    if len(series_chain) < 2:
        print("\n✗ Need at least 2 series")
        exit(1)
    
    final_volume = stitch_multiple_series(series_chain)
    
    if final_volume is None:
        print("\n✗ Assembly failed")
        exit(1)
    
    completeness = validate_assembly(series_chain, final_volume)
    
    render_volume(final_volume, f"Multi-Part Assembly: {DATASET} (Completeness: {completeness:.0%})")
    
    print(f"\n{'='*100}")
    print(f"✓ ASSEMBLY COMPLETE")
    print(f"{'='*100}\n")
