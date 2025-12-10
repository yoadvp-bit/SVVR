"""
Complete Body Reconstruction System
====================================

Assembles full human body from multiple DICOM series with proper anatomical ordering.

Key Features:
- Loads all axial series regardless of metadata
- Deep content analysis for anatomical classification
- Handles overlapping/duplicate series intelligently
- Ensures cranial→caudal ordering (head → feet)
- Validates completeness and continuity
- Organized output with proper documentation

Version: 2.0
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
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

DATASET = "DICOM-Maria"  # Options: "DICOM-Maria", "DICOM-Jan", "DICOM-Jarek"
BASE_PATH = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data"


# ==================== DATA LOADING ====================

def load_all_axial_series(root_path):
    """
    Load ALL axial DICOM series from directory
    Returns list of series with metadata and image data
    """
    print(f"\n{'='*100}")
    print(f"COMPLETE BODY RECONSTRUCTION: {os.path.basename(root_path)}")
    print(f"{'='*100}\n")
    
    print("Step 1: Scanning DICOM files...")
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
                is_axial = False
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if np.allclose(vec_z, [0, 0, 1], atol=0.3) or np.allclose(vec_z, [0, 0, -1], atol=0.3):
                        is_axial = True
                
                if not is_axial:
                    continue
                
                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'desc': getattr(dcm, 'SeriesDescription', 'Unknown'),
                        'slice_thickness': float(getattr(dcm, 'SliceThickness', 5.0)),
                        'body_part': getattr(dcm, 'BodyPartExamined', 'Unknown')
                    }
            except:
                continue
    
    print(f"  Found {len(series_map)} axial series\n")
    
    # Load and analyze each series
    print("Step 2: Loading series and analyzing content...")
    all_series = []
    
    for uid, file_list in series_map.items():
        if len(file_list) < 10:  # Minimum threshold
            continue
        
        file_list.sort(key=lambda x: x[0])
        z_coords = [x[0] for x in file_list]
        z_min, z_max = min(z_coords), max(z_coords)
        z_span = abs(z_max - z_min)
        
        if z_span < 50:  # Too small
            continue
        
        try:
            desc = series_metadata[uid]['desc'][:50]
            print(f"  Loading: {desc}...")
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames([x[1] for x in file_list])
            img = reader.Execute()
            volume_arr = sitk.GetArrayFromImage(img)
            
            # Analyze anatomical content
            features = analyze_volume_content(volume_arr)
            region, confidence = classify_region(features, z_min, z_max, z_span)
            
            print(f"    → {region} (confidence: {confidence:.0%}, Z: {z_min:.0f} to {z_max:.0f}mm)")
            
            series_info = {
                'uid': uid,
                'desc': series_metadata[uid]['desc'],
                'z_min': z_min,
                'z_max': z_max,
                'z_center': (z_min + z_max) / 2,
                'z_span': z_span,
                'num_slices': len(file_list),
                'slice_thickness': series_metadata[uid]['slice_thickness'],
                'region': region,
                'confidence': confidence,
                'features': features,
                'image': img,
                'volume_arr': volume_arr
            }
            
            all_series.append(series_info)
            
        except Exception as e:
            print(f"    → Failed: {e}")
            continue
    
    print(f"\n  Successfully loaded {len(all_series)} series\n")
    return all_series


# ==================== ANATOMICAL ANALYSIS ====================

def analyze_volume_content(volume_arr):
    """
    Analyze 3D volume for anatomical features
    Returns feature dictionary
    """
    features = {
        'has_skull': False,
        'has_ribs': False,
        'has_pelvis_bone': False,
        'has_leg_separation': False,
        'has_feet': False,
        'avg_width': 0,
        'width_std': 0,
        'bone_density_top': 0,
        'bone_density_mid': 0,
        'bone_density_bottom': 0,
        'body_volume': 0
    }
    
    nz, ny, nx = volume_arr.shape
    
    # Sample slices
    n_samples = min(nz, 20)
    sample_indices = np.linspace(0, nz-1, n_samples, dtype=int)
    
    widths = []
    
    for idx in sample_indices:
        slice_2d = volume_arr[idx, :, :]
        body_mask = (slice_2d > -500) & (slice_2d < 2000)
        
        if np.sum(body_mask) < 100:
            widths.append(0)
            continue
        
        # Width measurement
        horizontal_proj = np.sum(body_mask, axis=0)
        width = np.sum(horizontal_proj > 0)
        widths.append(width)
    
    widths_arr = np.array([w for w in widths if w > 0])
    
    if len(widths_arr) > 0:
        features['avg_width'] = np.mean(widths_arr)
        features['width_std'] = np.std(widths_arr)
    
    # Bone analysis in different regions
    top_section = volume_arr[:nz//3, :, :]
    mid_section = volume_arr[nz//3:2*nz//3, :, :]
    bottom_section = volume_arr[2*nz//3:, :, :]
    
    def bone_density(section):
        body = (section > -500) & (section < 2000)
        bone = section > 200
        if np.sum(body) > 0:
            return np.sum(bone) / np.sum(body)
        return 0
    
    features['bone_density_top'] = bone_density(top_section)
    features['bone_density_mid'] = bone_density(mid_section)
    features['bone_density_bottom'] = bone_density(bottom_section)
    
    # Skull: high bone density at top with specific pattern
    if features['bone_density_top'] > 0.15 and features['avg_width'] < 350:
        features['has_skull'] = True
    
    # Ribs: moderate bone density in mid-section with variation
    if features['bone_density_mid'] > 0.05 and features['width_std'] > 20:
        features['has_ribs'] = True
    
    # Pelvis: wide structure with bone in bottom section
    if features['bone_density_bottom'] > 0.08 and features['avg_width'] > 200:
        features['has_pelvis_bone'] = True
    
    # Leg separation: check bottom slices for two distinct regions
    bottom_test_idx = min(int(nz * 0.8), nz-1)
    if bottom_test_idx < nz:
        test_slice = volume_arr[bottom_test_idx, :, :]
        body_mask = (test_slice > -500) & (test_slice < 2000)
        labeled, n_regions = label(body_mask)
        if n_regions >= 2:
            features['has_leg_separation'] = True
    
    # Feet: very narrow average width
    if features['avg_width'] < 100:
        features['has_feet'] = True
    
    return features


def classify_region(features, z_min, z_max, z_span):
    """
    Classify anatomical region based on features and Z-position
    CRITICAL: Z-position is primary discriminator
    Returns (region_name, confidence)
    """
    z_center = (z_min + z_max) / 2
    
    # FIRST: Check Z-position to determine broad category
    # Positive Z = upper body (head/thorax)
    # Near-zero Z = mid-body (abdomen/pelvis)
    # Negative Z = lower body (pelvis/legs)
    
    is_upper = z_center > 200  # Head/thorax region
    is_mid = -200 <= z_center <= 200  # Abdomen/pelvis region
    is_lower = z_center < -200  # Pelvis/legs region
    
    # Priority 1: Skull detection (must be upper region)
    if features['has_skull'] and is_upper:
        return 'HEAD_THORAX', 0.95
    
    # Priority 2: Legs/feet (must be lower region)
    if (features['has_leg_separation'] or features['has_feet']) and is_lower:
        if features['has_feet'] or features['avg_width'] < 120:
            return 'LEGS_FEET', 0.90
        else:
            return 'PELVIS_LEGS', 0.85
    
    # Priority 3: Ribs (thorax) - but only if Z-position makes sense
    if features['has_ribs']:
        if is_upper:
            if z_span > 800 or z_max > 800:
                return 'HEAD_THORAX', 0.90
            else:
                return 'THORAX_UPPER', 0.85
        elif is_mid:
            return 'THORAX_ABDOMEN', 0.75
        # If ribs detected in lower region, it's probably misclassification
    
    # Priority 4: Pelvis bone
    if features['has_pelvis_bone']:
        if is_mid:
            return 'ABDOMEN_PELVIS', 0.80
        elif is_lower:
            return 'PELVIS_LEGS', 0.80
    
    # Fallback based on Z-position and size
    if is_upper:
        if z_span > 800:
            return 'HEAD_THORAX', 0.65
        elif z_span > 400:
            return 'THORAX_UPPER', 0.60
        else:
            return 'THORAX_ABDOMEN', 0.55
    elif is_mid:
        if z_center > 0:
            return 'THORAX_ABDOMEN', 0.55
        else:
            return 'ABDOMEN_PELVIS', 0.55
    else:  # is_lower
        if z_center < -500:
            return 'LEGS_FEET', 0.60
        else:
            return 'PELVIS_LEGS', 0.60
    
    return 'UNKNOWN', 0.30


# ==================== SERIES SELECTION ====================

def resolve_overlaps(all_series):
    """
    Handle overlapping series - keep best one for each region
    """
    print("Step 3: Resolving overlapping series...")
    
    # Group by region type
    region_groups = defaultdict(list)
    for s in all_series:
        region_groups[s['region']].append(s)
    
    selected = []
    
    for region, series_list in region_groups.items():
        if len(series_list) == 1:
            selected.append(series_list[0])
            print(f"  {region}: 1 series (unique)")
        else:
            # Pick best: highest confidence * longest span
            scored = [(s, s['confidence'] * s['z_span']) for s in series_list]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][0]
            selected.append(best)
            print(f"  {region}: Selected best of {len(series_list)} (score: {scored[0][1]:.0f})")
    
    print(f"\n  Final selection: {len(selected)} series\n")
    return selected


# ==================== ORDERING ====================

def order_series_cranial_to_caudal(series_list):
    """
    Order series from head to feet based on Z-position
    """
    print("Step 4: Ordering series (head → feet)...")
    
    # Define anatomical order priority
    region_priority = {
        'HEAD_THORAX': 1000,
        'THORAX_UPPER': 900,
        'THORAX_ABDOMEN': 800,
        'ABDOMEN_PELVIS': 700,
        'PELVIS': 600,
        'PELVIS_LEGS': 500,
        'LEGS_FEET': 400,
        'UNKNOWN': 0
    }
    
    # Sort by: 1) region priority, 2) Z-center (descending = top to bottom)
    sorted_series = sorted(series_list, 
                          key=lambda s: (region_priority.get(s['region'], 0), s['z_center']), 
                          reverse=True)
    
    print()
    for i, s in enumerate(sorted_series, 1):
        print(f"  {i}. {s['region']:<20} | Z: {s['z_min']:7.1f} to {s['z_max']:7.1f} mm | {s['num_slices']:4} slices")
    
    print()
    return sorted_series


# ==================== STITCHING ====================

def stitch_series_chain(ordered_series):
    """
    Stitch all series together with overlap detection
    """
    if len(ordered_series) == 0:
        return None
    
    print("Step 5: Stitching series together...")
    
    if len(ordered_series) == 1:
        print("  Only one series - no stitching needed\n")
        return ordered_series[0]['image']
    
    # Start with first (most cranial)
    combined_arr = ordered_series[0]['volume_arr'].copy()
    base_spacing = ordered_series[0]['image'].GetSpacing()
    base_origin = ordered_series[0]['image'].GetOrigin()
    base_direction = ordered_series[0]['image'].GetDirection()
    
    print(f"  Base: {ordered_series[0]['region']} ({len(combined_arr)} slices)")
    
    for i in range(1, len(ordered_series)):
        next_series = ordered_series[i]
        next_arr = next_series['volume_arr']
        
        # Find overlap
        max_overlap_search = min(len(combined_arr) // 3, len(next_arr) // 3, 80)
        best_overlap = 0
        best_corr = -9999
        
        for overlap in range(0, max_overlap_search + 1):
            if overlap == 0:
                corr = 0
            else:
                tail = combined_arr[-overlap:]
                head = next_arr[:overlap]
                
                # Simple correlation
                tail_flat = tail.flatten()
                head_flat = head.flatten()
                
                if len(tail_flat) > 0 and len(head_flat) > 0:
                    corr = np.corrcoef(tail_flat, head_flat)[0, 1]
                    if np.isnan(corr):
                        corr = 0
                else:
                    corr = 0
            
            if corr > best_corr:
                best_corr = corr
                best_overlap = overlap
        
        # Concatenate
        combined_arr = np.concatenate((combined_arr, next_arr[best_overlap:]), axis=0)
        
        print(f"  + {next_series['region']:<20} | Overlap: {best_overlap:3} slices (corr: {best_corr:5.3f}) → Total: {len(combined_arr)} slices")
    
    # Convert back to SimpleITK
    result_img = sitk.GetImageFromArray(combined_arr)
    result_img.SetSpacing(base_spacing)
    result_img.SetOrigin(base_origin)
    result_img.SetDirection(base_direction)
    
    total_length_cm = len(combined_arr) * base_spacing[2] / 10
    print(f"\n  ✓ Complete assembly: {len(combined_arr)} slices ({total_length_cm:.1f} cm)\n")
    
    return result_img


# ==================== VALIDATION ====================

def validate_completeness(ordered_series):
    """
    Check if body assembly is complete
    """
    print("Step 6: Validating anatomical completeness...")
    
    regions = [s['region'] for s in ordered_series]
    
    has_head = any('HEAD' in r for r in regions)
    has_thorax = any('THORAX' in r for r in regions)
    has_abdomen = any('ABDOMEN' in r or 'PELVIS' in r for r in regions)
    has_legs = any('LEGS' in r or 'FEET' in r for r in regions)
    
    completeness = sum([has_head, has_thorax, has_abdomen, has_legs]) / 4.0
    
    print(f"\n  Anatomical Coverage:")
    print(f"    Head/Neck:       {'✓' if has_head else '✗'}")
    print(f"    Thorax:          {'✓' if has_thorax else '✗'}")
    print(f"    Abdomen/Pelvis:  {'✓' if has_abdomen else '✗'}")
    print(f"    Legs/Feet:       {'✓' if has_legs else '✗'}")
    
    print(f"\n  Completeness Score: {completeness:.0%}")
    
    if completeness >= 0.75:
        status = "✓ GOOD - Body appears complete"
    elif completeness >= 0.50:
        status = "⚠ FAIR - Some body regions missing"
    else:
        status = "✗ POOR - Major body regions missing"
    
    print(f"  Status: {status}\n")
    
    return completeness


# ==================== RENDERING ====================

def render_final_volume(sitk_img, title, completeness):
    """
    Render the complete assembled body
    """
    print("Step 7: Rendering 3D volume...")
    
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
    opacity.AddPoint(-100, 0.05)
    opacity.AddPoint(40, 0.2)
    opacity.AddPoint(200, 0.5)
    opacity.AddPoint(500, 0.9)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-100, 0.7, 0.5, 0.4)
    color.AddRGBPoint(40, 0.9, 0.3, 0.3)
    color.AddRGBPoint(200, 1.0, 0.95, 0.95)
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
    render_window.SetSize(1400, 1400)
    render_window.SetWindowName(f"{title} | Completeness: {completeness:.0%}")
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
        cam.SetPosition(center[0], center[1] - 3500, center[2])
        cam.SetViewUp(0, 0, -1)
        renderer.ResetCameraClippingRange()
    
    print("  Opening 3D viewer...\n")
    render_window.Render()
    interactor.Start()


# ==================== MAIN PIPELINE ====================

def main():
    """Complete body reconstruction pipeline"""
    target_path = os.path.join(BASE_PATH, DATASET)
    
    # Step 1-2: Load all series
    all_series = load_all_axial_series(target_path)
    
    if len(all_series) == 0:
        print("✗ No valid series found\n")
        return
    
    # Step 3: Resolve overlaps
    selected_series = resolve_overlaps(all_series)
    
    if len(selected_series) == 0:
        print("✗ No series after overlap resolution\n")
        return
    
    # Step 4: Order head → feet
    ordered_series = order_series_cranial_to_caudal(selected_series)
    
    # Step 5: Stitch together
    final_volume = stitch_series_chain(ordered_series)
    
    if final_volume is None:
        print("✗ Stitching failed\n")
        return
    
    # Step 6: Validate
    completeness = validate_completeness(ordered_series)
    
    # Step 7: Render
    render_final_volume(final_volume, f"Complete Body: {DATASET}", completeness)
    
    print(f"{'='*100}")
    print(f"✓ RECONSTRUCTION COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
