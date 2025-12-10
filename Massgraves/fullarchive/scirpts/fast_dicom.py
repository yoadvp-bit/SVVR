import SimpleITK as sitk
import vtk
import os
import sys
import pydicom
import numpy as np
import math
from vtk.util import numpy_support
from collections import defaultdict

# --- CONFIGURATION ---
# Select dataset: "DICOM-Maria", "DICOM-Jan", or "DICOM-Jarek"
DATASET = "DICOM-Maria"
target_folder = rf"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/{DATASET}"
print(f"=== Processing Dataset: {DATASET} ===\n")

def get_verified_series(root_path):
    """Enhanced series detection with metadata extraction"""
    print("1. Scanning for valid Axial volumes...")
    series_map = defaultdict(list)
    series_metadata = {}
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # Filter: Must be AXIAL
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    if not np.allclose(vec_z, [0, 0, 1], atol=0.2): 
                        continue 

                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                # Store metadata on first encounter
                if uid not in series_metadata:
                    series_metadata[uid] = {
                        'description': getattr(dcm, 'SeriesDescription', '').upper(),
                        'body_part': getattr(dcm, 'BodyPartExamined', '').upper(),
                    }
            except: continue

    valid_series = []
    print(f"   -> Found {len(series_map)} candidates.")
    
    for uid, data in series_map.items():
        if len(data) < 20: continue
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 50.0: continue 
        
        metadata = series_metadata.get(uid, {})
        valid_series.append({
            'uid': uid, 
            'paths': [x[1] for x in data], 
            'span': span, 
            'z_min': z_min, 
            'z_max': z_max,
            'z_center': (z_min + z_max) / 2,
            'description': metadata.get('description', ''),
            'body_part': metadata.get('body_part', ''),
            'num_slices': len(data)
        })
    return valid_series

def classify_body_region(series):
    """Classify series as HEAD, THORAX, ABDOMEN, PELVIS, or LEGS"""
    desc = series['description']
    body = series['body_part']
    z_center = series['z_center']
    z_min = series['z_min']
    z_max = series['z_max']
    
    # Keywords for classification
    if 'HEAD' in desc or 'BRAIN' in desc or 'SKULL' in desc:
        return 'HEAD', 1000
    elif 'THORAX' in desc or 'CHEST' in desc or 'LUNG' in desc or 'THORAX' in body:
        return 'THORAX', 500
    elif 'ABDOMEN' in desc or 'LIVER' in desc or 'ABDOMEN' in body:
        return 'ABDOMEN', 0
    elif 'PELVIS' in desc or 'PELVIS' in body:
        return 'PELVIS', -300
    elif 'LEG' in desc or 'FEMUR' in desc or 'KNEE' in desc or 'LOWER' in desc or 'EXTREMITY' in body:
        return 'LEGS', -700
    
    # Z-position based heuristics (when metadata is unclear)
    if z_center > 600:
        return 'HEAD', 1000
    elif z_center > 200:
        return 'THORAX', 500
    elif z_center > -200:
        return 'ABDOMEN', 0
    elif z_center > -600:
        return 'PELVIS', -300
    else:
        return 'LEGS', -700

def remove_subsets(series_list):
    """Remove duplicate/subset series and classify body regions"""
    print(f"2. Analyzing {len(series_list)} series...")
    
    # Classify all series
    for s in series_list:
        region, priority = classify_body_region(s)
        s['region'] = region
        s['priority'] = priority
    
    # Sort by span (prefer larger scans)
    series_list.sort(key=lambda x: x['span'], reverse=True)
    final = []
    covered = []
    
    for s in series_list:
        is_subset = False
        for c_min, c_max in covered:
            overlap = max(0, min(s['z_max'], c_max) - max(s['z_min'], c_min))
            if s['span'] > 0 and (overlap / s['span']) > 0.8: 
                is_subset = True
                break
        
        if not is_subset:
            final.append(s)
            covered.append((s['z_min'], s['z_max']))
            print(f"   -> Series: {s['region']:10s} | Z: {s['z_min']:7.1f} to {s['z_max']:7.1f} | Slices: {s['num_slices']}")
            
    return final

def find_best_pair_to_stitch(series_list):
    """Intelligently find which two series should be stitched together"""
    print("\n3. Finding optimal series pair for stitching...")
    
    # Group by region
    by_region = {}
    for s in series_list:
        region = s['region']
        if region not in by_region:
            by_region[region] = []
        by_region[region].append(s)
    
    # Strategy 1: Look for Z-gap between consecutive series
    # Sort all series by Z CENTER (not min!)  - upper body has HIGHER Z
    sorted_series = sorted(series_list, key=lambda x: x['z_center'], reverse=True)
    
    best_pair = None
    best_score = -1
    
    for i in range(len(sorted_series) - 1):
        upper = sorted_series[i]  # Higher Z center = anatomically superior
        lower = sorted_series[i+1]  # Lower Z center = anatomically inferior
        
        # Calculate gap at the interface (bottom of upper vs top of lower)
        # Bottom of upper body part
        upper_bottom = upper['z_min']
        # Top of lower body part  
        lower_top = lower['z_max']
        
        gap = upper_bottom - lower_top  # Positive = gap, Negative = overlap
        
        # Check if there's a reasonable interface
        if -200 < gap < 200:  # Allow overlap or gap up to 200mm
            # Score based on: size, gap, and body part logic
            size_score = min(upper['span'], lower['span']) / 500.0  # Prefer larger scans
            gap_score = 1.0 / (1.0 + abs(gap) / 50.0)  # Prefer smaller gaps
            
            # Bonus if it's a logical body connection
            region_score = 0.0
            if (upper['region'] == 'THORAX' and lower['region'] in ['ABDOMEN', 'PELVIS', 'LEGS']) or \
               (upper['region'] == 'ABDOMEN' and lower['region'] in ['PELVIS', 'LEGS']) or \
               (upper['region'] == 'PELVIS' and lower['region'] == 'LEGS'):
                region_score = 2.0
            
            total_score = size_score + gap_score + region_score
            
            if total_score > best_score:
                best_score = total_score
                best_pair = (upper, lower)
                
    if best_pair:
        upper, lower = best_pair
        gap = upper['z_min'] - lower['z_max']
        print(f"   -> Selected UPPER: {upper['region']} (Z: {upper['z_min']:.1f} to {upper['z_max']:.1f})")
        print(f"   -> Selected LOWER: {lower['region']} (Z: {lower['z_min']:.1f} to {lower['z_max']:.1f})")
        if gap > 0:
            print(f"   -> Gap at interface: {gap:.1f} mm")
        else:
            print(f"   -> Overlap at interface: {abs(gap):.1f} mm")
        return upper, lower
    
    # Fallback: Just use first two by Z center
    if len(sorted_series) >= 2:
        print("   -> Warning: No clear connection found, using top two series by Z-position")
        return sorted_series[0], sorted_series[1]
    
    return None, None

# --- ALIGNMENT HELPERS ---

def get_slice_center_of_mass(slice_img):
    """ Calculates the physical (x,y) center of mass of tissue in the slice. """
    mask = sitk.BinaryThreshold(slice_img, lowerThreshold=-500, upperThreshold=3000)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if stats.GetNumberOfLabels() > 0:
        return stats.GetCentroid(1) 
    return slice_img.GetOrigin()[:2]

def get_min_physical_x(img):
    """ 
    Calculates the 'Least Positive X' (Minimum X) of the volume's bounding box in physical space.
    In standard DICOM (LPS), -X is the Patient's RIGHT.
    """
    sz = img.GetSize()
    # Check corners (0,0,0) and (Width,0,0) to see which is physically smaller X
    p0 = img.TransformIndexToPhysicalPoint((0,0,0))
    p1 = img.TransformIndexToPhysicalPoint((sz[0],0,0))
    return min(p0[0], p1[0])

def calculate_masked_similarity(vol_a, vol_b):
    """ NCC IGNORING AIR (< -500 HU) """
    a = vol_a.flatten().astype(np.float32)
    b = vol_b.flatten().astype(np.float32)
    
    threshold = -500.0
    mask = (a > threshold) & (b > threshold)
    
    if np.sum(mask) == 0: return 0.0
        
    a_tissue = a[mask]
    b_tissue = b[mask]
    a_tissue -= np.mean(a_tissue)
    b_tissue -= np.mean(b_tissue)
    
    numerator = np.sum(a_tissue * b_tissue)
    denominator = np.linalg.norm(a_tissue) * np.linalg.norm(b_tissue)
    
    if denominator == 0: return 0.0
    return numerator / denominator

def calculate_volume_similarity(vol_a, vol_b):
    """ Standard NCC with clamping """
    a = vol_a.flatten().astype(np.float32)
    b = vol_b.flatten().astype(np.float32)
    threshold = -500.0
    a[a < threshold] = threshold
    b[b < threshold] = threshold
    a -= np.mean(a)
    b -= np.mean(b)
    numerator = np.sum(a * b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0: return 0.0
    return numerator / denominator

def align_xy_centers(fixed_img, moving_img):
    """ Aligns Moving image to Fixed image in X and Y using Center of Mass. """
    # print("   -> Aligning X/Y Centers...")
    fixed_size = fixed_img.GetSize()
    fixed_slice = fixed_img[:, :, fixed_size[2]-1]
    moving_slice = moving_img[:, :, 0]
    
    com_fixed = get_slice_center_of_mass(fixed_slice)
    com_moving = get_slice_center_of_mass(moving_slice)
    
    dx = com_fixed[0] - com_moving[0]
    dy = com_fixed[1] - com_moving[1]
    
    transform = sitk.TranslationTransform(3)
    transform.SetOffset((-dx, -dy, 0))
    
    ref_origin = list(fixed_img.GetOrigin())
    ref_origin[2] = moving_img.GetOrigin()[2]
    ref_size = list(fixed_img.GetSize())
    ref_size[2] = moving_img.GetSize()[2]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetOutputOrigin(ref_origin)
    resampler.SetSize(ref_size)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000)
    
    return resampler.Execute(moving_img)

def optimize_torso_rotation(torso_img, legs_img):
    """ 
    Rotates the TORSO (Fixed) to match the Legs.
    Pivot: Least Positive X Axis (Minimum X) of the Torso.
    """
    print("   -> Phase 2: Optimizing Torso Rotation...")
    
    # Compare Interface Slabs (Bottom of Torso vs Top of Legs)
    t_sz = torso_img.GetSize()
    # Take bottom 5 slices of Torso
    ref_slab = torso_img[:, :, t_sz[2]-5 : t_sz[2]]
    # Take top 5 slices of Legs
    mov_slab = legs_img[:, :, 0:5]
    
    mov_arr = sitk.GetArrayFromImage(mov_slab)
    
    # Calculate Pivot Point (Hinge) - Using Min X 
    min_x = get_min_physical_x(torso_img)
    
    # Get Y center from the bottom slice
    bottom_slice = torso_img[:, :, t_sz[2]-1]
    com = get_slice_center_of_mass(bottom_slice)
    pivot_y = com[1]
    
    # The Z coordinate of the pivot doesn't affect Z-rotation geometry, 
    # but we use the Torso's origin Z for consistency.
    pivot_point = (min_x, pivot_y, torso_img.GetOrigin()[2])
    print(f"      Pivot Point (Min X, Center Y): ({min_x:.1f}, {pivot_y:.1f})")
    
    best_angle = 0.0
    best_score = -1.0
    best_deg_print = 0.0
    
    # Search angles
    print(f"      Searching rotation angles (-10 to +10 deg)...")
    for deg in range(-10, 11, 1): # +/- 10 degrees
        angle_rad = math.radians(deg)
        
        # Euler3DTransform rotates around the Z axis passing through 'Center'
        # when we set rotation (0, 0, angle). This creates an axis parallel to Z.
        transform = sitk.Euler3DTransform()
        transform.SetCenter(pivot_point)
        transform.SetRotation(0, 0, angle_rad)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_slab) # Resample Torso Slab onto itself
        resampler.SetTransform(transform)
        resampler.SetDefaultPixelValue(-1000)
        
        rotated_slab = resampler.Execute(ref_slab)
        ref_arr = sitk.GetArrayFromImage(rotated_slab)
        
        score = calculate_masked_similarity(ref_arr, mov_arr)
        
        if score > best_score:
            best_score = score
            best_angle = angle_rad
            best_deg_print = deg
            
    print(f"      Optimal Torso Rotation: {best_deg_print}° (Score: {best_score:.4f})")
    
    # Apply best rotation to FULL Torso
    final_transform = sitk.Euler3DTransform()
    final_transform.SetCenter(pivot_point)
    final_transform.SetRotation(0, 0, best_angle)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(torso_img)
    resampler.SetTransform(final_transform)
    resampler.SetDefaultPixelValue(-1000)
    
    return resampler.Execute(torso_img)

# --- CORE STITCHING PIPELINE ---

def perform_sliding_stitch(upper_vol, lower_vol):
    """
    Stitches upper body part to lower body part.
    upper_vol: The anatomically superior region (e.g., torso/thorax)
    lower_vol: The anatomically inferior region (e.g., legs/pelvis)
    """
    print("\n4. Starting Alignment Pipeline...")
    
    reader = sitk.ImageSeriesReader()
    print(f"   -> Loading {upper_vol['region']}...")
    reader.SetFileNames(upper_vol['paths'])
    upper_img = reader.Execute()
    
    print(f"   -> Loading {lower_vol['region']}...")
    reader.SetFileNames(lower_vol['paths'])
    lower_img = reader.Execute()
    
    # STEP A: Rough Align lower to upper (so comparison makes sense)
    lower_img = align_xy_centers(upper_img, lower_img)
    
    # STEP B: Rotate upper (The Correction)
    upper_img = optimize_torso_rotation(upper_img, lower_img)
    
    # STEP C: Re-Align lower to Rotated upper (Fine tune X/Y)
    print("   -> Phase 3: Re-aligning X/Y...")
    lower_img = align_xy_centers(upper_img, lower_img)
    
    # STEP D: Slide Z - with adaptive search
    print("   -> Phase 4: Sliding Z-Optimization...")
    upper_arr = sitk.GetArrayFromImage(upper_img) 
    lower_arr = sitk.GetArrayFromImage(lower_img)
    
    upper_len = len(upper_arr)
    lower_len = len(lower_arr)
    
    # Adaptive search window based on data
    max_overlap = min(upper_len // 2, lower_len // 2, 150)
    
    # Also allow for gap (negative overlap) if volumes don't touch
    min_overlap = max(-50, -upper_len // 4)  # Allow up to 50 slice gap
    
    best_score = -1.0
    best_overlap = 0
    
    print("-" * 60)
    print(f"{'OVERLAP':<10} | {'SCORE':<10} | {'DESCRIPTION'}")
    print("-" * 60)
    
    for k in range(min_overlap, max_overlap + 1):
        if k <= 0:
            # Negative overlap = gap between volumes
            # Compare last few slices of upper with first few of lower
            compare_size = min(10, upper_len, lower_len)
            block_upper = upper_arr[-compare_size:]
            block_lower = lower_arr[:compare_size]
            score = calculate_volume_similarity(block_upper, block_lower) * 0.5  # Penalize gaps
        else:
            # Positive overlap
            block_upper = upper_arr[-k:]
            block_lower = lower_arr[:k]
            score = calculate_volume_similarity(block_upper, block_lower)
        
        if score > best_score:
            best_score = score
            best_overlap = k
            
        if abs(k) % 10 == 0 or score > 0.7:
            if k <= 0:
                desc = f"GAP of {abs(k)} slices"
            else:
                desc = f"Overlap of {k} slices"
            print(f"{k:<10} | {score:.5f} | {desc}")
            
    if best_overlap <= 0:
        print(f"   -> OPTIMAL: GAP of {abs(best_overlap)} slices (Score: {best_score:.4f})")
        # Concatenate with gap (no overlap)
        stitched_arr = np.concatenate((upper_arr, lower_arr[abs(best_overlap):]), axis=0)
    else:
        print(f"   -> OPTIMAL: {best_overlap} slices overlap (Score: {best_score:.4f})")
        part_a = upper_arr
        part_b = lower_arr[best_overlap:]
        stitched_arr = np.concatenate((part_a, part_b), axis=0)
    
    stitched_img = sitk.GetImageFromArray(stitched_arr)
    stitched_img.SetOrigin(upper_img.GetOrigin())
    stitched_img.SetSpacing(upper_img.GetSpacing())
    stitched_img.SetDirection(upper_img.GetDirection())
    
    return stitched_img

def sitk_to_vtk(sitk_img):
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    dims = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims[0], dims[1], dims[2])
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(0,0,0)
    vtk_img.GetPointData().SetScalars(vtk_data)
    
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3): mat.SetElement(i, j, direction[i + j*3])
    mat.SetElement(0, 3, origin[0])
    mat.SetElement(1, 3, origin[1])
    mat.SetElement(2, 3, origin[2])
    
    return vtk_img, mat

# --- MAIN EXECUTION ---
series_data = get_verified_series(target_folder)
if len(series_data) < 2:
    print("Error: Need at least 2 valid series.")
    sys.exit(1)

# Deduplicate and classify
to_render = remove_subsets(series_data)

if len(to_render) < 2:
    print("Error: Need at least 2 distinct series after deduplication.")
    sys.exit(1)

# Find best pair to stitch
upper, lower = find_best_pair_to_stitch(to_render)

if upper is None or lower is None:
    print("Error: Could not identify appropriate series to stitch.")
    sys.exit(1)

# Perform the stitching
final_sitk_img = perform_sliding_stitch(upper, lower)

# --- VISUALIZATION ---
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.05, 0.05, 0.05)

opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(-1024, 0.00)
opacity.AddPoint(-200,  0.00)
opacity.AddPoint(-100,  0.10)
opacity.AddPoint(40,    0.25)
opacity.AddPoint(200,   0.50)
opacity.AddPoint(500,   1.00)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(-1024, 0.0, 0.0, 0.0) 
color.AddRGBPoint(-100,  0.8, 0.6, 0.5)
color.AddRGBPoint(40,    0.8, 0.2, 0.2)
color.AddRGBPoint(200,   1.0, 0.9, 0.9)
color.AddRGBPoint(500,   1.0, 1.0, 1.0)

print("5. Rendering Final Stitched Result...")
vtk_img_data, vtk_mat = sitk_to_vtk(final_sitk_img)

volume = vtk.vtkVolume()
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(vtk_img_data)
volume.SetMapper(mapper)
volume.SetUserMatrix(vtk_mat)

prop = vtk.vtkVolumeProperty()
prop.SetColor(color)
prop.SetScalarOpacity(opacity)
prop.ShadeOn()
prop.SetInterpolationTypeToLinear()
volume.SetProperty(prop)

renderer.AddVolume(volume)

render_window = vtk.vtkRenderWindow()
render_window.SetSize(1200, 1200)
render_window.SetWindowName("Corrected Stitch: Tissue & Bone")
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

print("   -> Rendering window opened.")
render_window.Render()
interactor.Start()