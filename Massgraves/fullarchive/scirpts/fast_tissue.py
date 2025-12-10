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
target_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/DICOM-Joop" 

def get_verified_series(root_path):
    print("1. Scanning for valid Axial volumes...")
    series_map = defaultdict(list)
    
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
            except: continue

    valid_series = []
    print(f"   -> Found {len(series_map)} candidates.")
    
    for uid, data in series_map.items():
        if len(data) < 20: continue
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 50.0: continue 
        
        valid_series.append({
            'uid': uid, 'paths': [x[1] for x in data], 
            'span': span, 'z_min': z_min, 'z_max': z_max
        })
    return valid_series

def remove_subsets(series_list):
    print(f"2. Deduplicating {len(series_list)} series...")
    series_list.sort(key=lambda x: x['span'], reverse=True)
    final = []
    covered = []
    
    for s in series_list:
        is_subset = False
        for c_min, c_max in covered:
            overlap = max(0, min(s['z_max'], c_max) - max(s['z_min'], c_min))
            if s['span'] > 0 and (overlap / s['span']) > 0.8: is_subset = True; break
        
        if not is_subset:
            final.append(s)
            covered.append((s['z_min'], s['z_max']))
            
    return final

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

def perform_sliding_stitch(torso_vol, legs_vol):
    print("\n3. Starting Alignment Pipeline...")
    
    reader = sitk.ImageSeriesReader()
    print("   -> Loading Torso...")
    reader.SetFileNames(torso_vol['paths'])
    torso_img = reader.Execute()
    
    print("   -> Loading Legs...")
    reader.SetFileNames(legs_vol['paths'])
    legs_img = reader.Execute()
    
    # STEP A: Rough Align Legs to Torso (so comparison makes sense)
    legs_img = align_xy_centers(torso_img, legs_img)
    
    # STEP B: Rotate Torso (The Correction)
    torso_img = optimize_torso_rotation(torso_img, legs_img)
    
    # STEP C: Re-Align Legs to Rotated Torso (Fine tune X/Y)
    print("   -> Phase 3: Re-aligning X/Y...")
    legs_img = align_xy_centers(torso_img, legs_img)
    
    # STEP D: Slide Z
    print("   -> Phase 4: Sliding Z-Optimization...")
    fixed_arr = sitk.GetArrayFromImage(torso_img) 
    moving_arr = sitk.GetArrayFromImage(legs_img)
    
    fixed_len = len(fixed_arr)
    max_overlap = min(fixed_len // 2, 150) 
    best_score = -1.0
    best_overlap = 0
    
    print("-" * 60)
    print(f"{'OVERLAP':<10} | {'SCORE':<10}")
    print("-" * 60)
    
    for k in range(1, max_overlap + 1):
        block_torso = fixed_arr[-k:]
        block_legs  = moving_arr[:k]
        score = calculate_volume_similarity(block_torso, block_legs)
        
        if score > best_score:
            best_score = score
            best_overlap = k
            
        if k % 10 == 0 or score > 0.8:
            print(f"{k:<10} | {score:.5f}")
            
    print(f"   -> OPTIMAL MATCH: {best_overlap} slices overlap (Score: {best_score:.4f})")
    
    part_a = fixed_arr
    part_b = moving_arr[best_overlap:]
    stitched_arr = np.concatenate((part_a, part_b), axis=0)
    
    stitched_img = sitk.GetImageFromArray(stitched_arr)
    stitched_img.SetOrigin(torso_img.GetOrigin())
    stitched_img.SetSpacing(torso_img.GetSpacing())
    stitched_img.SetDirection(torso_img.GetDirection())
    
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

series_data.sort(key=lambda x: x['z_min'])
to_render = remove_subsets(series_data)

if len(to_render) >= 2:
    torso = to_render[0]
    legs = to_render[1]
    final_sitk_img = perform_sliding_stitch(torso, legs)
else:
    print("Error: Could not isolate Torso and Legs.")
    sys.exit(1)

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

print("4. Rendering Final Stitched Result...")
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