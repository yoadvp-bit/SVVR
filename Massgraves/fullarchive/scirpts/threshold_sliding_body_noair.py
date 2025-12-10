import SimpleITK as sitk
import vtk
import os
import sys
import pydicom
import numpy as np
from vtk.util import numpy_support
from collections import defaultdict

# --- CONFIGURATION ---
target_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/DICOM-Jan" 

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

# --- CORE ALGORITHM: SLIDING DENSITY OPTIMIZATION ---

def calculate_volume_similarity(vol_a, vol_b):
    """
    Computes Normalized Cross-Correlation (NCC) between two 3D blocks.
    
    UPDATE: Now applies a threshold to ignore Air/Background noise.
    Only structures > -500 HU (Tissue/Bone) drive the alignment.
    """
    # Flatten both volumes into 1D arrays
    a = vol_a.flatten().astype(np.float32)
    b = vol_b.flatten().astype(np.float32)
    
    # --- FILTERING STEP ---
    # Clamp values below -500 (Air is ~-1000) to a flat baseline.
    # This removes "noise" in the empty space so it doesn't affect the score.
    threshold = -500.0
    a[a < threshold] = threshold
    b[b < threshold] = threshold
    # ----------------------
    
    # Normalize (subtract mean)
    a -= np.mean(a)
    b -= np.mean(b)
    
    # Compute Cosine Similarity
    numerator = np.sum(a * b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    
    if denominator == 0: return 0.0
    return numerator / denominator

def scan_xy_alignment(vol_a, vol_b, search_radius=15):
    """
    For a fixed Z-overlap, search for the best X/Y shift.
    vol_a: Fixed block (Torso chunk)
    vol_b: Moving block (Legs chunk)
    Increased search radius to 15 to account for patient shifting.
    """
    best_sim = -1.0
    best_shift = (0, 0)
    
    h, w = vol_a.shape[1], vol_a.shape[2]
    
    # Iterate Y shift (step 2 for speed optimization)
    for dy in range(-search_radius, search_radius + 1, 2): 
        # Iterate X shift
        for dx in range(-search_radius, search_radius + 1, 2):
            
            # Slice A: Valid region in A's frame
            sy_a = slice(max(0, -dy), min(h, h-dy))
            sx_a = slice(max(0, -dx), min(w, w-dx))
            
            # Slice B: The shifted region
            sy_b = slice(max(0, dy), min(h, h+dy))
            sx_b = slice(max(0, dx), min(w, w+dx))
            
            sub_a = vol_a[:, sy_a, sx_a]
            sub_b = vol_b[:, sy_b, sx_b]
            
            if sub_a.size == 0: continue
            
            score = calculate_volume_similarity(sub_a, sub_b)
            
            if score > best_sim:
                best_sim = score
                best_shift = (dx, dy)
                
    return best_sim, best_shift

def perform_sliding_stitch(fixed_series, moving_series):
    print("\n3. Running 3D Sliding Optimization (Z, X, Y)...")
    
    # 1. Load Data into Memory
    reader = sitk.ImageSeriesReader()
    
    print("   -> Loading Torso (Fixed)...")
    reader.SetFileNames(fixed_series['paths'])
    fixed_img = reader.Execute()
    # (Z, Y, X)
    fixed_arr = sitk.GetArrayFromImage(fixed_img) 
    
    print("   -> Loading Legs (Moving)...")
    reader.SetFileNames(moving_series['paths'])
    moving_img = reader.Execute()
    moving_arr = sitk.GetArrayFromImage(moving_img)
    
    # 2. Define Search Parameters
    fixed_len = len(fixed_arr)
    # Limit overlap search to reasonable bounds (e.g., 100 slices) to avoid false positives
    # deep inside the body
    max_overlap = min(fixed_len // 2, 120) 
    
    best_global_score = -1.0
    best_z_overlap = 0
    best_xy_shift = (0, 0)
    
    print(f"   -> Max Search Window: {max_overlap} slices overlap")
    print("-" * 90)
    print(f"{'Z-OVL':<6} | {'XY-SHIFT':<12} | {'SCORE':<10} | {'STATUS'}")
    print("-" * 90)
    
    # 3. The Sliding Loop (Z-Axis)
    for k in range(1, max_overlap + 1):
        
        # Extract Z-overlapping slabs (No X/Y shift yet)
        block_torso = fixed_arr[-k:]
        block_legs  = moving_arr[:k]
        
        # INNER SEARCH: Find best X/Y alignment for this Z-overlap
        # Search radius of 15 pixels (~10-15mm) covers most patient movement
        score, (dx, dy) = scan_xy_alignment(block_torso, block_legs, search_radius=15)
        
        if score > best_global_score:
            best_global_score = score
            best_z_overlap = k
            best_xy_shift = (dx, dy)
            
        # Log output
        if k % 5 == 0 or score > 0.85:
            status = "New Best!" if score == best_global_score else ""
            shift_str = f"({dx},{dy})"
            print(f"{k:<6} | {shift_str:<12} | {score:.5f}    | {status}")
            
    print("-" * 90)
    print(f"   -> OPTIMAL MATCH FOUND!")
    print(f"      Best Z-Overlap: {best_z_overlap} slices")
    print(f"      Best X/Y Shift: {best_xy_shift} pixels")
    print(f"      Max Similarity: {best_global_score:.5f}")
    
    # 4. Construct the Stitched Volume
    print("   -> Fusing Volumes...")
    
    # 4a. Resample Moving Image (Legs) to align X/Y with Torso
    spacing = moving_img.GetSpacing()
    phys_shift_x = best_xy_shift[0] * spacing[0]
    phys_shift_y = best_xy_shift[1] * spacing[1]
    
    transform = sitk.TranslationTransform(3)
    transform.SetOffset((phys_shift_x, phys_shift_y, 0))
    
    ref_size = list(fixed_img.GetSize())
    ref_size[2] = moving_img.GetSize()[2] 
    
    ref_origin = list(fixed_img.GetOrigin())
    ref_origin[2] = moving_img.GetOrigin()[2] 
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetOutputOrigin(ref_origin)
    resampler.SetSize(ref_size)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000)
    
    aligned_legs_img = resampler.Execute(moving_img)
    aligned_legs_arr = sitk.GetArrayFromImage(aligned_legs_img)
    
    # 4b. Splice Arrays
    part_a = fixed_arr # Keep all Torso
    part_b = aligned_legs_arr[best_z_overlap:] # Keep Legs after overlap
    
    stitched_arr = np.concatenate((part_a, part_b), axis=0)
    
    stitched_img = sitk.GetImageFromArray(stitched_arr)
    stitched_img.SetOrigin(fixed_img.GetOrigin())
    stitched_img.SetSpacing(fixed_img.GetSpacing())
    stitched_img.SetDirection(fixed_img.GetDirection())
    
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
    
    # Matrix
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

# Sort: Torso (Lower Z start) first, Legs (Higher Z start) second
series_data.sort(key=lambda x: x['z_min'])
to_render = remove_subsets(series_data)

if len(to_render) >= 2:
    torso = to_render[0]
    legs = to_render[1]
    
    # --- RUN THE ALGORITHM ---
    final_sitk_img = perform_sliding_stitch(torso, legs)
else:
    print("Error: Could not isolate Torso and Legs.")
    sys.exit(1)

# --- VISUALIZATION ---
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.05, 0.05, 0.05)

# --- UPDATED: LOW DENSITY PRESET (Skin & Muscle) ---
opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(-1000, 0.0) # Air
opacity.AddPoint(-500, 0.0)  # Background noise
opacity.AddPoint(-200, 0.05) # Skin starts (very faint)
opacity.AddPoint(-100, 0.2)  # Fat/Tissue
opacity.AddPoint(50, 0.4)    # Muscle
opacity.AddPoint(250, 0.6)   # Bone starts
opacity.AddPoint(1000, 0.9)  # Dense Bone

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)      # Black
color.AddRGBPoint(-200, 0.8, 0.6, 0.5)       # Skin (Beige)
color.AddRGBPoint(0, 0.7, 0.3, 0.3)          # Muscle (Reddish)
color.AddRGBPoint(300, 0.95, 0.95, 0.9)      # Bone (White)

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

# Camera
print("5. Positioning Camera...")
render_window = vtk.vtkRenderWindow()
render_window.SetSize(1200, 1200)
render_window.SetWindowName("Density-Optimized Stitch")
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

# Get bounds of the new single stitched volume
volume.Update()
bounds = volume.GetBounds()

if bounds:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
    
    cam = renderer.GetActiveCamera()
    cam.SetFocalPoint(center)
    # Front View (-Y)
    cam.SetPosition(center[0], center[1] - 3500, center[2]) 
    # Head Up (-Z is Up for raw DICOM supine data)
    cam.SetViewUp(0, 0, -1) 
    
    renderer.ResetCameraClippingRange()

print("   -> Rendering...")
render_window.Render()
interactor.Start()