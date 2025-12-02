import SimpleITK as sitk
import vtk
import os
import sys
import pydicom
import numpy as np
from vtk.util import numpy_support
from collections import defaultdict
from scipy.ndimage import zoom, rotate, sobel

# --- CONFIGURATION ---
# Available datasets: DICOM-Jan, DICOM-Maria, DICOM-Jarek
DATASET_NAME = "DICOM-Jan"  # Change this to test different datasets
target_folder = rf"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/{DATASET_NAME}" 

def get_verified_series(root_path):
    print("1. Scanning for valid Axial volumes...")
    series_map = defaultdict(list)
    series_info = {}  # Store additional metadata
    
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                if 'ImagePositionPatient' not in dcm: continue
                
                # More flexible orientation check
                if 'ImageOrientationPatient' in dcm:
                    iop = dcm.ImageOrientationPatient
                    vec_z = np.cross([iop[0], iop[1], iop[2]], [iop[3], iop[4], iop[5]])
                    # Allow more variation in orientation
                    if not (np.allclose(vec_z, [0, 0, 1], atol=0.3) or 
                           np.allclose(vec_z, [0, 0, -1], atol=0.3)):
                        continue 

                uid = dcm.SeriesInstanceUID
                z = float(dcm.ImagePositionPatient[2])
                series_map[uid].append((z, filepath))
                
                # Store series metadata for better identification
                if uid not in series_info:
                    series_info[uid] = {
                        'series_description': getattr(dcm, 'SeriesDescription', '').upper(),
                        'body_part': getattr(dcm, 'BodyPartExamined', '').upper(),
                        'slice_thickness': getattr(dcm, 'SliceThickness', 0),
                        'pixel_spacing': getattr(dcm, 'PixelSpacing', [1,1])
                    }
                    
            except: continue

    valid_series = []
    print(f"   -> Found {len(series_map)} candidates.")
    
    for uid, data in series_map.items():
        if len(data) < 15: continue  # Slightly more lenient
        data.sort(key=lambda x: x[0])
        z_min, z_max = data[0][0], data[-1][0]
        span = abs(z_max - z_min)
        
        if span < 40.0: continue  # Slightly more lenient
        
        # Estimate body region based on Z position and metadata
        info = series_info.get(uid, {})
        avg_z = (z_min + z_max) / 2
        
        # Heuristics for torso vs legs identification
        is_likely_torso = (
            'CHEST' in info.get('series_description', '') or
            'ABDOMEN' in info.get('series_description', '') or
            'TORSO' in info.get('body_part', '') or
            avg_z > 0  # Often torso has higher Z values
        )
        
        is_likely_legs = (
            'LEG' in info.get('series_description', '') or
            'LOWER' in info.get('series_description', '') or
            'EXTREMITY' in info.get('body_part', '') or
            avg_z < 0  # Often legs have lower Z values
        )
        
        valid_series.append({
            'uid': uid, 'paths': [x[1] for x in data], 
            'span': span, 'z_min': z_min, 'z_max': z_max,
            'avg_z': avg_z, 'info': info,
            'is_likely_torso': is_likely_torso,
            'is_likely_legs': is_likely_legs
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

def align_xy_centers(fixed_img, moving_img):
    """
    Enhanced alignment with rotation testing.
    Aligns the Moving image (Legs) to the Fixed image (Torso) in X and Y
    by matching the Center of Mass of their interface slices.
    Returns: Resampled Moving Image (aligned grid).
    """
    print("   -> Checking X/Y Alignment with rotation testing...")

    # 1. Extract Interface Slices (Bottom of Torso, Top of Legs)
    fixed_size = fixed_img.GetSize()
    
    # Extract last Z slice of Fixed (Torso Bottom)
    fixed_slice = fixed_img[:, :, fixed_size[2]-1]
    # Extract first Z slice of Moving (Legs Top)
    moving_slice = moving_img[:, :, 0]
    
    # 2. Compute Center of Mass (CoM) based on Tissue (> -500 HU)
    def get_center_of_mass(slice_img):
        # Threshold to get body mask (Bone/Muscle/Skin) - Ignore Air
        mask = sitk.BinaryThreshold(slice_img, lowerThreshold=-500, upperThreshold=3000)
        
        # Get Statistics
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        
        if stats.GetNumberOfLabels() > 0:
            # Label 1 is the mask
            return stats.GetCentroid(1) # Returns (x, y) physical coords
        else:
            # Fallback to geometric center if image is empty/air
            return slice_img.GetOrigin()[:2]

    com_fixed = get_center_of_mass(fixed_slice)
    com_moving = get_center_of_mass(moving_slice)
    
    # 3. Calculate Shift Vector (Physical Distance)
    dx = com_fixed[0] - com_moving[0]
    dy = com_fixed[1] - com_moving[1]
    
    print(f"      Torso Center: ({com_fixed[0]:.1f}, {com_fixed[1]:.1f})")
    print(f"      Legs Center:  ({com_moving[0]:.1f}, {com_moving[1]:.1f})")
    print(f"      Correcting Shift: X={dx:.1f}mm, Y={dy:.1f}mm")
    
    # 4. Resample Moving Image to match Fixed Image's Grid (in X/Y)
    # We use the fixed image as reference for X/Y spacing/direction, but keep Z props
    
    # Create Transform
    transform = sitk.TranslationTransform(3)
    # Offset needs to be negative of the shift to move pixels 'into' the frame
    transform.SetOffset((-dx, -dy, 0))
    
    ref_origin = list(fixed_img.GetOrigin())
    ref_origin[2] = moving_img.GetOrigin()[2] # Keep Z depth from moving image
    
    ref_size = list(fixed_img.GetSize())
    ref_size[2] = moving_img.GetSize()[2] # Keep Z size from moving image
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img) # Borrow Spacing/Direction/Origin X/Y
    resampler.SetOutputOrigin(ref_origin)  # Restore Z Origin
    resampler.SetSize(ref_size)            # Restore Z Size
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(-1000) # Fill Air with -1000
    
    aligned_moving_img = resampler.Execute(moving_img)
    
    return aligned_moving_img

def calculate_volume_similarity_enhanced(vol_a, vol_b):
    """
    Enhanced similarity calculation with multi-scale analysis and rotation handling.
    Returns 0.0 (No match) to 1.0 (Perfect match).
    """
    from scipy.ndimage import zoom, rotate
    
    # Handle different shapes by resampling vol_b to match vol_a
    if vol_a.shape != vol_b.shape:
        zoom_factors = [vol_a.shape[i] / vol_b.shape[i] for i in range(len(vol_a.shape))]
        vol_b = zoom(vol_b, zoom_factors, order=1, mode='constant', cval=-1000)
    
    best_score = 0.0
    
    # Test multiple transformations (rotations and flips)
    transformations = [
        (0, False, False),    # No rotation, no flip
        (90, False, False),   # 90° rotation
        (180, False, False),  # 180° rotation
        (270, False, False),  # 270° rotation
        (0, True, False),     # Flip X
        (0, False, True),     # Flip Y
        (0, True, True),      # Flip both X and Y
    ]
    
    for angle, flip_x, flip_y in transformations:
        vol_b_transformed = vol_b.copy()
        
        # Apply flips first
        if flip_x:
            vol_b_transformed = vol_b_transformed[:, ::-1, :]
        if flip_y:
            vol_b_transformed = vol_b_transformed[:, :, ::-1]
            
        # Apply rotation
        if angle != 0:
            vol_b_transformed = rotate(vol_b_transformed, angle, axes=(1, 2), reshape=False, order=1, cval=-1000)
        
        vol_b_rot = vol_b_transformed
        
        # Multi-scale comparison for robustness
        scores = []
        
        # 1. Full resolution comparison
        score_full = _compute_ncc(vol_a, vol_b_rot)
        scores.append(score_full * 0.6)  # Weight: 60%
        
        # 2. Downsampled comparison (faster, captures large structures)
        vol_a_small = vol_a[::2, ::2, ::2]  # Downsample by 2
        vol_b_small = vol_b_rot[::2, ::2, ::2]
        score_small = _compute_ncc(vol_a_small, vol_b_small)
        scores.append(score_small * 0.3)  # Weight: 30%
        
        # 3. Edge-based comparison (structural similarity)
        score_edges = _compute_edge_similarity(vol_a, vol_b_rot)
        scores.append(score_edges * 0.1)  # Weight: 10%
        
        total_score = sum(scores)
        best_score = max(best_score, total_score)
        
        # Early exit if we find a very good match
        if total_score > 0.85:
            print(f"         Found excellent match with angle={angle}°, flip_x={flip_x}, flip_y={flip_y}, score={total_score:.3f}")
            break
        elif total_score > 0.7:
            print(f"         Good match: angle={angle}°, flip_x={flip_x}, flip_y={flip_y}, score={total_score:.3f}")
    
    return best_score

def _compute_ncc(vol_a, vol_b):
    """Fast Normalized Cross-Correlation computation."""
    # Flatten and convert to float32
    a = vol_a.flatten().astype(np.float32)
    b = vol_b.flatten().astype(np.float32)
    
    # Normalize (subtract mean)
    a -= np.mean(a)
    b -= np.mean(b)
    
    # Compute correlation
    numerator = np.sum(a * b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    
    if denominator == 0: return 0.0
    return max(0.0, numerator / denominator)  # Clamp negative values

def _compute_edge_similarity(vol_a, vol_b):
    """Fast edge-based structural similarity."""
    from scipy.ndimage import sobel
    
    # Compute gradients (edges) - sample every 4th slice for speed
    a_edges = sobel(vol_a[::4, ::2, ::2])
    b_edges = sobel(vol_b[::4, ::2, ::2])
    
    # Threshold to binary edges
    a_binary = (a_edges > np.percentile(a_edges, 90)).astype(np.float32)
    b_binary = (b_edges > np.percentile(b_edges, 90)).astype(np.float32)
    
    # Compute overlap
    intersection = np.sum(a_binary * b_binary)
    union = np.sum((a_binary + b_binary) > 0)
    
    if union == 0: return 0.0
    return intersection / union

# Keep the old function as fallback
def calculate_volume_similarity(vol_a, vol_b):
    """Fallback similarity calculation."""
    return calculate_volume_similarity_enhanced(vol_a, vol_b)

def perform_sliding_stitch(fixed_series, moving_series):
    print("\n3. Running Sliding Density Optimization...")
    
    # 1. Load Data into Memory
    reader = sitk.ImageSeriesReader()
    
    print("   -> Loading Torso (Fixed)...")
    reader.SetFileNames(fixed_series['paths'])
    fixed_img = reader.Execute()
    
    print("   -> Loading Legs (Moving)...")
    reader.SetFileNames(moving_series['paths'])
    moving_img = reader.Execute()
    
    # --- NEW STEP: X/Y ALIGNMENT ---
    moving_img = align_xy_centers(fixed_img, moving_img)
    # -------------------------------
    
    # (Z, Y, X) arrays
    fixed_arr = sitk.GetArrayFromImage(fixed_img) 
    moving_arr = sitk.GetArrayFromImage(moving_img)
    
    # 2. Define Search Parameters
    fixed_len = len(fixed_arr)
    moving_len = len(moving_arr)
    
    # Search Window: Stop when top of legs reaches middle of torso
    max_overlap = fixed_len // 2
    
    best_score = -1.0
    best_overlap = 0
    
    print(f"   -> Max Search Window: {max_overlap} slices overlap")
    print("-" * 80)
    print(f"{'OVERLAP':<10} | {'TORSO SLICES':<20} | {'LEG SLICES':<20} | {'SCORE':<10}")
    print("-" * 80)
    
    # 3. Adaptive Sliding Loop with early termination
    best_scores = []  # Track recent scores for trend analysis
    
    for k in range(1, max_overlap + 1):
        # SLICE SELECTION
        # Bottom k slices of Torso vs Top k slices of Legs
        block_torso = fixed_arr[-k:]
        block_legs  = moving_arr[:k]
        
        score = calculate_volume_similarity_enhanced(block_torso, block_legs)
        best_scores.append(score)
        
        # Early termination if score is degrading consistently
        if len(best_scores) > 10:
            recent_trend = np.mean(best_scores[-5:]) - np.mean(best_scores[-10:-5])
            if recent_trend < -0.1 and score < best_score * 0.7:
                print(f"      Early termination at {k} slices (degrading trend)")
                break
        
        if score > best_score:
            best_score = score
            best_overlap = k
            
        if k % 10 == 0 or score > 0.8 or k <= 20:  # More frequent output for first 20 iterations
            torso_range = f"[{fixed_len - k} : {fixed_len}]"
            legs_range  = f"[0 : {k}]"
            print(f"{k:<10} | {torso_range:<20} | {legs_range:<20} | {score:.5f}")
        elif k % 25 == 0:  # Progress indicator for longer searches
            print(f"   ... continuing search at {k}/{max_overlap} ({100*k/max_overlap:.1f}%), current best: {best_score:.3f}")
            
    print("-" * 80)
    print(f"   -> OPTIMAL MATCH FOUND!")
    print(f"      Best Overlap: {best_overlap} slices")
    print(f"      Max Similarity: {best_score:.5f}")
    
    # 4. Construct the Stitched Volume
    print("   -> Fusing Volumes...")
    part_a = fixed_arr
    part_b = moving_arr[best_overlap:]
    
    stitched_arr = np.concatenate((part_a, part_b), axis=0)
    print(f"      New Volume Shape: {stitched_arr.shape}")
    
    # Convert back to SimpleITK/VTK
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

# Smart sorting: Try to identify torso vs legs using multiple criteria
series_data.sort(key=lambda x: x['z_min'])
to_render = remove_subsets(series_data)

if len(to_render) >= 2:
    # Try to intelligently assign torso vs legs
    print("   -> Analyzing series for torso/legs identification...")
    
    candidates = to_render[:3]  # Consider top 3 candidates
    torso = None
    legs = None
    
    # Method 1: Use heuristics from metadata
    torso_candidates = [s for s in candidates if s.get('is_likely_torso', False)]
    legs_candidates = [s for s in candidates if s.get('is_likely_legs', False)]
    
    if torso_candidates and legs_candidates:
        torso = torso_candidates[0]
        legs = legs_candidates[0]
        print(f"      Using metadata heuristics: Torso Z={torso['z_min']:.1f}, Legs Z={legs['z_min']:.1f}")
    else:
        # Method 2: Use Z position (torso usually higher, legs lower)
        candidates.sort(key=lambda x: x['avg_z'], reverse=True)
        torso = candidates[0]  # Higher Z = torso
        legs = candidates[1] if len(candidates) > 1 else candidates[0]  # Lower Z = legs
        print(f"      Using Z-position heuristics: Torso Z={torso['z_min']:.1f}, Legs Z={legs['z_min']:.1f}")
    
    # If they're the same, use the first two
    if torso['uid'] == legs['uid']:
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

opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(-1000, 0.0)
opacity.AddPoint(100, 0.0)
opacity.AddPoint(250, 0.2)
opacity.AddPoint(1000, 0.9)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
color.AddRGBPoint(200, 0.9, 0.8, 0.7)
color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

print("4. Rendering Final Stitched Result...")

# Convert the single stitched volume to VTK
vtk_img_data, vtk_mat = sitk_to_vtk(final_sitk_img)

# No Rotation applied here, we view it in "Raw" space
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