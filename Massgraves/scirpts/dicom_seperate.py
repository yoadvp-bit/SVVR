import vtk
import os
import sys
import pydicom
import numpy as np
from vtk.util import numpy_support

# --- CONFIGURATION ---
target_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/DICOM-Maria"
def get_series_map(start_path):
    print(f"Scanning {start_path} with pydicom...")
    series_map = {}
    files_checked = 0
    
    for root, dirs, files in os.walk(start_path):
        for f in files:
            if f.startswith('.'): continue
            filepath = os.path.join(root, f)
            files_checked += 1
            if files_checked % 500 == 0: print(f"  Scanned {files_checked} files...")

            try:
                # Read header only
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                uid = dcm.SeriesInstanceUID
                
                if 'ImagePositionPatient' in dcm:
                    z_pos = dcm.ImagePositionPatient[2]
                    if uid not in series_map: series_map[uid] = []
                    series_map[uid].append((z_pos, filepath))
            except:
                continue 
    return series_map

def load_volume_from_pydicom(file_list):
    """
    Manually reads DICOM pixels using pydicom and converts to VTK.
    This works even if the files are compressed/weird.
    """
    # 1. Sort files by Z
    file_list.sort(key=lambda x: x[0])
    sorted_paths = [x[1] for x in file_list]
    
    # 2. Load the first file to get dimensions/spacing
    ref = pydicom.dcmread(sorted_paths[0])
    rows = int(ref.Rows)
    cols = int(ref.Columns)
    depth = len(sorted_paths)
    
    pixel_spacing = ref.PixelSpacing # [x, y]
    # Estimate Z-spacing (slice thickness) from the first two slices
    if len(sorted_paths) > 1:
        z1 = file_list[0][0]
        z2 = file_list[1][0]
        z_spacing = abs(z2 - z1)
    else:
        z_spacing = ref.SliceThickness if 'SliceThickness' in ref else 1.0

    print(f"   -> Loading {depth} slices ({rows}x{cols})...")
    
    # 3. Create empty Numpy 3D Array
    # We use Int16 because CT values (Hounsfield) can be negative
    volume_array = np.zeros((depth, rows, cols), dtype=np.int16)
    
    # 4. Fill the array
    for i, path in enumerate(sorted_paths):
        dcm = pydicom.dcmread(path)
        # Intercept RescaleSlope/Intercept to get real Hounsfield Units
        slope = getattr(dcm, 'RescaleSlope', 1)
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        
        # Get raw pixels
        slice_data = dcm.pixel_array.astype(np.float64)
        
        # Apply HU correction:  HU = pixel * slope + intercept
        slice_data = (slice_data * slope) + intercept
        
        volume_array[i, :, :] = slice_data.astype(np.int16)

    # 5. Convert Numpy -> VTK
    print("   -> Converting to VTK...")
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=volume_array.ravel(), 
        deep=True, 
        array_type=vtk.VTK_SHORT
    )
    
    img = vtk.vtkImageData()
    img.SetDimensions(cols, rows, depth)
    img.SetSpacing(pixel_spacing[0], pixel_spacing[1], z_spacing)
    img.GetPointData().SetScalars(vtk_data)
    
    return img

# --- MAIN EXECUTION ---

# 1. FIND DATA
actual_path = None
for root, dirs, files in os.walk(target_folder):
    valid = [f for f in files if not f.startswith('.')]
    if len(valid) > 10:
        actual_path = root
        break

if not actual_path:
    print("Error: Data folder not found.")
    sys.exit(1)

series_data = get_series_map(actual_path)
print(f"\n--- FOUND {len(series_data)} SERIES ---")

# 2. SHOW ONE BY ONE
count = 0
for uid, file_list in series_data.items():
    count += 1
    num_slices = len(file_list)
    
    print(f"\nLOADING SERIES {count}/{len(series_data)} (UID: ...{uid[-6:]})")
    
    if num_slices < 10:
        print("-> Skipping (Too small)")
        continue

    # --- THE MAGIC BRIDGE ---
    try:
        vtk_image = load_volume_from_pydicom(file_list)
    except Exception as e:
        print(f"CRITICAL LOAD ERROR: {e}")
        print("Tip: If the error mentions 'GDCM' or 'Pillow', you need to install them.")
        continue

    # Setup Volume Rendering
    mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    mapper.SetInputData(vtk_image) # Use SetInputData for raw objects
    
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1000, 0.0)
    opacity.AddPoint(-200, 0.1)  # Skin
    opacity.AddPoint(200, 1.0)   # Bone
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-200, 0.6, 0.4, 0.3)
    color.AddRGBPoint(1000, 1.0, 0.9, 0.8)
    
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)
    render_window.SetWindowName(f"Series {count} - Close to Next")
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print(f"-> Displaying Series {count}... (Close window to check next)")
    render_window.Render()
    interactor.Start()