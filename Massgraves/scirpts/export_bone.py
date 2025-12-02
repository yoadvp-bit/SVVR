import vtk
import os
import sys

# --- CONFIGURATION ---
root_folder = r"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data" 
output_filename = "skeleton_mesh.obj" # <--- CHANGED EXTENSION TO .OBJ
bone_threshold = 200 

# --- HELPER FUNCTION ---
def find_dicom_directory(start_path):
    print(f"Searching in: {start_path}")
    for root, dirs, files in os.walk(start_path):
        valid_files = [f for f in files if not f.startswith('.')]
        if len(valid_files) > 10:
            return root
    return None

# 1. LOCATE & READ
actual_path = find_dicom_directory(root_folder)
if not actual_path:
    print("Error: No data found.")
    sys.exit(1)

reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(actual_path)
reader.Update()

dims = reader.GetOutput().GetDimensions()
if dims[2] <= 0:
    print("Error: 0 slices found.")
    sys.exit(1)

# 2. EXTRACT SURFACE
print("Extracting surface...")
surface_extractor = vtk.vtkContourFilter()
surface_extractor.SetInputConnection(reader.GetOutputPort())
surface_extractor.SetValue(0, bone_threshold) 

# 3. OPTIMIZE
print("Optimizing...")
decimator = vtk.vtkDecimatePro()
decimator.SetInputConnection(surface_extractor.GetOutputPort())
decimator.SetTargetReduction(0.5)
decimator.PreserveTopologyOn()

smoother = vtk.vtkWindowedSincPolyDataFilter()
smoother.SetInputConnection(decimator.GetOutputPort())
smoother.SetNumberOfIterations(15)
smoother.SetPassBand(0.1)

# 4. SAVE AS OBJ (This is the fix)
print(f"Saving to {output_filename}...")
writer = vtk.vtkOBJWriter() # <--- THE MAGIC CHANGE
writer.SetInputConnection(smoother.GetOutputPort())
writer.SetFileName(output_filename)
writer.Write()

print("DONE! Drag 'skeleton_mesh.obj' into Unity.")