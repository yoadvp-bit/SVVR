#!/usr/bin/env python3
"""
VISUALIZE STITCHED BODIES (ONE AT A TIME)
Shows each final stitched body in 3D, one window at a time.
Press 'q' to close and move to next body.
Reads from final_stitched_bodies/ folder.
"""

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import os
import sys


def sitk_to_vtk(sitk_img):
    """Convert SimpleITK image to VTK with transformation matrix"""
    img_np = sitk.GetArrayFromImage(sitk_img)
    vtk_data = numpy_support.numpy_to_vtk(num_array=img_np.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    dims = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(dims[0], dims[1], dims[2])
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
    
    return vtk_img, mat


def setup_transfer_functions():
    """Setup transfer functions for BONES ONLY (no skin/soft tissue)"""
    # Opacity - only show bone (HU > 150), more visible
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.00)  # Air - invisible
    opacity.AddPoint(100,   0.00)  # Soft tissue - invisible
    opacity.AddPoint(150,   0.05)  # Start faint bone
    opacity.AddPoint(250,   0.40)  # Visible bone
    opacity.AddPoint(400,   0.70)  # Cancellous bone
    opacity.AddPoint(700,   0.90)  # Cortical bone
    opacity.AddPoint(1500,  1.00)  # Dense bone
    
    # Color - bone coloring (white/cream)
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)   # Air - black
    color.AddRGBPoint(150,   0.0, 0.0, 0.0)   # Below bone - black
    color.AddRGBPoint(250,   0.85, 0.8, 0.7)  # Light bone color
    color.AddRGBPoint(400,   0.95, 0.9, 0.8)  # Bone
    color.AddRGBPoint(700,   1.0, 0.95, 0.9)  # Dense bone
    color.AddRGBPoint(1500,  1.0, 1.0, 1.0)   # Very dense - pure white
    
    return color, opacity


def visualize_body(name, filepath, body_num, total_bodies):
    """Visualize a single body in its own window"""
    print("=" * 80)
    print(f"VISUALIZING: {name} ({body_num}/{total_bodies})")
    print("=" * 80)
    print(f"Loading: {os.path.basename(filepath)}")
    
    img = sitk.ReadImage(filepath)
    print(f"   Size: {img.GetSize()}")
    print(f"   Spacing: {img.GetSpacing()}")
    print(f"   Origin: {img.GetOrigin()}")
    print()
    
    # Setup transfer functions
    color, opacity = setup_transfer_functions()
    
    # Create VTK volume
    vtk_img_data, vtk_mat = sitk_to_vtk(img)
    
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
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.05, 0.05)
    renderer.AddVolume(volume)
    
    # Position camera - increased distance to show full body including arms
    volume.Update()
    bounds = volume.GetBounds()
    if bounds:
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
        
        # Calculate a good camera distance based on volume size
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        max_dim = max(size_x, size_y, size_z)
        
        cam = renderer.GetActiveCamera()
        cam.SetFocalPoint(center)
        cam.SetPosition(center[0], center[1] - max_dim * 3, center[2])  # Dynamic distance
        cam.SetViewUp(0, 0, -1)
        cam.SetClippingRange(1, max_dim * 10)  # Wider clipping range
        renderer.ResetCamera()  # Reset to fit entire volume
        cam.Zoom(0.8)  # Zoom out a bit to ensure everything is visible
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)
    render_window.SetWindowName(f"{name} - Final Stitched Body ({body_num}/{total_bodies})")
    render_window.AddRenderer(renderer)
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print("Controls:")
    print("   - Left Click + Drag: Rotate")
    print("   - Right Click + Drag: Zoom")
    print("   - Middle Click + Drag: Pan")
    print("   - 'q' or Close Window: Next body")
    print()
    print(f"Rendering {name}... (Press 'q' to close and move to next)")
    print()
    
    render_window.Render()
    interactor.Start()
    
    # Clean up
    render_window.Finalize()
    interactor.TerminateApp()
    del render_window, interactor


def main():
    # Check for all 6 bodies
    body_order = ['Maria', 'Jan', 'Jarek', 'Gerda', 'Joop', 'Loes']
    
    # Read from merged-resampled-bodies folder (in finalMG)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    massgraves_root = os.path.dirname(script_dir)
    source_folder = os.path.join(massgraves_root, 'finalMG', 'merged-resampled-bodies')
    
    print("=" * 80)
    print("VISUALIZING MERGED RESAMPLED BODIES INDIVIDUALLY")
    print("=" * 80)
    print()
    print(f"Source: {source_folder}")
    print()
    
    # Find all available bodies
    available_bodies = []
    for name in body_order:
        body_folder = os.path.join(source_folder, name)
        if os.path.isdir(body_folder):
            # Look for a file with the patient name and no suffix (e.g., "Maria", "Maria.nii")
            nii_files = [f for f in os.listdir(body_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
            if nii_files:
                # Prefer file matching the patient name exactly
                matching_file = None
                for f in nii_files:
                    # Check if filename starts with patient name
                    if f.startswith(name):
                        matching_file = f
                        break
                # If no exact match, use the first .nii file
                if not matching_file:
                    matching_file = nii_files[0]
                
                filepath = os.path.join(body_folder, matching_file)
                available_bodies.append((name, filepath))
            else:
                print(f"⚠️  {name} folder found but no .nii files: {body_folder}")
        else:
            print(f"⚠️  {name} not found: {body_folder}")
    
    if not available_bodies:
        print()
        print(f"❌ No merged bodies found in {source_folder}/")
        print("   Run the merge script first!")
        return
    
    print(f"Found {len(available_bodies)} bodies:")
    for name, path in available_bodies:
        print(f"   ✓ {name}")
    print()
    
    # Visualize each body one by one
    total = len(available_bodies)
    for idx, (name, filepath) in enumerate(available_bodies, 1):
        visualize_body(name, filepath, idx, total)
    
    print("=" * 80)
    print("✓ ALL BODIES VISUALIZED")
    print("=" * 80)


if __name__ == '__main__':
    main()