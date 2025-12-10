#!/usr/bin/env python3
"""
NIfTI Volume Visualizer
========================
Visualizes all .nii files in a specified folder using VTK volume rendering.
Press 'q' to move to the next volume.
"""

import os
import sys
import glob
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


def visualize_nifti(filepath):
    """Visualize a single NIfTI file with VTK volume rendering"""
    filename = os.path.basename(filepath)
    print(f"\nVisualizing: {filename}")
    
    # Load NIfTI
    sitk_img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(sitk_img)  # Z, Y, X
    
    print(f"  Size: {sitk_img.GetSize()} ({arr.shape[0]} slices)")
    print(f"  Spacing: {sitk_img.GetSpacing()}")
    
    # Convert to VTK
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])  # X, Y, Z
    vtk_image.SetSpacing(sitk_img.GetSpacing())
    vtk_image.SetOrigin(sitk_img.GetOrigin())
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
    # Transfer functions - bone rendering
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-500, 0.0)
    opacity.AddPoint(-200, 0.0)
    opacity.AddPoint(200, 0.15)
    opacity.AddPoint(400, 0.4)
    opacity.AddPoint(800, 0.8)
    opacity.AddPoint(1500, 1.0)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-500, 0.3, 0.15, 0.1)
    color.AddRGBPoint(200, 0.6, 0.4, 0.3)
    color.AddRGBPoint(500, 0.9, 0.85, 0.8)
    color.AddRGBPoint(1000, 1.0, 1.0, 0.95)
    color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
    
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    
    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.15)
    renderer.ResetCamera()
    
    # Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1200, 900)
    window.SetWindowName(f"NIfTI Viewer: {filename}")
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    print(f"  Controls: Left-drag=Rotate, Right-drag=Zoom, Middle-drag=Pan, 'q'=Next")
    
    window.Render()
    interactor.Start()


def main():
    """Main function to visualize all NIfTI files in a folder"""
    if len(sys.argv) < 2:
        print("Usage: python niivis.py <folder_path>")
        print("Example: python niivis.py ../improved_stitched_bodies")
        sys.exit(1)
    
    folder = sys.argv[1]
    
    if not os.path.exists(folder):
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    # Find all .nii files
    nii_files = sorted(glob.glob(os.path.join(folder, "*.nii")))
    
    if not nii_files:
        print(f"No .nii files found in: {folder}")
        sys.exit(1)
    
    print("=" * 80)
    print("NIfTI VOLUME VISUALIZER")
    print("=" * 80)
    print(f"\nFolder: {folder}")
    print(f"Found {len(nii_files)} .nii file(s):\n")
    
    for i, filepath in enumerate(nii_files, 1):
        print(f"  {i}. {os.path.basename(filepath)}")
    
    print("\n" + "=" * 80)
    print("Press 'q' in the window to move to next volume")
    print("=" * 80)
    
    # Visualize each file
    for filepath in nii_files:
        try:
            visualize_nifti(filepath)
        except Exception as e:
            print(f"\n❌ Error visualizing {os.path.basename(filepath)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
