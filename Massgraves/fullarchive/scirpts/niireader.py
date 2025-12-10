import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import vtk
from vtk.util import numpy_support

# --- CONFIGURATION ---
# Point this to the folder containing your .nii files
DATASET = "DICOM-Maria"  # Options: DICOM-Maria, DICOM-Jan, DICOM-Jarek
DATA_FOLDER = rf"/Users/yoad/Desktop/CLS year 2/SVandVR/Massgraves/data/nii_exports/{DATASET}"

# To visualize ONLY the stitched result, set this to True:
SHOW_STITCHED_ONLY = True

VISUALIZE_3D = True  # Set to False to only show 2D slices


def render_3d_volume(sitk_img, title):
    """
    Render volume in 3D using VTK
    """
    print(f"  Opening 3D viewer for {title}...")
    
    # Convert SimpleITK to numpy
    arr = sitk.GetArrayFromImage(sitk_img)  # Z, Y, X
    
    # Create VTK image
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=arr.ravel(),
        deep=True,
        array_type=vtk.VTK_SHORT
    )
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_img.GetSize())
    vtk_image.SetSpacing(sitk_img.GetSpacing())
    vtk_image.SetOrigin(sitk_img.GetOrigin())
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    # Volume mapper
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(vtk_image)
    
    # Transfer functions
    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(-1024, 0.0)
    opacity.AddPoint(-500, 0.0)
    opacity.AddPoint(-200, 0.05)
    opacity.AddPoint(200, 0.3)
    opacity.AddPoint(500, 0.6)
    opacity.AddPoint(1500, 0.8)
    
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
    color.AddRGBPoint(-500, 0.5, 0.3, 0.2)
    color.AddRGBPoint(0, 0.9, 0.7, 0.6)
    color.AddRGBPoint(500, 1.0, 0.9, 0.9)
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
    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    
    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 1000)
    render_window.SetWindowName(title)
    render_window.AddRenderer(renderer)
    
    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    render_window.Render()
    interactor.Start()


def check_files():
    if not os.path.exists(DATA_FOLDER):
        print(f"ERROR: Directory not found: {DATA_FOLDER}")
        return
        
    files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.nii')])
    
    # Filter to show only stitched file if requested
    if SHOW_STITCHED_ONLY:
        files = [f for f in files if 'stitched' in f.lower()]
    
    if not files:
        print(f"ERROR: No .nii files found in {DATA_FOLDER}")
        return

    print(f"\n{'='*80}")
    print(f"NIfTI VIEWER - {DATASET if 'DATASET' in locals() else 'Results'}")
    print(f"{'='*80}")
    print(f"Found {len(files)} NIfTI files\n")

    for idx, fname in enumerate(files, 1):
        path = os.path.join(DATA_FOLDER, fname)
        print(f"\n[{idx}/{len(files)}] Processing: {fname}")
        print("-" * 80)
        
        try:
            # Load Image
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img) # Order: (Z, Y, X)
            
            # Get metadata
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            size = img.GetSize()
            
            # Statistics
            min_val = np.min(arr)
            max_val = np.max(arr)
            mean_val = np.mean(arr)
            
            print(f"  Dimensions: {size[0]}×{size[1]}×{size[2]} voxels")
            print(f"  Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm")
            print(f"  Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
            print(f"  HU Range: [{min_val:.1f}, {max_val:.1f}] (mean: {mean_val:.1f})")
            
            if max_val < -500:
                print("  ⚠️  WARNING: Image looks empty (All Air/Black)")
                continue

            # Extract 3 orthogonal slices for quick preview
            mid_z = arr.shape[0] // 2
            mid_y = arr.shape[1] // 2
            mid_x = arr.shape[2] // 2
            
            slice_axial = arr[mid_z, :, :]
            slice_coronal = arr[:, mid_y, :]
            slice_sagittal = arr[:, :, mid_x]
            
            # Plot 2D slices
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{fname}\nDimensions: {size[0]}×{size[1]}×{size[2]}", fontsize=12)
            
            # Axial
            axes[0].imshow(slice_axial, cmap='gray', vmin=-200, vmax=300)
            axes[0].set_title(f'Axial (Z={mid_z})')
            axes[0].axis('off')
            
            # Coronal
            axes[1].imshow(slice_coronal, cmap='gray', vmin=-200, vmax=300, aspect='auto')
            axes[1].set_title(f'Coronal (Y={mid_y})')
            axes[1].axis('off')
            
            # Sagittal
            axes[2].imshow(slice_sagittal, cmap='gray', vmin=-200, vmax=300, aspect='auto')
            axes[2].set_title(f'Sagittal (X={mid_x})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # 3D Visualization
            if VISUALIZE_3D:
                print(f"\n  {'='*76}")
                print(f"  3D VOLUME RENDERING - Close window to continue to next file")
                print(f"  {'='*76}")
                render_3d_volume(img, f"{fname}")
            
        except Exception as e:
            print(f"  ❌ Failed to read: {e}")
    
    print(f"\n{'='*80}")
    print(f"✓ Finished processing {len(files)} files")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    check_files()