#!/usr/bin/env python3
"""
INTERACTIVE AXIAL ORDERING AND LABELING TOOL

Allows manual ordering and selection of axial CT series for body reconstruction.

OVERVIEW:
  - Displays all axial NIfTI volumes for a selected patient in a web browser
  - User assigns order numbers to each axial series:
    * Order 1 = highest/superior body part (e.g., head)
    * Higher numbers = lower/inferior body parts (e.g., legs)
  - Multiple axials can share the same order number (indicates left/right body splits)
  - Axials without an order number will be excluded from further processing

USAGE:
  1. Set the PATIENT variable in the script (e.g., 'Maria', 'Jarek')
  2. Run the script with the play button in VS Code
  3. Web browser opens automatically with the interactive interface
  4. For each axial:
     - Enter an order number (1, 2, 3, etc.) or leave blank to exclude
     - Same numbers indicate left/right splits at that body level
  5. Click "Save & Rename Files" button
  6. Files are renamed: [ORDER_NR]_[original_filename].nii
  
OUTPUT:
  - Files with order numbers: Will be used in downstream stitching/reconstruction
  - Files without order numbers: Excluded from processing (remain unchanged)
  
EXAMPLE RENAMING:
  axial_001.nii → 1_axial_001.nii (head region)
  axial_002.nii → 2_axial_002.nii (torso)
  axial_003.nii → 2_axial_003.nii (torso, left/right split with axial_002)
  axial_004.nii → 3_axial_004.nii (legs)
  axial_005.nii → (no number assigned, excluded)
"""

import SimpleITK as sitk
import numpy as np
import os
import base64
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
import threading
import vtk
from vtk.util import numpy_support
from io import BytesIO


class AxialOrderingTool:
    """Web-based interface for ordering and labeling axial CT series"""
    
    def __init__(self, patient_folder):
        self.patient_folder = Path(patient_folder)
        self.patient_name = self.patient_folder.name.replace('NIfTI-', '')
        
        # Load all axial files
        self.axial_files = sorted([f for f in self.patient_folder.glob('*.nii') 
                                   if not f.name.startswith('.')])
        
        if not self.axial_files:
            raise ValueError(f"No .nii files found in {patient_folder}")
        
        # Load preview slices (middle slice of each volume)
        print(f"\nLoading {len(self.axial_files)} axial volumes for {self.patient_name}...")
        print("Generating 3D spinning volume previews (horizontal + vertical)...")
        self.preview_gifs = []
        self.volume_info = []
        
        for idx, filepath in enumerate(self.axial_files):
            print(f"  [{idx+1}/{len(self.axial_files)}] Processing {filepath.name}...")
            
            # Load with SimpleITK to preserve spacing/origin
            sitk_img = sitk.ReadImage(str(filepath))
            
            # Generate both horizontal and vertical spinning previews
            gif_horizontal = self.generate_spinning_volume(sitk_img, rotation_axis='horizontal')
            gif_vertical = self.generate_spinning_volume(sitk_img, rotation_axis='vertical')
            
            self.preview_gifs.append({
                'horizontal': gif_horizontal,
                'vertical': gif_vertical
            })
            
            array = sitk.GetArrayFromImage(sitk_img)
            spacing = sitk_img.GetSpacing()  # (X, Y, Z) spacing in mm
            self.volume_info.append({
                'filename': filepath.name,
                'path': filepath,
                'shape': array.shape,
                'size_mb': filepath.stat().st_size / (1024 * 1024),
                'z_spacing': spacing[2]  # Z-spacing (slice thickness) in mm
            })
        
        print("✓ All previews generated\n")
        self.server = None
        self.server_thread = None
    
    def generate_spinning_volume(self, sitk_img, num_frames=48, rotation_axis='horizontal'):
        """Generate a spinning 3D volume rendering - EXACT copy of niireader.py approach"""
        from PIL import Image
        
        # Convert SimpleITK to numpy - exactly like niireader.py
        arr = sitk.GetArrayFromImage(sitk_img)  # Z, Y, X
        
        # Downsample if needed for speed
        if arr.shape[0] > 100:
            step = max(1, arr.shape[0] // 100)
            arr = arr[::step, ::step, ::step]
            # Update spacing
            original_spacing = sitk_img.GetSpacing()
            new_spacing = tuple(s * step for s in original_spacing)
        else:
            new_spacing = sitk_img.GetSpacing()
        
        # Create VTK image - EXACT same as niireader.py
        vtk_data = numpy_support.numpy_to_vtk(
            num_array=arr.ravel(),
            deep=True,
            array_type=vtk.VTK_SHORT
        )
        
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(arr.shape[2], arr.shape[1], arr.shape[0])  # X, Y, Z from Z, Y, X
        vtk_image.SetSpacing(new_spacing)
        vtk_image.SetOrigin(sitk_img.GetOrigin())
        vtk_image.GetPointData().SetScalars(vtk_data)
        
        # Volume mapper - EXACT same as niireader.py
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_image)
        
        # Transfer functions - different for horizontal (bones) vs vertical (skin)
        opacity = vtk.vtkPiecewiseFunction()
        color = vtk.vtkColorTransferFunction()
        
        if rotation_axis == 'horizontal':
            # Horizontal: Show bones prominently
            opacity.AddPoint(-1024, 0.0)
            opacity.AddPoint(-500, 0.0)
            opacity.AddPoint(-200, 0.0)
            opacity.AddPoint(200, 0.15)
            opacity.AddPoint(400, 0.4)
            opacity.AddPoint(800, 0.8)
            opacity.AddPoint(1500, 1.0)
            
            color.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
            color.AddRGBPoint(-500, 0.3, 0.15, 0.1)
            color.AddRGBPoint(200, 0.6, 0.4, 0.3)
            color.AddRGBPoint(500, 0.9, 0.85, 0.8)
            color.AddRGBPoint(1000, 1.0, 1.0, 0.95)
            color.AddRGBPoint(1500, 1.0, 1.0, 1.0)
        else:
            # Vertical: Also show bones (same as horizontal)
            opacity.AddPoint(-1024, 0.0)
            opacity.AddPoint(-500, 0.0)
            opacity.AddPoint(-200, 0.0)
            opacity.AddPoint(200, 0.15)
            opacity.AddPoint(400, 0.4)
            opacity.AddPoint(800, 0.8)
            opacity.AddPoint(1500, 1.0)
            
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
        renderer.SetBackground(0.1, 0.1, 0.1)
        renderer.ResetCamera()
        
        # Camera setup - zoom out more
        camera = renderer.GetActiveCamera()
        
        # For vertical rotation, rotate the volume itself instead of camera
        if rotation_axis == 'vertical':
            # Position camera to look from the side
            position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
            # Move camera to the side (rotate 90 degrees around Y-axis)
            camera.SetPosition(position[2], position[1], -position[0])
            camera.SetViewUp(0, 0, 1)  # Set Z as up direction
            renderer.ResetCamera()
        
        camera.Zoom(1.1)  # Less zoom = more zoomed out
        
        # Render window with off-screen rendering
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(350, 350)
        
        # Capture frames
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGB()
        
        frames = []
        for i in range(num_frames):
            if rotation_axis == 'horizontal':
                camera.Azimuth(360.0 / num_frames)
            else:  # vertical - rotate volume around Z-axis (length of body)
                volume.RotateZ(360.0 / num_frames)
            
            render_window.Render()
            
            window_to_image.Modified()
            window_to_image.Update()
            
            # Get image
            vtk_image_out = window_to_image.GetOutput()
            width, height, _ = vtk_image_out.GetDimensions()
            vtk_array = vtk_image_out.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            
            arr_img = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)
            arr_img = np.flipud(arr_img)
            
            pil_img = Image.fromarray(arr_img, 'RGB')
            frames.append(pil_img)
        
        # Create GIF with slower animation (100ms per frame for 48 frames)
        buffer = BytesIO()
        frames[0].save(
            buffer, 
            format='GIF', 
            save_all=True, 
            append_images=frames[1:], 
            duration=100,  # 100ms per frame
            loop=0
        )
        
        gif_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/gif;base64,{gif_base64}"
    
    def generate_rotating_preview(self, volume_array, num_frames=24):
        """Generate a rotating 3D preview as animated GIF"""
        from PIL import Image
        
        # Downsample volume for faster rendering
        if volume_array.shape[0] > 150:
            step = volume_array.shape[0] // 150
            volume_array = volume_array[::step, ::step, ::step]
        
        # Convert to VTK image data
        vtk_data_array = numpy_support.numpy_to_vtk(
            volume_array.ravel(order='F'),
            deep=True,
            array_type=vtk.VTK_SHORT
        )
        
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(volume_array.shape[2], volume_array.shape[1], volume_array.shape[0])
        vtk_image.GetPointData().SetScalars(vtk_data_array)
        
        # Use GPU ray cast mapper for better quality
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)
        
        # Volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Opacity transfer function - more aggressive to show body structure
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(-1024, 0.0)
        opacity_func.AddPoint(-500, 0.0)
        opacity_func.AddPoint(-100, 0.05)
        opacity_func.AddPoint(100, 0.2)
        opacity_func.AddPoint(300, 0.4)
        opacity_func.AddPoint(500, 0.6)
        opacity_func.AddPoint(1500, 0.9)
        
        # Color transfer function
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(-500, 0.2, 0.1, 0.1)
        color_func.AddRGBPoint(-100, 0.5, 0.2, 0.2)
        color_func.AddRGBPoint(100, 0.8, 0.6, 0.5)
        color_func.AddRGBPoint(300, 0.9, 0.8, 0.7)
        color_func.AddRGBPoint(1000, 1.0, 0.95, 0.9)
        color_func.AddRGBPoint(2000, 1.0, 1.0, 1.0)
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Gradient opacity for edge enhancement
        gradient_func = vtk.vtkPiecewiseFunction()
        gradient_func.AddPoint(0, 0.0)
        gradient_func.AddPoint(90, 0.5)
        gradient_func.AddPoint(100, 1.0)
        volume_property.SetGradientOpacity(gradient_func)
        
        # Volume actor
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        # Renderer setup
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.15, 0.15, 0.2)
        renderer.AddVolume(volume)
        
        # Camera setup
        camera = renderer.GetActiveCamera()
        renderer.ResetCamera()
        camera.Zoom(1.3)
        camera.Elevation(15)
        
        # Render window with off-screen rendering
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(350, 350)
        
        # Capture frames
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGB()
        
        frames = []
        for i in range(num_frames):
            camera.Azimuth(360.0 / num_frames)
            render_window.Render()
            
            window_to_image.Modified()
            window_to_image.Update()
            
            # Get image data
            vtk_image_out = window_to_image.GetOutput()
            width, height, _ = vtk_image_out.GetDimensions()
            vtk_array = vtk_image_out.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            
            arr = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, components)
            arr = np.flipud(arr)  # Flip vertically
            
            pil_img = Image.fromarray(arr, 'RGB')
            frames.append(pil_img)
        
        # Create animated GIF
        buffer = BytesIO()
        frames[0].save(
            buffer, 
            format='GIF', 
            save_all=True, 
            append_images=frames[1:], 
            duration=80,  # ms per frame
            loop=0
        )
        
        gif_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/gif;base64,{gif_base64}"
    
    def generate_html(self):
        """Generate HTML interface"""
        
        # Build table rows with spinning 3D previews (horizontal and vertical)
        rows_html = ""
        for idx, (info, gifs) in enumerate(zip(self.volume_info, self.preview_gifs)):
            rows_html += f"""
            <tr>
                <td style="text-align: center; vertical-align: middle;">
                    <div style="display: flex; gap: 10px; justify-content: center; align-items: center;">
                        <div>
                            <img src="{gifs['horizontal']}" style="max-width: 280px; height: auto; border: 2px solid #34495e; border-radius: 5px;">
                            <br><small style="color: #7f8c8d;">Horizontal rotation</small>
                        </div>
                        <div>
                            <img src="{gifs['vertical']}" style="max-width: 280px; height: auto; border: 2px solid #34495e; border-radius: 5px;">
                            <br><small style="color: #7f8c8d;">Vertical rotation</small>
                        </div>
                    </div>
                </td>
                <td style="vertical-align: middle;">
                    <strong>{info['filename']}</strong><br>
                    <small>Shape: {info['shape']} | Size: {info['size_mb']:.1f} MB</small><br>
                    <small>Z-spacing: {info['z_spacing']:.2f} mm</small>
                </td>
                <td style="vertical-align: middle;">
                    <input type="number" id="order_{idx}" name="order_{idx}" 
                           style="width: 80px; padding: 5px; font-size: 14px;">
                </td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Axial Ordering Tool - {self.patient_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .subtitle {{
            color: #ecf0f1;
            font-size: 14px;
            margin-top: 10px;
        }}
        table {{
            width: 100%;
            background-color: white;
            border-collapse: collapse;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .buttons {{
            margin-top: 20px;
            text-align: right;
        }}
        button {{
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 10px;
        }}
        .save-btn {{
            background-color: #27ae60;
            color: white;
        }}
        .save-btn:hover {{
            background-color: #229954;
        }}
        .cancel-btn {{
            background-color: #e74c3c;
            color: white;
        }}
        .cancel-btn:hover {{
            background-color: #c0392b;
        }}
        .note {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Patient: {self.patient_name}</h1>
        <div class="subtitle">
            Assign order numbers: 1=Superior (head), Higher=Inferior (legs)<br>
            Same number = Left/Right split | Leave blank to exclude<br>
            <em>Previews show 3D spinning volume renderings (same as niireader.py)</em>
        </div>
    </div>
    
    <div class="note">
        <strong>Instructions:</strong> Enter order numbers for each axial series. 
        Files with numbers will be renamed and used in processing. 
        Files without numbers will be excluded.
    </div>
    
    <form id="orderForm">
        <table>
            <tr>
                <th>Preview</th>
                <th>File Information</th>
                <th>Order Number</th>
            </tr>
            {rows_html}
        </table>
        
        <div class="buttons">
            <button type="submit" class="save-btn">Save & Copy Ordered Files</button>
        </div>
    </form>
    
    <script>
        document.getElementById('orderForm').onsubmit = function(e) {{
            e.preventDefault();
            const formData = new FormData(this);
            const data = {{}};
            for (let [key, value] of formData.entries()) {{
                data[key] = value;
            }}
            
            fetch('/save', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }})
            .then(response => response.json())
            .then(result => {{
                alert(result.message);
                window.close();
            }});
        }};
    </script>
</body>
</html>
        """
        return html
    
    def save_and_rename(self, order_data):
        """Save order assignments and copy ordered files to nifti-ordered folder"""
        import shutil
        
        # Collect assignments
        assignments = []
        for idx in range(len(self.volume_info)):
            order_str = order_data.get(f'order_{idx}', '').strip()
            if order_str:
                try:
                    order_num = int(order_str)
                    assignments.append((idx, order_num))
                except ValueError:
                    print(f"⚠ Warning: Invalid order number '{order_str}' for {self.volume_info[idx]['filename']}")
        
        if not assignments:
            return "⚠ No order numbers assigned. No files will be copied."
        
        # Create nifti-ordered folder in same parent as nifti-raw
        nifti_raw_parent = self.patient_folder.parent.parent
        nifti_ordered = nifti_raw_parent / 'nifti-ordered' / self.patient_folder.name.replace('NIfTI-', '')
        nifti_ordered.mkdir(parents=True, exist_ok=True)
        
        # Copy and rename files
        print(f"\n{'=' * 70}")
        print("COPYING ORDERED FILES")
        print('=' * 70)
        print(f"Destination: {nifti_ordered}\n")
        
        copied_count = 0
        for idx, order_num in assignments:
            old_path = self.volume_info[idx]['path']
            old_name = old_path.name
            
            # Check if already has order prefix, remove it
            if old_name[0].isdigit() and '_' in old_name:
                old_name = '_'.join(old_name.split('_')[1:])
            
            new_name = f"{order_num}_{old_name}"
            new_path = nifti_ordered / new_name
            
            try:
                shutil.copy2(old_path, new_path)
                print(f"  ✓ {old_path.name:30s} → {new_name}")
                copied_count += 1
            except Exception as e:
                print(f"  ✗ Failed to copy {old_path.name}: {e}")
        
        # Report excluded files
        excluded = [self.volume_info[i]['filename'] for i in range(len(self.volume_info))
                   if i not in [a[0] for a in assignments]]
        
        if excluded:
            print(f"\n{'=' * 70}")
            print("EXCLUDED FILES (no order number assigned):")
            print('=' * 70)
            for filename in excluded:
                print(f"  ⊘ {filename}")
        
        print(f"\n{'=' * 70}")
        print(f"✓ COMPLETE: Copied {copied_count} ordered files to nifti-ordered/{self.patient_folder.name.replace('NIfTI-', '')}")
        print(f"  Excluded: {len(excluded)} files")
        print('=' * 70 + '\n')
        
        return f"✓ Copied {copied_count} files to nifti-ordered. Excluded {len(excluded)} files."
    
    def run(self):
        """Start web server and open browser"""
        
        tool = self
        
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress server logs
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(tool.generate_html().encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path == '/save':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    order_data = json.loads(post_data.decode())
                    
                    message = tool.save_and_rename(order_data)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'message': message}).encode())
                    
                    # Shutdown server after save
                    threading.Thread(target=self.server.shutdown).start()
        
        # Start server
        self.server = HTTPServer(('localhost', 8765), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Open browser
        url = 'http://localhost:8765'
        print(f"\n{'=' * 70}")
        print(f"Opening browser at: {url}")
        print(f"{'=' * 70}\n")
        webbrowser.open(url)
        
        # Wait for server to finish
        self.server_thread.join()


def find_nifti_data_folder():
    """Search for nifti-raw folder in parent directories"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for _ in range(4):
        nifti_path = os.path.join(current_dir, 'nifti-raw')
        if os.path.isdir(nifti_path):
            return nifti_path
        current_dir = os.path.dirname(current_dir)
    
    return None


if __name__ == '__main__':
    # ==================== CONFIGURATION ====================
    PATIENT = 'Jarek'  # Change this to: 'Maria', 'Jarek', 'Jan', 'Gerda', 'Loes', 'Joop'
    # =======================================================
    
    # Find nifti-raw folder
    nifti_raw = find_nifti_data_folder()
    
    if nifti_raw is None:
        print("✗ ERROR: Could not find 'nifti-raw' folder")
        exit(1)
    
    patient_folder = os.path.join(nifti_raw, f'NIfTI-{PATIENT}')
    
    if not os.path.isdir(patient_folder):
        print(f"✗ ERROR: Patient folder not found: {patient_folder}")
        print(f"\nAvailable patients:")
        for item in os.listdir(nifti_raw):
            if item.startswith('NIfTI-'):
                print(f"  - {item.replace('NIfTI-', '')}")
        exit(1)
    
    # Run the tool
    tool = AxialOrderingTool(patient_folder)
    tool.run()
