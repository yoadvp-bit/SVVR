#!/usr/bin/env python3
"""
NIFTI TO OBJ CONVERTER
Converts skeleton NIfTI files to OBJ meshes for Unity import.
- Reads from nifti-final folder
- Outputs to obj-final folder
- Preserves full quality and correct physical dimensions
- Uses bone threshold for skeleton extraction
- Generates MTL material file with anatomically correct bone coloring
"""

import os
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINALMG_DIR = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(FINALMG_DIR, "nifti-final")
OUTPUT_DIR = os.path.join(FINALMG_DIR, "obj-final")

# Bone threshold in Hounsfield Units
# Bones typically start at HU > 200-300
BONE_THRESHOLD = 200

# Step size for marching cubes (1 = full quality, higher = faster but lower quality)
# Keep at 1 for best quality since meshes are static
STEP_SIZE = 1

# Shared material file name
MATERIAL_FILENAME = "skeleton_material.mtl"
MATERIAL_NAME = "bone"


# ============================================================================
# MATERIAL DEFINITION - SCIENTIFICALLY ACCURATE BONE COLORING
# ============================================================================
"""
Real bone color science:
- Fresh/wet cortical bone: Pale ivory/cream (RGB ~235, 220, 200)
- Dry/cleaned bone: Lighter, more yellowish (RGB ~245, 235, 215)  
- Archaeological bone: Varies from cream to brown depending on soil
- CT visualization standard: Often uses white/cream for dense bone

For mass grave forensic context, we use cleaned skeletal remains appearance:
- Base color: Warm ivory/cream matching real skeletal specimens
- Slight subsurface scattering effect via ambient
- Moderate specularity for slight bone sheen (real bone has some reflectance)

Reference: Forensic anthropology imaging standards
"""

# Bone material properties (RGB values 0-1)
BONE_MATERIAL = {
    # Diffuse color - main bone color (warm ivory/cream)
    # Based on cleaned skeletal remains: RGB(235, 223, 205) 
    'Kd': (0.922, 0.875, 0.804),
    
    # Ambient color - shadow areas (slightly darker, warmer)
    # Simulates subsurface scattering in bone
    'Ka': (0.15, 0.14, 0.12),
    
    # Specular color - highlights (slight yellowish tint)
    # Real bone has low but visible specularity
    'Ks': (0.25, 0.24, 0.22),
    
    # Specular exponent (shininess) - 10-30 for matte materials
    # Bone is relatively matte, not shiny like metal
    'Ns': 15.0,
    
    # Transparency (1.0 = fully opaque)
    'd': 1.0,
    
    # Illumination model (2 = diffuse + specular)
    'illum': 2,
}


# ============================================================================
# FUNCTIONS
# ============================================================================

def create_material_file(output_dir):
    """
    Create a shared MTL material file for all skeleton meshes.
    Uses anatomically accurate bone coloring.
    """
    mtl_path = os.path.join(output_dir, MATERIAL_FILENAME)
    
    with open(mtl_path, 'w') as f:
        f.write("# Skeleton Material File\n")
        f.write("# Anatomically accurate bone coloring for forensic visualization\n")
        f.write("# Based on cleaned skeletal remains appearance\n")
        f.write("#\n")
        f.write("# Color reference: Warm ivory/cream matching real bone specimens\n")
        f.write("# RGB approximately (235, 223, 205) - pale cream/ivory\n")
        f.write("#\n\n")
        
        f.write(f"newmtl {MATERIAL_NAME}\n")
        f.write(f"# Diffuse color (main bone color - warm ivory)\n")
        f.write(f"Kd {BONE_MATERIAL['Kd'][0]:.6f} {BONE_MATERIAL['Kd'][1]:.6f} {BONE_MATERIAL['Kd'][2]:.6f}\n")
        f.write(f"# Ambient color (shadow areas - simulates subsurface scattering)\n")
        f.write(f"Ka {BONE_MATERIAL['Ka'][0]:.6f} {BONE_MATERIAL['Ka'][1]:.6f} {BONE_MATERIAL['Ka'][2]:.6f}\n")
        f.write(f"# Specular color (highlights - bone has slight sheen)\n")
        f.write(f"Ks {BONE_MATERIAL['Ks'][0]:.6f} {BONE_MATERIAL['Ks'][1]:.6f} {BONE_MATERIAL['Ks'][2]:.6f}\n")
        f.write(f"# Specular exponent (shininess - bone is matte)\n")
        f.write(f"Ns {BONE_MATERIAL['Ns']:.1f}\n")
        f.write(f"# Opacity (fully opaque)\n")
        f.write(f"d {BONE_MATERIAL['d']:.1f}\n")
        f.write(f"# Illumination model (diffuse + specular)\n")
        f.write(f"illum {BONE_MATERIAL['illum']}\n")
    
    print(f"\n  Created material file: {MATERIAL_FILENAME}")
    print(f"    Bone color: RGB({int(BONE_MATERIAL['Kd'][0]*255)}, {int(BONE_MATERIAL['Kd'][1]*255)}, {int(BONE_MATERIAL['Kd'][2]*255)})")
    
    return mtl_path


def save_obj(filename, verts, faces, normals=None, material_file=None):
    """
    Writes the mesh data to a standard .obj file.
    Includes vertex normals for better rendering quality.
    References the material file for coloring.
    """
    with open(filename, 'w') as f:
        f.write("# OBJ file generated from NIfTI skeleton data\n")
        f.write(f"# Vertices: {len(verts)}, Faces: {len(faces)}\n")
        
        # Reference material file
        if material_file:
            f.write(f"\nmtllib {material_file}\n")
        
        f.write("\n")
        
        # Write vertices
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write vertex normals if provided (improves rendering quality)
        if normals is not None:
            f.write("\n")
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        f.write("\n")
        
        # Use material
        if material_file:
            f.write(f"usemtl {MATERIAL_NAME}\n")
        
        # Write faces (OBJ is 1-indexed)
        if normals is not None:
            # Include normal indices (same as vertex indices for per-vertex normals)
            for i, face in enumerate(faces):
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        else:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def convert_nifti_to_obj(input_path, output_path, material_file):
    """
    Convert a NIfTI skeleton file to OBJ mesh.
    Preserves physical dimensions using the affine transform.
    """
    print(f"\n  Loading: {os.path.basename(input_path)}")
    
    # Load NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    # Get voxel dimensions (spacing)
    zooms = header.get_zooms()
    print(f"    Shape: {data.shape}")
    print(f"    Voxel spacing: {zooms[0]:.3f} x {zooms[1]:.3f} x {zooms[2]:.3f} mm")
    
    # Extract bone surface using marching cubes
    print(f"    Extracting surface (threshold={BONE_THRESHOLD} HU)...")
    
    try:
        # Marching cubes with spacing to get correct physical dimensions
        verts, faces, normals, values = marching_cubes(
            data, 
            level=BONE_THRESHOLD,
            spacing=zooms,  # Apply voxel spacing for correct size
            step_size=STEP_SIZE,
            allow_degenerate=False  # Cleaner mesh
        )
        
        print(f"    Vertices: {len(verts):,}")
        print(f"    Faces: {len(faces):,}")
        
        # Calculate mesh dimensions
        min_coords = verts.min(axis=0)
        max_coords = verts.max(axis=0)
        dimensions = max_coords - min_coords
        print(f"    Mesh dimensions: {dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f} mm")
        
        # Save OBJ file with normals and material reference
        save_obj(output_path, verts, faces, normals, material_file)
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    Saved: {os.path.basename(output_path)} ({file_size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    """Convert all NIfTI files in nifti-final to OBJ meshes."""
    print("=" * 60)
    print("NIFTI TO OBJ CONVERTER")
    print("=" * 60)
    print(f"\nInput:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Bone threshold: {BONE_THRESHOLD} HU")
    print(f"Quality: Maximum (step_size={STEP_SIZE})")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create shared material file
    create_material_file(OUTPUT_DIR)
    
    # Find all NIfTI files
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        return
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.nii', '.nii.gz'))]
    
    if not files:
        print(f"\n❌ No NIfTI files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(files)} NIfTI files")
    
    # Convert each file
    success_count = 0
    for filename in sorted(files):
        input_path = os.path.join(INPUT_DIR, filename)
        
        # Create output filename
        output_name = filename.replace('.nii.gz', '').replace('.nii', '') + ".obj"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        if convert_nifti_to_obj(input_path, output_path, MATERIAL_FILENAME):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success_count}/{len(files)} files converted")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  - {MATERIAL_FILENAME} (shared material)")
    for f in sorted(files):
        obj_name = f.replace('.nii.gz', '').replace('.nii', '') + ".obj"
        print(f"  - {obj_name}")
    print(f"{'='*60}")
    
    # Print Unity instructions
    print("\n" + "=" * 60)
    print("UNITY IMPORT INSTRUCTIONS")
    print("=" * 60)
    print("""
1. IMPORT FILES:
   - Copy the entire 'obj-final' folder into your Unity project's Assets folder
   - Unity will automatically detect .obj and .mtl files

2. VERIFY MATERIAL IMPORT:
   - Select any skeleton mesh in the Project window
   - In the Inspector, check 'Materials' tab
   - You should see 'bone' material listed
   - If not, click 'Extract Materials' button

3. MATERIAL SETUP (if needed):
   - The MTL file creates a material called 'bone'
   - Unity converts this to a Standard shader material
   - Color: RGB(235, 223, 205) - warm ivory/cream
   
4. FOR BEST RESULTS IN UNITY:
   - Select the imported 'bone' material
   - Shader: Standard
   - Rendering Mode: Opaque
   - Smoothness: 0.1 - 0.2 (bone is matte)
   - Enable 'Emission' slightly for better visibility in dark scenes
   
5. LIGHTING RECOMMENDATIONS:
   - Use soft directional light (simulates museum/lab lighting)
   - Light color: Slightly warm white (255, 250, 240)
   - Add subtle ambient occlusion for depth

6. IF MATERIAL DOESN'T APPEAR:
   - Create new Material in Unity
   - Set Albedo color to: #EBDFCD (RGB 235, 223, 205)
   - Set Smoothness to: 0.15
   - Apply to all skeleton meshes
""")


if __name__ == "__main__":
    main()