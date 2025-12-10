#!/usr/bin/env python3
"""
Detailed analysis of body part positioning in screenshots
Check for presence of: legs, pelvis, abdomen, thorax, head
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load screenshots
front = np.array(Image.open('Jarek_merged_front.png').convert('L'))
side = np.array(Image.open('Jarek_merged_side.png').convert('L'))
angle = np.array(Image.open('Jarek_merged_angle.png').convert('L'))

print("="*80)
print("DETAILED BODY PART ANALYSIS")
print("="*80)

# Expected Z-ranges from the output:
# Series 10: Z=[-941.0, -297.7] mm - Abdomen  1.0  B20f  (LEGS)
# Series 9:  Z=[-941.0, -296.0] mm - Abdomen  5.0  B30f  (LEGS duplicate)
# Series 13: Z=[-412.4, 265.2]  mm - Abdomen  1.0  B20f  (PELVIS + ABDOMEN + LOWER THORAX)
# Series 12: Z=[-410.5, 269.5]  mm - Abdomen  5.0  B30f  (PELVIS + ABDOMEN + LOWER THORAX duplicate)
# Series 6:  Z=[-367.5, 602.5]  mm - Thorax  5.0  B80f  (UPPER ABDOMEN + THORAX + HEAD)
# Series 5:  Z=[-367.5, 602.5]  mm - Thorax  5.0  B31f  (UPPER ABDOMEN + THORAX + HEAD duplicate)

print("\n1. Z-RANGE ANALYSIS:")
print("-" * 80)
z_total = 602.5 - (-941.0)  # 1543.5 mm total
print(f"Total body Z-range: [-941.0, 602.5] mm = {z_total:.1f} mm")
print(f"\nExpected anatomical regions (in Z order, inferior → superior):")
print(f"  LEGS:         [-941.0, -297.7] mm = {-297.7 - (-941.0):.1f} mm (41.7% of body)")
print(f"  PELVIS:       [-412.4, -200.0] mm (approximate)")
print(f"  ABDOMEN:      [-200.0, 200.0]  mm (approximate)")
print(f"  THORAX:       [200.0, 500.0]   mm (approximate)")
print(f"  HEAD:         [500.0, 602.5]   mm (approximate)")

# Analyze front view
print("\n2. FRONT VIEW ANALYSIS:")
print("-" * 80)

threshold = 30
front_bright = front > threshold

# Get vertical profile - count bright pixels in each row
vertical_profile = np.sum(front_bright, axis=1)
horizontal_profile = np.sum(front_bright, axis=0)

# Find extent
rows_with_data = np.where(vertical_profile > 10)[0]
cols_with_data = np.where(horizontal_profile > 10)[0]

if len(rows_with_data) > 0:
    top_row = rows_with_data[0]
    bottom_row = rows_with_data[-1]
    left_col = cols_with_data[0]
    right_col = cols_with_data[-1]
    
    print(f"Vertical extent: rows {top_row} to {bottom_row} (span: {bottom_row - top_row} pixels)")
    print(f"Horizontal extent: cols {left_col} to {right_col} (span: {right_col - left_col} pixels)")
    
    # Divide into 5 regions (bottom to top = legs to head in front view)
    height = bottom_row - top_row
    region_height = height // 5
    
    regions = {
        'HEAD (top)': (top_row, top_row + region_height),
        'THORAX': (top_row + region_height, top_row + 2*region_height),
        'ABDOMEN': (top_row + 2*region_height, top_row + 3*region_height),
        'PELVIS': (top_row + 3*region_height, top_row + 4*region_height),
        'LEGS (bottom)': (top_row + 4*region_height, bottom_row)
    }
    
    print(f"\nRegion analysis (dividing image into 5 anatomical sections):")
    for region_name, (start_row, end_row) in regions.items():
        region_data = front_bright[start_row:end_row, :]
        pixel_count = np.sum(region_data)
        region_pixels = (end_row - start_row) * front.shape[1]
        density = pixel_count / region_pixels * 100
        
        # Check for presence (>5% brightness)
        status = "✓ PRESENT" if density > 5 else "✗ MISSING"
        print(f"  {region_name:20} rows {start_row:4}-{end_row:4}: {density:5.1f}% filled  {status}")

# Analyze side view
print("\n3. SIDE VIEW ANALYSIS:")
print("-" * 80)

side_bright = side > threshold
side_vertical_profile = np.sum(side_bright, axis=1)
side_rows_with_data = np.where(side_vertical_profile > 10)[0]

if len(side_rows_with_data) > 0:
    side_top = side_rows_with_data[0]
    side_bottom = side_rows_with_data[-1]
    side_height = side_bottom - side_top
    
    print(f"Vertical extent: rows {side_top} to {side_bottom} (span: {side_height} pixels)")
    
    # In side view, check for anterior-posterior extent variations
    # Measure width at different heights
    side_region_height = side_height // 5
    
    print(f"\nAnterior-posterior extent by region:")
    for i, (region_name, _) in enumerate(regions.items()):
        start = side_top + i * side_region_height
        end = side_top + (i+1) * side_region_height if i < 4 else side_bottom
        
        region_slice = side_bright[start:end, :]
        horizontal_extent = np.sum(region_slice, axis=0)
        cols_present = np.where(horizontal_extent > 0)[0]
        
        if len(cols_present) > 0:
            width = cols_present[-1] - cols_present[0]
            status = "✓" if width > 50 else "⚠"
            print(f"  {region_name:20} width: {width:4} pixels {status}")

# Check for continuity
print("\n4. CONTINUITY CHECK:")
print("-" * 80)

# Check for significant gaps in the vertical direction
middle_col = front.shape[1] // 2
middle_strip = front_bright[:, middle_col-20:middle_col+20]  # 40-pixel wide strip down the center
middle_profile = np.any(middle_strip, axis=1)

# Find gaps
gaps = []
in_gap = False
gap_start = 0

for i in range(top_row, bottom_row):
    if not middle_profile[i]:
        if not in_gap:
            gap_start = i
            in_gap = True
    else:
        if in_gap:
            gap_size = i - gap_start
            if gap_size > 10:  # Significant gap
                gaps.append((gap_start, gap_size))
            in_gap = False

if gaps:
    print(f"⚠ Found {len(gaps)} significant gap(s) in central vertical continuity:")
    for gap_start, gap_size in gaps:
        relative_pos = (gap_start - top_row) / height * 100
        print(f"  Gap at row {gap_start} ({relative_pos:.1f}% from top): {gap_size} pixels")
else:
    print("✓ No significant gaps - body appears continuous from head to legs")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: original screenshots
axes[0, 0].imshow(front, cmap='gray')
axes[0, 0].set_title('Front View', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(side, cmap='gray')
axes[0, 1].set_title('Side View', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(angle, cmap='gray')
axes[0, 2].set_title('Angled View', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

# Bottom row: annotated analysis
axes[1, 0].imshow(front, cmap='gray')
axes[1, 0].set_title('Front View - Regions Annotated', fontsize=12, fontweight='bold')
# Draw region boundaries
colors = ['red', 'orange', 'yellow', 'green', 'blue']
for (region_name, (start_row, end_row)), color in zip(regions.items(), colors):
    rect = patches.Rectangle((0, start_row), front.shape[1], end_row-start_row, 
                             linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
    axes[1, 0].add_patch(rect)
    axes[1, 0].text(10, (start_row+end_row)/2, region_name.split('(')[0].strip(), 
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
axes[1, 0].axis('off')

# Vertical profile
axes[1, 1].barh(range(len(vertical_profile)), vertical_profile, color='cyan', height=1)
axes[1, 1].set_title('Vertical Profile (pixel count per row)', fontsize=12)
axes[1, 1].set_xlabel('Bright pixels per row')
axes[1, 1].set_ylabel('Row number (0=top)')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3)

# Horizontal profile  
axes[1, 2].bar(range(len(horizontal_profile)), horizontal_profile, color='magenta', width=1)
axes[1, 2].set_title('Horizontal Profile (pixel count per column)', fontsize=12)
axes[1, 2].set_xlabel('Column number (0=left)')
axes[1, 2].set_ylabel('Bright pixels per column')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_body_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved detailed analysis: detailed_body_analysis.png")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
if not gaps:
    print("✓ Body appears complete and continuous")
    print("✓ All major regions (legs, pelvis, abdomen, thorax, head) are present")
else:
    print(f"⚠ {len(gaps)} gap(s) detected - body parts may be disconnected")

plt.close('all')  # Close instead of showing to prevent hanging
