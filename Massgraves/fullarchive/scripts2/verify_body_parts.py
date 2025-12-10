#!/usr/bin/env python3
"""
Verify body part presence by examining specific anatomical regions
"""
import numpy as np
from PIL import Image

# Load screenshots
front = np.array(Image.open('Jarek_merged_front.png').convert('L'))
side = np.array(Image.open('Jarek_merged_side.png').convert('L'))

print("="*80)
print("BODY PART VERIFICATION - SPECIFIC ANATOMICAL CHECKS")
print("="*80)

threshold = 30
front_bright = front > threshold
side_bright = side > threshold

# Get actual body extent
front_rows = np.where(np.any(front_bright, axis=1))[0]
side_rows = np.where(np.any(side_bright, axis=1))[0]

front_top = front_rows[0]
front_bottom = front_rows[-1]
front_height = front_bottom - front_top

side_top = side_rows[0]
side_bottom = side_rows[-1]
side_height = side_bottom - side_top

print(f"\nFRONT VIEW:")
print(f"  Body spans: rows {front_top} to {front_bottom} ({front_height} pixels)")
print(f"  This is {front_height/front.shape[0]*100:.1f}% of the image height")

print(f"\nSIDE VIEW:")
print(f"  Body spans: rows {side_top} to {side_bottom} ({side_height} pixels)")
print(f"  This is {side_height/side.shape[0]*100:.1f}% of the image height")

# Detailed region check - divide side view into anatomical sections
# Side view should show full body from legs (bottom) to head (top) most clearly
print(f"\n" + "="*80)
print("ANATOMICAL REGION VERIFICATION (using side view for best clarity)")
print("="*80)

# Divide into 5 equal regions from top to bottom
# Top = HEAD, Bottom = LEGS (in display coordinates)
region_names = [
    "HEAD (superior)",
    "THORAX (chest)", 
    "ABDOMEN (belly)",
    "PELVIS (hips)",
    "LEGS (inferior)"
]

region_height = side_height // 5

print(f"\nChecking each anatomical region:")
print(f"Total body height in side view: {side_height} pixels")
print(f"Each region: ~{region_height} pixels\n")

all_present = True
for i, name in enumerate(region_names):
    start = side_top + i * region_height
    end = side_top + (i+1) * region_height if i < 4 else side_bottom
    
    region_data = side_bright[start:end, :]
    pixel_count = np.sum(region_data)
    region_area = (end - start) * side.shape[1]
    density = pixel_count / region_area * 100
    
    # Check presence (>3% is sufficient for bone/tissue)
    if density > 3:
        status = "✓ PRESENT"
    else:
        status = "✗ MISSING"
        all_present = False
    
    print(f"{i+1}. {name:20} | rows {start:4}-{end:4} | density: {density:5.1f}% | {status}")

# Check for anatomical correctness: legs should be narrower than torso
print(f"\n" + "="*80)
print("ANATOMICAL PROPORTION CHECK")
print("="*80)

# Measure width (horizontal extent) at different vertical positions
def measure_width_at_position(img_bright, row):
    """Measure horizontal extent at a given row"""
    cols_with_data = np.where(img_bright[row, :])[0]
    if len(cols_with_data) > 0:
        return cols_with_data[-1] - cols_with_data[0]
    return 0

# Sample width at center of each region
head_row = side_top + region_height // 2
thorax_row = side_top + region_height + region_height // 2
abdomen_row = side_top + 2 * region_height + region_height // 2
pelvis_row = side_top + 3 * region_height + region_height // 2
legs_row = side_top + 4 * region_height + region_height // 2

head_width = measure_width_at_position(side_bright, head_row)
thorax_width = measure_width_at_position(side_bright, thorax_row)
abdomen_width = measure_width_at_position(side_bright, abdomen_row)
pelvis_width = measure_width_at_position(side_bright, pelvis_row)
legs_width = measure_width_at_position(side_bright, legs_row)

print(f"\nAnterior-posterior body width at each region:")
print(f"  HEAD:    {head_width:3d} pixels")
print(f"  THORAX:  {thorax_width:3d} pixels")
print(f"  ABDOMEN: {abdomen_width:3d} pixels")
print(f"  PELVIS:  {pelvis_width:3d} pixels")
print(f"  LEGS:    {legs_width:3d} pixels")

# Anatomical checks
checks_passed = []
checks_failed = []

if thorax_width > legs_width:
    checks_passed.append("✓ Thorax wider than legs (correct)")
else:
    checks_failed.append("✗ Thorax should be wider than legs")

if abdomen_width > legs_width:
    checks_passed.append("✓ Abdomen wider than legs (correct)")
else:
    checks_failed.append("✗ Abdomen should be wider than legs")

if head_width > 50:
    checks_passed.append("✓ Head has reasonable size")
else:
    checks_failed.append("✗ Head appears too small")

# Check continuity - no large gaps
print(f"\n" + "="*80)
print("CONTINUITY VERIFICATION")
print("="*80)

# Check center column for continuous data
center_col = side.shape[1] // 2
center_strip = side_bright[:, center_col-30:center_col+30]
center_profile = np.any(center_strip, axis=1)

# Find gaps in the body region
gaps_in_body = []
in_gap = False
gap_start = 0

for row in range(side_top, side_bottom):
    if not center_profile[row]:
        if not in_gap:
            gap_start = row
            in_gap = True
    else:
        if in_gap:
            gap_size = row - gap_start
            if gap_size > 20:  # Significant gap (>20 pixels)
                gaps_in_body.append((gap_start, gap_size, (gap_start - side_top) / side_height))
            in_gap = False

if not gaps_in_body:
    checks_passed.append("✓ Body is continuous (no gaps between parts)")
    print("✓ No significant gaps detected - body parts are properly connected")
else:
    print(f"⚠ Found {len(gaps_in_body)} significant gap(s):")
    for gap_start, gap_size, rel_pos in gaps_in_body:
        checks_failed.append(f"✗ Gap at {rel_pos*100:.0f}% height: {gap_size} pixels")
        print(f"  Gap at row {gap_start} ({rel_pos*100:.1f}% from top): {gap_size} pixels")

print(f"\n" + "="*80)
print("FINAL VERIFICATION SUMMARY")
print("="*80)

print(f"\n✓ CHECKS PASSED ({len(checks_passed)}):")
for check in checks_passed:
    print(f"  {check}")

if checks_failed:
    print(f"\n✗ CHECKS FAILED ({len(checks_failed)}):")
    for check in checks_failed:
        print(f"  {check}")

print(f"\n" + "="*80)
if all_present and len(checks_failed) == 0:
    print("✓✓✓ VERIFICATION COMPLETE: All body parts present and correctly positioned ✓✓✓")
elif all_present:
    print("⚠ VERIFICATION: All parts present but some proportions unusual")
else:
    print("✗ VERIFICATION FAILED: Some body parts missing or incorrectly positioned")
print("="*80)
