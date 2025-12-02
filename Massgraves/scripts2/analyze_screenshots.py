#!/usr/bin/env python3
"""
Analyze the generated screenshots to understand body part positioning
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load screenshots
front = np.array(Image.open('Jarek_merged_front.png').convert('L'))
side = np.array(Image.open('Jarek_merged_side.png').convert('L'))
angle = np.array(Image.open('Jarek_merged_angle.png').convert('L'))

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(front, cmap='gray')
axes[0].set_title('Front View', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(side, cmap='gray')
axes[1].set_title('Side View', fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(angle, cmap='gray')
axes[2].set_title('Angled View', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('screenshot_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved analysis: screenshot_analysis.png")

# Analyze vertical distribution (for checking if body parts are connected)
# In front view, find vertical extent of bright pixels
threshold = 30  # pixels brighter than this
front_bright = front > threshold

# Find topmost and bottommost rows with bright pixels
rows_with_data = np.where(np.any(front_bright, axis=1))[0]
if len(rows_with_data) > 0:
    top_row = rows_with_data[0]
    bottom_row = rows_with_data[-1]
    height_span = bottom_row - top_row
    
    print(f"\nFront view vertical analysis:")
    print(f"  Top edge: row {top_row} ({top_row/front.shape[0]*100:.1f}% from top)")
    print(f"  Bottom edge: row {bottom_row} ({bottom_row/front.shape[0]*100:.1f}% from top)")
    print(f"  Vertical span: {height_span} pixels ({height_span/front.shape[0]*100:.1f}% of image)")
    
    # Check for gaps (rows with no bright pixels in the middle of the body)
    middle_section = front_bright[top_row:bottom_row+1, :]
    rows_without_data = np.where(~np.any(middle_section, axis=1))[0]
    
    if len(rows_without_data) > 0:
        # Find contiguous gaps
        gaps = []
        gap_start = rows_without_data[0]
        gap_size = 1
        
        for i in range(1, len(rows_without_data)):
            if rows_without_data[i] == rows_without_data[i-1] + 1:
                gap_size += 1
            else:
                if gap_size > 5:  # Only report significant gaps
                    gaps.append((gap_start, gap_size))
                gap_start = rows_without_data[i]
                gap_size = 1
        
        if gap_size > 5:
            gaps.append((gap_start, gap_size))
        
        if gaps:
            print(f"\n  ⚠ Found {len(gaps)} gap(s) in vertical continuity:")
            for gap_start, gap_size in gaps:
                actual_row = top_row + gap_start
                print(f"    Gap at row {actual_row}: {gap_size} pixels ({gap_size/height_span*100:.1f}% of body)")
        else:
            print(f"\n  ✓ No significant gaps in vertical continuity")
    else:
        print(f"\n  ✓ Perfect vertical continuity (no gaps)")

plt.show()
