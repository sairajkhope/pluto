import os

import numpy as np
from PIL import Image

this_file = os.path.abspath(__file__)
this_dir = os.path.dirname(this_file)

# Create heightfield data for flat box with resolution of 16 pixels per meter for a 0.6x0.6m area
h = np.ones((10, 10), dtype=np.uint8) * 255  # White = maximum height

# Save as PNG image for MuJoCo heightfield
img = Image.fromarray(h, mode="L")  # 'L' mode for grayscale
img.save(os.path.join(this_dir, "toddlerbot_2xc", "assets", "hfield_box.png"))

print(f"Generated hfield_box.png with shape: {h.shape}")
print(
    f"Saved to: {os.path.join(this_dir, 'toddlerbot_2xc', 'assets', 'hfield_box.png')}"
)
