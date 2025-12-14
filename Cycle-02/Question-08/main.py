import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# Define Laplacian Kernel
# This kernel detects edges.
# To sharpen, we subtract the edges from the original image (if center is negative)
# or add if center is positive.
# Here: Center is -4.
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)

# Apply Laplacian Filter
# We use float32 to keep negative values during calculation
laplacian = cv2.filter2D(img.astype(np.float32), -1, laplacian_kernel)

# Sharpening: Original - Laplacian
# (Image) - (Edges) -> Enhances the edges
# Since our kernel has negative center, subtracting it effectively adds the edges back?
# Let's look at the math:
# Sharpened = Original + c * (Original - Blurred)
# Laplacian ~ (Original - Blurred)
# Actually, standard sharpening formula with this kernel:
# Sharpened = Original - Laplacian
sharpened = img.astype(np.float32) - laplacian

# Clip values to 0-255
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Normalize Laplacian for visualization
laplacian_disp = cv2.convertScaleAbs(laplacian)

results = [
    ('Original', img),
    ('Laplacian (Edges)', laplacian_disp),
    ('Sharpened Image', sharpened)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
