import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Define kernels
# Horizontal
h_kernel = np.array([[-1, -1, -1],
                     [ 2,  2,  2],
                     [-1, -1, -1]], dtype=np.float32)

# Vertical
v_kernel = np.array([[-1,  2, -1],
                     [-1,  2, -1],
                     [-1,  2, -1]], dtype=np.float32)

# +45 Degree Diagonal
d45_kernel = np.array([[-1, -1,  2],
                       [-1,  2, -1],
                       [ 2, -1, -1]], dtype=np.float32)

# -45 Degree Diagonal
dm45_kernel = np.array([[ 2, -1, -1],
                        [-1,  2, -1],
                        [-1, -1,  2]], dtype=np.float32)

# Apply filters
res_h = cv2.filter2D(img, -1, h_kernel)
res_v = cv2.filter2D(img, -1, v_kernel)
res_d45 = cv2.filter2D(img, -1, d45_kernel)
res_dm45 = cv2.filter2D(img, -1, dm45_kernel)

# Combine all for a total edge map
res_all = res_h + res_v + res_d45 + res_dm45

results = [
    ('Original', img),
    ('Horizontal Lines', res_h),
    ('Vertical Lines', res_v),
    ('+45 Diagonal', res_d45),
    ('-45 Diagonal', res_dm45),
    ('All Lines Combined', res_all)
]

plt.figure(figsize=(12, 8))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
