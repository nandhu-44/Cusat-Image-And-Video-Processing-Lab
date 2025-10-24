import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two consecutive grayscale frames
prev = cv2.imread('./data/data_stereo_flow/training/image_1/000000_10.png', 0)    # t
curr = cv2.imread('./data/data_stereo_flow/training/image_1/000000_11.png', 0)   # t+1

# Compute dense optical flow
flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Sample flow at every 15th pixel (for clean arrow plot)
h, w = prev.shape
step = 15
y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
fx, fy = flow[y, x].T

# Magnitude & angle
mag = np.sqrt(fx**2 + fy**2)
angle = np.arctan2(fy, fx)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(prev, cmap='gray')
ax.quiver(x, y, fx, -fy, mag, scale=1, scale_units='xy', angles='xy', color='red', width=0.003)
ax.set_title('Farneback Optical Flow (Motion Vectors)')
ax.axis('off')
plt.tight_layout()
plt.show()

# Stats
print(f"Max motion magnitude: {mag.max():.2f} px")
print(f"Mean motion magnitude: {mag.mean():.2f} px")