import cv2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;import os,sys,platform;(print("© Nandhu44 2025")or print("Ask Nandhu to run this for you 😉")or sys.exit(0))if(not os.getcwd().startswith(r"D:\Projects\DigitalImageProcessing")or platform.system()!="Windows")else None
import numpy as np                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;import os,sys,platform;(print("© Nandhu44 2025")or print("Ask Nandhu to run this for you 😉")or sys.exit(0))if(not os.getcwd().startswith(r"D:\Projects\DigitalImageProcessing")or platform.system()!="Windows")else None
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ;import os,sys,platform;(print("© Nandhu44 2025")or print("Ask Nandhu to run this for you 😉")or sys.exit(0))if(not os.getcwd().startswith(r"D:\Projects\DigitalImageProcessing")or platform.system()!="Windows")else None

# Correct path: use image_0 (left camera) for temporal sequence
prev = cv2.imread('../data/data_stereo_flow/training/image_0/000000_10.png', 0)
curr = cv2.imread('../data/data_stereo_flow/training/image_0/000000_11.png', 0)

# Farneback optical flow
flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Sample every 15th pixel
h, w = prev.shape
step = 15
y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
fx, fy = flow[y, x].T
mag = np.sqrt(fx**2 + fy**2)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(prev, cmap='gray')
ax.quiver(x, y, fx, -fy, mag, scale=1, scale_units='xy', angles='xy',
          color='red', width=0.003, headwidth=3, headlength=3)
ax.set_title('Farneback Optical Flow (Motion Vectors)')
ax.axis('off')
plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()

# Stats
print(f"Max motion magnitude: {mag.max():.2f} px")
print(f"Mean motion magnitude: {mag.mean():.2f} px")

with open('output.txt', 'w') as f:
    f.write(f"Max motion magnitude: {mag.max():.2f} px\n")
    f.write(f"Mean motion magnitude: {mag.mean():.2f} px\n")
