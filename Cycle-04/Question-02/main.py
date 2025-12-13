import cv2
import numpy as np
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ;import os,sys,platform;(print("© Nandhu44 2025")or print("Ask Nandhu to run this for you 😉")or sys.exit(0))if(not os.getcwd().startswith(r"D:\Projects\DigitalImageProcessing")or platform.system()!="Windows")else None

# Load consecutive frames (image_0 = left camera = temporal sequence)
prev = cv2.imread('../data/data_stereo_flow/training/image_0/000000_10.png', 0)
curr = cv2.imread('../data/data_stereo_flow/training/image_0/000000_11.png', 0)

h, w = prev.shape
block_size = 16
half = block_size // 2
search_range = 32  # ±32 px search window

# Containers
vectors = []  # (x, y, dx, dy)
costs = []    # SAD values

# Grid: block centers
for y in range(half, h - half, block_size):
    for x in range(half, w - half, block_size):
        block = prev[y-half:y+half, x-half:x+half]
        min_sad = float('inf')
        best_dx = best_dy = 0

        # Search window
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y2 = y + dy
                x2 = x + dx
                if (y2-half >= 0 and y2+half < h and
                    x2-half >= 0 and x2+half < w):
                    candidate = curr[y2-half:y2+half, x2-half:x2+half]
                    sad = np.sum(np.abs(block.astype(int) - candidate.astype(int)))
                    if sad < min_sad:
                        min_sad = sad
                        best_dx, best_dy = dx, dy

        vectors.append((x, y, best_dx, best_dy))
        costs.append(min_sad)

# Unpack
x, y, dx, dy = np.array(vectors).T
mag = np.sqrt(dx**2 + dy**2)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(prev, cmap='gray')
ax.quiver(x, y, dx, -dy, mag, scale=1, scale_units='xy', angles='xy',
          color='lime', width=0.003, headwidth=3)
ax.set_title('Block Matching (SAD, 16×16) – Motion Vectors')
ax.axis('off')
plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()

# Stats
print(f"Max motion: {mag.max():.1f} px")
print(f"Mean motion: {mag.mean():.1f} px")
print(f"Blocks: {len(vectors)}")

with open('output.txt', 'w') as f:
    f.write(f"Max motion: {mag.max():.1f} px\n")
    f.write(f"Mean motion: {mag.mean():.1f} px\n")
    f.write(f"Blocks: {len(vectors)}\n")
