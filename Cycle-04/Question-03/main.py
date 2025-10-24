import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load frames
prev = cv2.imread('../data/data_stereo_flow/training/image_0/000000_10.png', 0).astype(np.float32)
curr = cv2.imread('../data/data_stereo_flow/training/image_0/000000_11.png', 0).astype(np.float32)

h, w = prev.shape
block_size = 16
half = block_size // 2
search_range = 32

# === Block Matching (same as Q2) ===
motion_vectors = []
for y in range(half, h - half, block_size):
    for x in range(half, w - half, block_size):
        block = prev[y-half:y+half, x-half:x+half]
        min_sad = float('inf')
        best_dx = best_dy = 0
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                y2 = y + dy
                x2 = x + dx
                if (y2-half >= 0 and y2+half < h and x2-half >= 0 and x2+half < w):
                    cand = curr[y2-half:y2+half, x2-half:x2+half]
                    sad = np.sum(np.abs(block - cand))
                    if sad < min_sad:
                        min_sad = sad
                        best_dx, best_dy = dx, dy
        motion_vectors.append((x, y, best_dx, best_dy))

# === Predict next frame ===
predicted = np.zeros_like(prev)
for (x, y, dx, dy) in motion_vectors:
    src_y = y + dy
    src_x = x + dx
    if (src_y-half >= 0 and src_y+half < h and src_x-half >= 0 and src_x+half < w):
        block = curr[src_y-half:src_y+half, src_x-half:src_x+half]
        predicted[y-half:y+half, x-half:x+half] = block

# === Residual (absolute difference) ===
residual = np.abs(curr - predicted)
residual_vis = np.clip(residual, 0, 255).astype(np.uint8)

# === Save & Show ===
cv2.imwrite('predicted_frame.png', predicted.astype(np.uint8))
cv2.imwrite('residual_image.png', residual_vis)

# Plot: 3-panel
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(curr.astype(np.uint8), cmap='gray')
axes[0].set_title('Actual Frame (t+1)')
axes[0].axis('off')

axes[1].imshow(predicted.astype(np.uint8), cmap='gray')
axes[1].set_title('Predicted Frame')
axes[1].axis('off')

axes[2].imshow(residual_vis, cmap='hot')
axes[2].set_title('Residual (Abs Diff)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()

# Stats
mae = np.mean(residual)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Max Residual: {residual.max():.1f}")

with open('output.txt', 'w') as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    f.write(f"Max Residual: {residual.max():.1f}\n")