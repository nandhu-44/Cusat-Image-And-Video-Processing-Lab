import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load Stereo Pair ===
left  = cv2.imread('../data/data_stereo_flow/training/image_0/000000_10.png', 0)
right = cv2.imread('../data/data_stereo_flow/training/image_1/000000_10.png', 0)

# === StereoSGBM ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5,
    P1=8*5*5**2, P2=32*5*5**2, mode=cv2.STEREO_SGBM_MODE_SGBM
)
disparity = stereo.compute(left, right).astype(np.float32) / 16.0

# === Calibration ===
calib_file = '../data/data_stereo_flow/training/calib/000000.txt'
with open(calib_file) as f:
    lines = f.readlines()
P0 = np.array(lines[0].strip().split()[1:], dtype=float).reshape(3,4)
P1 = np.array(lines[1].strip().split()[1:], dtype=float).reshape(3,4)
fx = P0[0,0]
baseline = -P1[0,3] / fx
depth = (fx * baseline) / (disparity + 1e-6)
depth[disparity <= 0] = 0

# === Obstacle Detection ===
depth_threshold = 10.0
obstacle_mask = (depth > 0) & (depth < depth_threshold)

# === Overlay on Original (Red) ===
result = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
result[obstacle_mask] = [0, 0, 255]  # BGR Red

# === Optional: Dim non-obstacles (FIXED) ===
overlay = cv2.applyColorMap(left, cv2.COLORMAP_JET)
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# FIXED: Use integer math to stay in uint8
dim_factor = 77  # 0.3 * 255 ≈ 77
overlay_dimmed = overlay.copy()
overlay_dimmed[~obstacle_mask] = (overlay[~obstacle_mask].astype(np.uint16) * dim_factor // 255).astype(np.uint8)

# === Save & Show ===
cv2.imwrite('obstacle_overlay.png', result[:, :, ::-1])  # BGR → RGB
cv2.imwrite('obstacle_overlay_dimmed.png', overlay_dimmed)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax[0].set_title(f'Obstacle Detection (Depth < {depth_threshold}m)')
ax[0].axis('off')

ax[1].imshow(depth, cmap='magma')
ax[1].set_title('Depth Map')
ax[1].axis('off')

plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Obstacle pixels: {obstacle_mask.sum()} ({100*obstacle_mask.mean():.1f}%)")

print(f"Baseline: {baseline:.3f} m")
print(f"Max disparity: {disparity.max():.1f}")
print(f"Min depth (valid): {depth[depth>0].min():.1f} m")

with open('output.txt', 'w') as f:
    f.write(f"Obstacle pixels: {obstacle_mask.sum()} ({100*obstacle_mask.mean():.1f}%)\n")
    f.write(f"Baseline: {baseline:.3f} m\n")
    f.write(f"Max disparity: {disparity.max():.1f}\n")
    f.write(f"Min depth (valid): {depth[depth>0].min():.1f} m\n")