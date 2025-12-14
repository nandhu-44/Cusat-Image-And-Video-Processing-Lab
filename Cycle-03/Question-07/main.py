import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic image with blobs
def create_blobs():
    img = np.zeros((256, 256), dtype=np.uint8)
    blobs = [
        (50, 50, 20, 200),
        (150, 50, 30, 180),
        (50, 150, 25, 220),
        (180, 180, 40, 150),
        (100, 100, 15, 250)
    ]
    for (x, y, r, i) in blobs:
        cv2.circle(img, (x, y), r, i, -1)
    
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, blobs

img, blobs = create_blobs()

# Region Growing
def region_growing(img, seeds, threshold=20):
    rows, cols = img.shape
    segmentation = np.zeros_like(img)
    visited = np.zeros_like(img)
    
    for i, seed in enumerate(seeds):
        seed_x, seed_y = seed
        seed_val = float(img[seed_y, seed_x])
        
        stack = [(seed_x, seed_y)]
        visited[seed_y, seed_x] = 1
        
        while stack:
            x, y = stack.pop()
            segmentation[y, x] = 255 # Mark as part of region
            
            # Check 4-neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx]:
                    if abs(float(img[ny, nx]) - seed_val) < threshold:
                        visited[ny, nx] = 1
                        stack.append((nx, ny))
    return segmentation

# Use blob centers as seeds
seeds = [(b[0], b[1]) for b in blobs]

# Apply Region Growing
seg_img = region_growing(img, seeds, threshold=30)

# Visualize Seeds
img_seeds = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for s in seeds:
    cv2.circle(img_seeds, s, 3, (0, 0, 255), -1)

results = [
    ('Original with Noise', img),
    ('Seeds', img_seeds),
    ('Region Growing Result', seg_img)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    if title == 'Seeds':
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()