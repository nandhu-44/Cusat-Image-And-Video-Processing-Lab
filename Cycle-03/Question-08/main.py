import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic image
def create_image():
    img = np.zeros((256, 256), dtype=np.uint8)
    img[0:128, 0:128] = 50
    img[0:128, 128:256] = 150
    img[128:256, 0:128] = 200
    img[128:256, 128:256] = 100
    
    # Add detail
    cv2.circle(img, (64, 64), 30, 255, -1)
    cv2.rectangle(img, (150, 150), (200, 200), 0, -1)
    
    # Noise
    noise = np.random.normal(0, 5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

img = create_image()

# Quadtree Split
class QuadTree:
    def __init__(self, img, x, y, w, h, min_dim=4, std_thresh=10):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.children = []
        self.mean = np.mean(img[y:y+h, x:x+w])
        self.std = np.std(img[y:y+h, x:x+w])
        
        if w > min_dim and h > min_dim and self.std > std_thresh:
            # Split
            hw, hh = w//2, h//2
            self.children.append(QuadTree(img, x, y, hw, hh, min_dim, std_thresh))
            self.children.append(QuadTree(img, x+hw, y, w-hw, hh, min_dim, std_thresh))
            self.children.append(QuadTree(img, x, y+hh, hw, h-hh, min_dim, std_thresh))
            self.children.append(QuadTree(img, x+hw, y+hh, w-hw, h-hh, min_dim, std_thresh))

def draw_quadtree(node, img_out, seg_out):
    if not node.children:
        cv2.rectangle(img_out, (node.x, node.y), (node.x+node.w, node.y+node.h), (0, 255, 0), 1)
        seg_out[node.y:node.y+node.h, node.x:node.x+node.w] = node.mean
    else:
        for child in node.children:
            draw_quadtree(child, img_out, seg_out)

# Apply
root = QuadTree(img, 0, 0, 256, 256, min_dim=4, std_thresh=10)

img_blocks = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
seg_img = np.zeros_like(img)
draw_quadtree(root, img_blocks, seg_img)

results = [
    ('Original', img),
    ('Quadtree Blocks', img_blocks),
    ('Segmented (Mean)', seg_img)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    if title == 'Quadtree Blocks':
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()