import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic image with blobs
def create_blobs():
    img = np.zeros((256, 256), dtype=np.uint8)
    # Draw some circles
    blobs = [
        (50, 50, 20, 200),
        (150, 50, 30, 180),
        (50, 150, 25, 220),
        (180, 180, 40, 150),
        (100, 100, 15, 250)
    ]
    for (x, y, r, i) in blobs:
        cv2.circle(img, (x, y), r, i, -1)
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

img = create_blobs()

# Thresholding (Otsu)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Connected Components (Labeling)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Colorize labels
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

results = [
    ('Original with Noise', img),
    ('Histogram', None), # Special handling
    ('Otsu Thresholding', thresh),
    ('Labeled Blobs', labeled_img)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    if title == 'Histogram':
        plt.hist(img.ravel(), 256, range=[0, 256])
        plt.axvline(ret, color='r', linestyle='dashed', linewidth=2, label=f'Otsu: {ret:.1f}')
        plt.legend()
        plt.title(title)
    elif title == 'Labeled Blobs':
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    else:
        plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()