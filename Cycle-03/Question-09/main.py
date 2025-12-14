import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/coins.png')

# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Threshold (Otsu inverse)
_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Morphological opening (remove noise)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Sure background area
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

# 5. Finding sure foreground area
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# 6. Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# 7. Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 8. Watershed
markers = cv2.watershed(img, markers)

# Create result image with contours
img_out = img.copy()
contours = []
for label in np.unique(markers):
    if label <= 1: continue # Skip background and unknown
    mask = np.where(markers == label, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        contours.append(cnts[0])

cv2.drawContours(img_out, contours, -1, (0, 0, 255), 2)

# Visualization
results = [
    ('Original', img, None),
    ('Binary (Otsu)', bin_img, 'gray'),
    ('Distance Transform', dist, 'gray'),
    ('Sure Foreground', sure_fg, 'gray'),
    ('Sure Background', sure_bg, 'gray'),
    ('Unknown Region', unknown, 'gray'),
    ('Markers', markers, 'tab20b'),
    ('Final Result', img_out, None)
]

plt.figure(figsize=(9, 9))
for i, (title, image, cmap) in enumerate(results):
    plt.subplot(3, 3, i+1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap=cmap if cmap else 'gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.savefig('output.png')
plt.show()