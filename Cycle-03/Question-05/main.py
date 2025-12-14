import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/road.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Edge Detection (Canny)
edges = cv2.Canny(img, 50, 150)

# Hough Transform (Probabilistic)
# minLineLength: Minimum length of line. Line segments shorter than this are rejected.
# maxLineGap: Maximum allowed gap between line segments to treat them as single line.
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

# Draw lines
img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

results = [
    ('Original', img),
    ('Edges (Canny)', edges),
    ('Detected Lines', img_lines)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    if title == 'Detected Lines':
        plt.imshow(result)
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()