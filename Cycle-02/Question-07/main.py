import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# 1. Average Filter (Box Filter) - Linear
# Kernel is a matrix of ones normalized by size
box_filtered = cv2.blur(img, (5, 5))

# 2. Gaussian Filter - Linear
# Kernel values follow Gaussian distribution
gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)

results = [
    ('Original RGB', img),
    ('Average Filter (5x5)', box_filtered),
    ('Gaussian Filter (5x5)', gaussian_filtered)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
