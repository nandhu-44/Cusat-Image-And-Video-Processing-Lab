import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../images/lenna.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols = img.shape[:2]

# a. Translation
M_trans = np.float32([[1, 0, 100], [0, 1, 50]])
translation = cv2.warpAffine(img, M_trans, (cols, rows))

# b. Rotation
M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotation = cv2.warpAffine(img, M_rot, (cols, rows))

# c. Scaling
scaling = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

# d. Skewing
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_skew = cv2.getAffineTransform(pts1, pts2)
skewing = cv2.warpAffine(img, M_skew, (cols, rows))

# Plotting
titles = ['Translation (100, 50)', 'Rotation (45 deg)', 'Scaling (0.5x)', 'Skewing']
images = [translation, rotation, scaling, skewing]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('output.png')
plt.show()