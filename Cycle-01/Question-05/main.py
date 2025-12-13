import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../images/lenna.jpg', 0)

# a. Brightness Enhancement
bright = cv2.add(img, 50)

# b. Contrast Enhancement
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# c. Complement
comp = 255 - img

# d. Bi-level (Binary)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# e. Brightness Slicing (Preserving background)
slicing = img.copy()
mask = (img >= 100) & (img <= 200)
slicing[mask] = 255

# f. Low-pass Filtering (Gaussian Blur)
low_pass = cv2.GaussianBlur(img, (9, 9), 0)

# g. High-pass Filtering (Laplacian)
high_pass = cv2.Laplacian(img, cv2.CV_64F)
high_pass = np.uint8(np.absolute(high_pass))

# Plotting
titles = ['Original', 'Brightness (+50)', 'Contrast (x1.5)', 'Complement', 
          'Bi-level (Binary)', 'Brightness Slicing', 'Low-pass (Blur)', 'High-pass (Laplacian)']
images = [img, bright, contrast, comp, binary, slicing, low_pass, high_pass]

plt.figure(figsize=(10, 10))
for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout(w_pad=0.1)
plt.savefig('output.png')
plt.show()
