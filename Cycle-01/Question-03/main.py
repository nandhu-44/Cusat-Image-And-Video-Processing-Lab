import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/lenna.jpg', 0)
equ = cv2.equalizeHist(img)

plt.figure(figsize=(10, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, range=[0, 256])
plt.title('Original Histogram')

# Equalized
plt.subplot(2, 2, 3)
plt.imshow(equ, cmap='gray')
plt.title('Equalized')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(equ.ravel(), 256, range=[0, 256])
plt.title('Equalized Histogram')

plt.tight_layout()
plt.savefig('output.png')
plt.show()
