import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/lenna.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

plt.figure(figsize=(10, 8))

# Original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, range=[0, 256])
plt.title('Original Histogram')

# Local Equalized (CLAHE)
plt.subplot(2, 2, 3)
plt.imshow(cl1, cmap='gray')
plt.title('Local Histogram Equalization (CLAHE)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(cl1.ravel(), 256, range=[0, 256])
plt.title('CLAHE Histogram')

plt.tight_layout()
plt.savefig('output.png')
plt.show()
