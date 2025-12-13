import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../images/lenna.jpg')
colors = ('b', 'g', 'r')

plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Image')
plt.axis('off')

# Histogram
plt.subplot(1, 2, 2)
for i, color in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title('Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('output.png')
plt.show()
