import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("../images/lena.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = img1.copy()

# Arithmetic Operations
add = cv2.add(img1, img2)
sub = cv2.subtract(img1, img2)
mul = cv2.multiply(img1, img2)
div = cv2.divide(img1, img2)

# Plotting
images = [img1, img2, add, sub, mul, div]
titles = ["Image 1", "Image 2", "Addition", "Subtraction", "Multiplication", "Division"]

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("output.png")
plt.show()
