import cv2
import numpy as np
import matplotlib.pyplot as plt

def sobel_edge(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(np.clip(sobel, 0, 255))

def prewitt_edge(img):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(img, cv2.CV_64F, kernely)
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    return np.uint8(np.clip(prewitt, 0, 255))

def roberts_edge(img):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    robertsx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(img, cv2.CV_64F, kernely)
    roberts = np.sqrt(robertsx**2 + robertsy**2)
    return np.uint8(np.clip(roberts, 0, 255))

def log_edge(img, kernel_size=5, sigma=1.4):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))

def canny_edge(img, low=50, high=150):
    return cv2.Canny(img, low, high)

img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)

sobel = sobel_edge(img)
prewitt = prewitt_edge(img)
roberts = roberts_edge(img)
log = log_edge(img)
canny = canny_edge(img)

results = [
    ('Original', img),
    ('Sobel', sobel),
    ('Prewitt', prewitt),
    ('Roberts', roberts),
    ('LoG', log),
    ('Canny', canny)
]

plt.figure(figsize=(15, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig('edge_detection.png', dpi=150, bbox_inches='tight')
plt.show()

def edge_density(edge_img):
    return np.sum(edge_img > 0) / edge_img.size * 100

def edge_strength(edge_img):
    return np.mean(edge_img[edge_img > 0]) if np.any(edge_img > 0) else 0

print("Edge Detection Performance Metrics:")
print(f"Sobel - Density: {edge_density(sobel):.2f}%, Strength: {edge_strength(sobel):.2f}")
print(f"Prewitt - Density: {edge_density(prewitt):.2f}%, Strength: {edge_strength(prewitt):.2f}")
print(f"Roberts - Density: {edge_density(roberts):.2f}%, Strength: {edge_strength(roberts):.2f}")
print(f"LoG - Density: {edge_density(log):.2f}%, Strength: {edge_strength(log):.2f}")
print(f"Canny - Density: {edge_density(canny):.2f}%, Strength: {edge_strength(canny):.2f}")