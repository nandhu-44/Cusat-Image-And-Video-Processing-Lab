import cv2
import numpy as np
import matplotlib.pyplot as plt

def smoothing_average(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))

def smoothing_gaussian(img, kernel_size, sigma):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def smoothing_median(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

def sharpening_laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(img - laplacian)
    return sharpened

def sharpening_unsharp_mask(img, kernel_size, sigma, amount):
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened

def edge_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(np.clip(sobel, 0, 255))

def edge_prewitt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    return np.uint8(np.clip(prewitt, 0, 255))

def edge_roberts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    robertsx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
    robertsy = cv2.filter2D(gray, cv2.CV_64F, kernely)
    roberts = np.sqrt(robertsx**2 + robertsy**2)
    return np.uint8(np.clip(roberts, 0, 255))

def edge_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))

def emboss(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed = cv2.filter2D(gray, cv2.CV_64F, kernel)
    return np.uint8(np.clip(embossed + 128, 0, 255))

img = cv2.imread('../images/elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = {
    'Original': img,
    'Avg Smooth': smoothing_average(img, 5),
    'Gaussian': smoothing_gaussian(img, 5, 1),
    'Median': smoothing_median(img, 5),
    'Laplacian Sharp': sharpening_laplacian(img),
    'Unsharp Mask': sharpening_unsharp_mask(img, 5, 1, 1.5),
    'Sobel': edge_sobel(img),
    'Prewitt': edge_prewitt(img),
    'Roberts': edge_roberts(img),
    'Laplacian Edge': edge_laplacian(img),
    'Emboss': emboss(img)
}

plt.figure(figsize=(15, 10))
for i, (title, result) in enumerate(results.items()):
    plt.subplot(3, 4, i+1)
    plt.imshow(result, cmap='gray' if len(result.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()