import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_kernel(filename):
    return np.loadtxt(filename)

def convolve(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def correlate(img, kernel):
    kernel_flipped = np.flip(kernel)
    return cv2.filter2D(img, -1, kernel_flipped)

def create_kernel_file(size, filename):
    kernel = np.ones((size, size)) / (size * size)
    np.savetxt(filename, kernel, fmt='%.6f')

create_kernel_file(3, 'kernel_3x3.txt')
create_kernel_file(7, 'kernel_7x7.txt')
create_kernel_file(11, 'kernel_11x11.txt')

img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)

k3 = load_kernel('kernel_3x3.txt')
k7 = load_kernel('kernel_7x7.txt')
k11 = load_kernel('kernel_11x11.txt')

results = [
    ('Original', img),
    ('Convolve 3x3', convolve(img, k3)),
    ('Correlate 3x3', correlate(img, k3)),
    ('Convolve 7x7', convolve(img, k7)),
    ('Correlate 7x7', correlate(img, k7)),
    ('Convolve 11x11', convolve(img, k11)),
    ('Correlate 11x11', correlate(img, k11))
]

plt.figure(figsize=(12, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(3, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Adjust layout to minimize whitespace
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Kernel 3x3: \n{k3}" )
print(f"\nKernel 7x7: \n{k7}")
print(f"\nKernel 11x11: \n{k11}")