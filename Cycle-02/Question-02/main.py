import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in Color
img = cv2.imread('../images/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 500
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

# Averaging
counts = [2, 8, 16, 32, 128]
results = []

for N in counts:
    accum = np.zeros_like(img, dtype=np.float64)
    for _ in range(N):
        noisy = add_noise(img)
        accum += noisy
    avg = accum / N
    results.append(np.clip(avg, 0, 255).astype(np.uint8))

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

for i, N in enumerate(counts):
    plt.subplot(2, 3, i+2)
    plt.imshow(results[i])
    plt.title(f'Avg of {N}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('output.png')
plt.show()
