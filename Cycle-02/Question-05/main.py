import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    noisy = img.copy()
    total_pixels = img.size
    
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy[tuple(salt_coords)] = 255
    
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy[tuple(pepper_coords)] = 0
    
    return noisy

def median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

img = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)

noisy = add_salt_pepper_noise(img, 0.05, 0.05)

median3 = median_filter(noisy, 3)
median5 = median_filter(noisy, 5)
median7 = median_filter(noisy, 7)

mean3 = cv2.blur(noisy, (3, 3))
mean5 = cv2.blur(noisy, (5, 5))

results = [
    ('Original', img),
    ('Salt & Pepper Noise', noisy),
    ('Median 3x3', median3),
    ('Median 5x5', median5),
    ('Median 7x7', median7),
    ('Mean 3x3 (Compare)', mean3),
    ('Mean 5x5 (Compare)', mean5)
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

psnr_median3 = cv2.PSNR(img, median3)
psnr_median5 = cv2.PSNR(img, median5)
psnr_mean3 = cv2.PSNR(img, mean3)

print(f"PSNR Median 3x3: {psnr_median3:.2f} dB")
print(f"PSNR Median 5x5: {psnr_median5:.2f} dB")
print(f"PSNR Mean 3x3: {psnr_mean3:.2f} dB")