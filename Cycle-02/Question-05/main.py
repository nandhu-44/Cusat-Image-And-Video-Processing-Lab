import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = image.copy().astype(np.float32)
    
    if len(image.shape) == 3:
        # For color images, apply noise to each channel
        for channel in range(image.shape[2]):
            random_matrix = np.random.random(image.shape[:2])
            salt_mask = random_matrix < salt_prob
            pepper_mask = random_matrix > (1 - pepper_prob)
            noisy_image[:, :, channel][salt_mask] = 255
            noisy_image[:, :, channel][pepper_mask] = 0
    else:
        random_matrix = np.random.random(noisy_image.shape)
        salt_mask = random_matrix < salt_prob
        pepper_mask = random_matrix > (1 - pepper_prob)
        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask] = 0
    
    return noisy_image.astype(np.uint8)

def median_filter(image, kernel_size=3):
    pad_size = kernel_size // 2
    
    if len(image.shape) == 3:
        # For color images, apply filter to each channel
        filtered_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            padded_channel = np.pad(image[:, :, channel], pad_size, mode='edge')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    neighborhood = padded_channel[i:i+kernel_size, j:j+kernel_size]
                    filtered_image[i, j, channel] = np.median(neighborhood)
    else:
        padded_image = np.pad(image, pad_size, mode='edge')
        filtered_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
                filtered_image[i, j] = np.median(neighborhood)
    
    return filtered_image.astype(np.uint8)

def main():
    image = cv2.imread("../images/lena.jpg")
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    noisy_image = add_salt_pepper_noise(original)
    filtered_3x3 = median_filter(noisy_image, 3)
    filtered_7x7 = median_filter(noisy_image, 7)
    filtered_11x11 = median_filter(noisy_image, 11)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(noisy_image)
    plt.title('Salt & Pepper Noise')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(filtered_3x3)
    plt.title('Median Filter 3x3')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(filtered_7x7)
    plt.title('Median Filter 7x7')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(filtered_11x11)
    plt.title('Median Filter 11x11')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
