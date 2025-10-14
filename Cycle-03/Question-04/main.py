import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(rgb_image):
    """
    Convert RGB image to HSI color space
    """
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    R = rgb_normalized[:, :, 2]  # OpenCV uses BGR
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 0]
    
    # Calculate Intensity
    I = (R + G + B) / 3.0
    
    # Calculate Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    non_zero_mask = I > 1e-6
    S[non_zero_mask] = 1 - (min_rgb[non_zero_mask] / I[non_zero_mask])
    
    # Calculate Hue
    H = np.zeros_like(I)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B))
    valid_mask = denominator > 1e-6
    theta = np.zeros_like(I)
    theta[valid_mask] = np.arccos(np.clip(numerator[valid_mask] / denominator[valid_mask], -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H * 180 / np.pi
    H[S < 1e-6] = 0
    
    return H, S, I

def hsi_to_rgb(H, S, I):
    """
    Convert HSI components back to RGB
    """
    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)
    
    H_rad = H * np.pi / 180
    
    # RG sector (0 <= H < 120)
    mask1 = (H >= 0) & (H < 120)
    if np.any(mask1):
        H_sector = H_rad[mask1]
        S_sector = S[mask1]
        I_sector = I[mask1]
        B[mask1] = I_sector * (1 - S_sector)
        R[mask1] = I_sector * (1 + (S_sector * np.cos(H_sector)) / np.cos(np.pi/3 - H_sector))
        G[mask1] = 3 * I_sector - (R[mask1] + B[mask1])
    
    # GB sector (120 <= H < 240)
    mask2 = (H >= 120) & (H < 240)
    if np.any(mask2):
        H_sector = H_rad[mask2] - 2 * np.pi / 3
        S_sector = S[mask2]
        I_sector = I[mask2]
        R[mask2] = I_sector * (1 - S_sector)
        G[mask2] = I_sector * (1 + (S_sector * np.cos(H_sector)) / np.cos(np.pi/3 - H_sector))
        B[mask2] = 3 * I_sector - (R[mask2] + G[mask2])
    
    # BR sector (240 <= H <= 360)
    mask3 = (H >= 240) & (H <= 360)
    if np.any(mask3):
        H_sector = H_rad[mask3] - 4 * np.pi / 3
        S_sector = S[mask3]
        I_sector = I[mask3]
        G[mask3] = I_sector * (1 - S_sector)
        B[mask3] = I_sector * (1 + (S_sector * np.cos(H_sector)) / np.cos(np.pi/3 - H_sector))
        R[mask3] = 3 * I_sector - (G[mask3] + B[mask3])
    
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    rgb_image = np.stack([B, G, R], axis=2)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

def histogram_equalization(intensity):
    """
    Perform histogram equalization on intensity component
    """
    # Convert to uint8 for histogram calculation
    intensity_uint8 = (intensity * 255).astype(np.uint8)
    
    # Equalize histogram
    equalized = cv2.equalizeHist(intensity_uint8)
    
    # Convert back to float
    equalized_float = equalized.astype(np.float32) / 255.0
    
    return equalized_float

def main():
    # Load the color image
    image_path = '../images/lena.jpg'
    rgb_image = cv2.imread(image_path)
    
    if rgb_image is None:
        print("Error: Could not load image")
        return
    
    print("Color image loaded successfully!")
    print(f"Image shape: {rgb_image.shape}")
    
    # Convert RGB to HSI
    print("Converting RGB to HSI...")
    H, S, I = rgb_to_hsi(rgb_image)
    
    print("RGB to HSI conversion completed!")
    
    # Perform histogram equalization on Intensity component
    print("Performing histogram equalization on Intensity component...")
    I_equalized = histogram_equalization(I)
    
    print("Histogram equalization completed!")
    
    # Convert back to RGB with equalized intensity
    print("Converting HSI back to RGB...")
    rgb_equalized = hsi_to_rgb(H, S, I_equalized)
    
    print("Conversion completed!")
    
    # Display results
    plt.figure(figsize=(18, 12))
    
    # Original RGB image
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB Image')
    plt.axis('off')
    
    # Original Hue
    plt.subplot(3, 4, 2)
    plt.imshow(H, cmap='hsv')
    plt.title('Hue (Original)')
    plt.colorbar()
    plt.axis('off')
    
    # Original Saturation
    plt.subplot(3, 4, 3)
    plt.imshow(S, cmap='gray')
    plt.title('Saturation (Original)')
    plt.colorbar()
    plt.axis('off')
    
    # Original Intensity
    plt.subplot(3, 4, 4)
    plt.imshow(I, cmap='gray')
    plt.title('Intensity (Original)')
    plt.colorbar()
    plt.axis('off')
    
    # Histogram of original intensity
    plt.subplot(3, 4, 5)
    plt.hist(I.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Original Intensity Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Equalized Intensity
    plt.subplot(3, 4, 6)
    plt.imshow(I_equalized, cmap='gray')
    plt.title('Intensity (Equalized)')
    plt.colorbar()
    plt.axis('off')
    
    # Histogram of equalized intensity
    plt.subplot(3, 4, 7)
    plt.hist(I_equalized.flatten(), bins=50, alpha=0.7, color='green')
    plt.title('Equalized Intensity Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Comparison of histograms
    plt.subplot(3, 4, 8)
    plt.hist(I.flatten(), bins=50, alpha=0.5, color='blue', label='Original')
    plt.hist(I_equalized.flatten(), bins=50, alpha=0.5, color='green', label='Equalized')
    plt.title('Histogram Comparison')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # New HSI image (equalized)
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(rgb_equalized, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced RGB Image\n(Equalized Intensity)')
    plt.axis('off')
    
    # Side by side comparison
    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(cv2.cvtColor(rgb_equalized, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced')
    plt.axis('off')
    
    # Difference image
    plt.subplot(3, 4, 12)
    diff = cv2.absdiff(rgb_image, rgb_equalized)
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    plt.title('Difference Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    print("\nImage Statistics:")
    print(f"Original Intensity - Mean: {np.mean(I):.3f}, Std: {np.std(I):.3f}")
    print(f"Equalized Intensity - Mean: {np.mean(I_equalized):.3f}, Std: {np.std(I_equalized):.3f}")
    
    # Calculate contrast improvement
    contrast_original = np.std(I)
    contrast_equalized = np.std(I_equalized)
    improvement = ((contrast_equalized - contrast_original) / contrast_original) * 100
    print(f"Contrast improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()