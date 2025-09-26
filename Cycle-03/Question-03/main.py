import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(rgb_image):
    """
    Convert RGB image to HSI color space
    
    Parameters:
    rgb_image: Input RGB image (numpy array)
    
    Returns:
    H, S, I: Hue, Saturation, and Intensity components
    """
    # Normalize RGB values to [0, 1]
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    R = rgb_normalized[:, :, 2]  # OpenCV uses BGR, so R is index 2
    G = rgb_normalized[:, :, 1]
    B = rgb_normalized[:, :, 0]  # B is index 0
    
    # Calculate Intensity
    I = (R + G + B) / 3.0
    
    # Calculate Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    
    # Avoid division by zero
    non_zero_mask = I > 1e-6
    S[non_zero_mask] = 1 - (min_rgb[non_zero_mask] / I[non_zero_mask])
    
    # Calculate Hue
    H = np.zeros_like(I)
    
    # Calculate numerator and denominator for hue
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B))
    
    # Avoid division by zero
    valid_mask = denominator > 1e-6
    
    # Calculate theta (angle)
    theta = np.zeros_like(I)
    theta[valid_mask] = np.arccos(np.clip(numerator[valid_mask] / denominator[valid_mask], -1, 1))
    
    # Determine hue based on B <= G or B > G
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    
    # Convert hue from radians to degrees and normalize to [0, 360]
    H = H * 180 / np.pi
    
    # Handle the case where S = 0 (achromatic)
    H[S < 1e-6] = 0
    
    return H, S, I

def hsi_to_rgb(H, S, I):
    """
    Convert HSI components back to RGB
    
    Parameters:
    H: Hue component (0-360 degrees)
    S: Saturation component (0-1)
    I: Intensity component (0-1)
    
    Returns:
    RGB image
    """
    # Initialize RGB arrays
    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)
    
    # Convert hue to radians
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
    
    # Clip values to valid range
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    # Combine into RGB image (convert back to BGR for OpenCV)
    rgb_image = np.stack([B, G, R], axis=2)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

def display_hsi_components(rgb_image, H, S, I):
    """
    Display the original RGB image and its HSI components
    """
    plt.figure(figsize=(16, 12))
    
    # Original RGB image
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB Image')
    plt.axis('off')
    
    # Hue component
    plt.subplot(2, 4, 2)
    plt.imshow(H, cmap='hsv')
    plt.title('Hue Component')
    plt.colorbar()
    plt.axis('off')
    
    # Saturation component
    plt.subplot(2, 4, 3)
    plt.imshow(S, cmap='gray')
    plt.title('Saturation Component')
    plt.colorbar()
    plt.axis('off')
    
    # Intensity component
    plt.subplot(2, 4, 4)
    plt.imshow(I, cmap='gray')
    plt.title('Intensity Component')
    plt.colorbar()
    plt.axis('off')
    
    # Display individual components as grayscale images
    plt.subplot(2, 4, 5)
    plt.imshow((H / 360 * 255).astype(np.uint8), cmap='gray')
    plt.title('Hue (Grayscale)')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow((S * 255).astype(np.uint8), cmap='gray')
    plt.title('Saturation (Grayscale)')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow((I * 255).astype(np.uint8), cmap='gray')
    plt.title('Intensity (Grayscale)')
    plt.axis('off')
    
    # Reconstructed RGB from HSI
    reconstructed_rgb = hsi_to_rgb(H, S, I)
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(reconstructed_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Reconstructed RGB')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

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
    print(f"Hue range: [{np.min(H):.2f}, {np.max(H):.2f}] degrees")
    print(f"Saturation range: [{np.min(S):.3f}, {np.max(S):.3f}]")
    print(f"Intensity range: [{np.min(I):.3f}, {np.max(I):.3f}]")
    
    # Display the results
    display_hsi_components(rgb_image, H, S, I)
    
    # Test reconstruction accuracy
    reconstructed = hsi_to_rgb(H, S, I)
    mse = np.mean((rgb_image.astype(np.float32) - reconstructed.astype(np.float32))**2)
    print(f"\nReconstruction MSE: {mse:.2f}")
    
    # Show histograms of HSI components
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(H.flatten(), bins=50, alpha=0.7, color='red')
    plt.title('Hue Histogram')
    plt.xlabel('Hue (degrees)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(S.flatten(), bins=50, alpha=0.7, color='green')
    plt.title('Saturation Histogram')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.hist(I.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Intensity Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()