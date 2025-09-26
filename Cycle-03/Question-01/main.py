import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_degradation_psf(size, motion_length=10, angle=45):
    """
    Create a motion blur Point Spread Function (PSF)
    
    Parameters:
    size: Size of the PSF kernel
    motion_length: Length of the motion blur
    angle: Angle of the motion blur in degrees
    
    Returns:
    PSF kernel
    """
    # Create motion blur kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate center
    center = size // 2
    
    # Create line for motion blur
    for i in range(motion_length):
        x = int(center + i * np.cos(angle_rad))
        y = int(center + i * np.sin(angle_rad))
        
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel

def degrade_image(image, psf, noise_var=0.01):
    """
    Degrade an image using convolution with PSF and adding noise
    
    Parameters:
    image: Input image
    psf: Point Spread Function
    noise_var: Noise variance
    
    Returns:
    Degraded image
    """
    # Convert image to float
    image_float = image.astype(np.float32) / 255.0
    
    # Apply blur (convolution with PSF)
    blurred = cv2.filter2D(image_float, -1, psf)
    
    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_var), image_float.shape)
    degraded = blurred + noise
    
    # Clip values to valid range
    degraded = np.clip(degraded, 0, 1)
    
    return (degraded * 255).astype(np.uint8)

def direct_inverse_filtering(degraded_image, psf, epsilon=1e-6):
    """
    Restore image using direct inverse filtering
    
    Parameters:
    degraded_image: Degraded input image
    psf: Point Spread Function used for degradation
    epsilon: Small value to avoid division by zero
    
    Returns:
    Restored image
    """
    # Convert to grayscale if needed
    if len(degraded_image.shape) == 3:
        degraded_gray = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2GRAY)
    else:
        degraded_gray = degraded_image
    
    # Convert to float
    degraded_float = degraded_gray.astype(np.float32) / 255.0
    
    # Pad images for FFT
    rows, cols = degraded_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Pad PSF to same size as image
    psf_padded = np.zeros((rows, cols), dtype=np.float32)
    psf_rows, psf_cols = psf.shape
    psf_start_row = crow - psf_rows // 2
    psf_start_col = ccol - psf_cols // 2
    psf_padded[psf_start_row:psf_start_row + psf_rows, 
               psf_start_col:psf_start_col + psf_cols] = psf
    
    # Take FFT
    degraded_fft = np.fft.fft2(degraded_float)
    psf_fft = np.fft.fft2(psf_padded)
    
    # Direct inverse filtering: F(u,v) = G(u,v) / H(u,v)
    # Add epsilon to avoid division by zero
    psf_fft_safe = psf_fft + epsilon
    restored_fft = degraded_fft / psf_fft_safe
    
    # Take inverse FFT
    restored = np.fft.ifft2(restored_fft)
    restored = np.real(restored)
    
    # Normalize and convert back to uint8
    restored = np.clip(restored, 0, 1)
    restored = (restored * 255).astype(np.uint8)
    
    return restored

def main():
    # Load the image
    image_path = '../images/lena.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    print("Original image loaded successfully!")
    
    # Create degradation PSF (motion blur)
    psf_size = 15
    motion_length = 8
    angle = 45
    psf = create_degradation_psf(psf_size, motion_length, angle)
    
    print(f"Created motion blur PSF: {psf_size}x{psf_size}, length={motion_length}, angle={angle}°")
    
    # Degrade the image
    degraded = degrade_image(image, psf, noise_var=0.001)
    print("Image degraded with motion blur and noise")
    
    # Restore using direct inverse filtering
    restored = direct_inverse_filtering(degraded, psf, epsilon=1e-6)
    print("Image restored using direct inverse filtering")
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # PSF
    plt.subplot(1, 4, 2)
    plt.imshow(psf, cmap='hot')
    plt.title('Motion Blur PSF')
    plt.axis('off')
    
    # Degraded image
    plt.subplot(1, 4, 3)
    plt.imshow(degraded, cmap='gray')
    plt.title('Degraded Image\n(Blur + Noise)')
    plt.axis('off')
    
    # Restored image
    plt.subplot(1, 4, 4)
    plt.imshow(restored, cmap='gray')
    plt.title('Restored Image\n(Direct Inverse Filter)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics
    mse_degraded = np.mean((image.astype(np.float32) - degraded.astype(np.float32))**2)
    mse_restored = np.mean((image.astype(np.float32) - restored.astype(np.float32))**2)
    
    print(f"\nQuality Metrics:")
    print(f"MSE (Original vs Degraded): {mse_degraded:.2f}")
    print(f"MSE (Original vs Restored): {mse_restored:.2f}")
    print(f"Improvement: {((mse_degraded - mse_restored) / mse_degraded * 100):.2f}%")

if __name__ == "__main__":
    main()