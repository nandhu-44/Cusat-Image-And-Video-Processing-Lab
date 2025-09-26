import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_motion_blur_psf(size, length, angle):
    """Create a motion blur PSF"""
    kernel = np.zeros((size, size), dtype=np.float32)
    angle_rad = np.radians(angle)
    center = size // 2
    
    for i in range(length):
        x = int(center + i * np.cos(angle_rad))
        y = int(center + i * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    return kernel / np.sum(kernel)

def degrade_image(image, psf, noise_var=0.01):
    """Degrade image with PSF and noise"""
    image_float = image.astype(np.float32) / 255.0
    blurred = cv2.filter2D(image_float, -1, psf)
    noise = np.random.normal(0, np.sqrt(noise_var), image_float.shape)
    degraded = blurred + noise
    degraded = np.clip(degraded, 0, 1)
    return (degraded * 255).astype(np.uint8)

def wiener_filter_constant_ratio(degraded_image, psf, K=0.01):
    """
    Wiener filtering with constant noise-to-signal ratio
    
    Parameters:
    degraded_image: Degraded input image
    psf: Point Spread Function
    K: Constant noise-to-signal power ratio
    
    Returns:
    Restored image
    """
    # Convert to grayscale if needed
    if len(degraded_image.shape) == 3:
        degraded_gray = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2GRAY)
    else:
        degraded_gray = degraded_image
    
    degraded_float = degraded_gray.astype(np.float32) / 255.0
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
    
    # Wiener filter: W(u,v) = H*(u,v) / (|H(u,v)|^2 + K)
    psf_conj = np.conj(psf_fft)
    psf_magnitude_squared = np.abs(psf_fft)**2
    
    wiener_filter = psf_conj / (psf_magnitude_squared + K)
    restored_fft = degraded_fft * wiener_filter
    
    # Inverse FFT
    restored = np.fft.ifft2(restored_fft)
    restored = np.real(restored)
    restored = np.clip(restored, 0, 1)
    
    return (restored * 255).astype(np.uint8)

def estimate_autocorrelation(image):
    """
    Estimate autocorrelation function of the image
    """
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # Remove mean
    image_centered = image_float - np.mean(image_float)
    
    # Compute autocorrelation using FFT
    image_fft = np.fft.fft2(image_centered)
    power_spectrum = np.abs(image_fft)**2
    autocorr = np.fft.ifft2(power_spectrum)
    autocorr = np.real(autocorr)
    
    # Shift zero frequency to center
    autocorr = np.fft.fftshift(autocorr)
    
    return autocorr

def wiener_filter_autocorr(degraded_image, psf, noise_var=0.01):
    """
    Wiener filtering using autocorrelation function
    
    Parameters:
    degraded_image: Degraded input image
    psf: Point Spread Function
    noise_var: Noise variance
    
    Returns:
    Restored image
    """
    # Convert to grayscale if needed
    if len(degraded_image.shape) == 3:
        degraded_gray = cv2.cvtColor(degraded_image, cv2.COLOR_BGR2GRAY)
    else:
        degraded_gray = degraded_image
    
    degraded_float = degraded_gray.astype(np.float32) / 255.0
    rows, cols = degraded_float.shape
    crow, ccol = rows // 2, cols // 2
    
    # Estimate signal power spectrum from autocorrelation
    autocorr = estimate_autocorrelation(degraded_gray)
    signal_power = np.abs(np.fft.fft2(np.fft.ifftshift(autocorr)))
    
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
    
    # Wiener filter using power spectrum
    psf_conj = np.conj(psf_fft)
    psf_magnitude_squared = np.abs(psf_fft)**2
    
    # Noise power is assumed uniform
    noise_power = noise_var * np.ones_like(signal_power)
    
    # Wiener filter: W(u,v) = H*(u,v) * S(u,v) / (|H(u,v)|^2 * S(u,v) + N(u,v))
    wiener_filter = (psf_conj * signal_power) / (psf_magnitude_squared * signal_power + noise_power)
    restored_fft = degraded_fft * wiener_filter
    
    # Inverse FFT
    restored = np.fft.ifft2(restored_fft)
    restored = np.real(restored)
    restored = np.clip(restored, 0, 1)
    
    return (restored * 255).astype(np.uint8)

def main():
    # Load the image
    image_path = '../images/lena.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    print("Original image loaded successfully!")
    
    # Create motion blur PSF
    psf_size = 15
    motion_length = 10
    angle = 30
    psf = create_motion_blur_psf(psf_size, motion_length, angle)
    
    print(f"Created motion blur PSF: {psf_size}x{psf_size}, length={motion_length}, angle={angle}°")
    
    # Degrade the image
    noise_var = 0.005
    degraded = degrade_image(image, psf, noise_var)
    print("Image degraded with motion blur and noise")
    
    # Restore using Wiener filter with constant ratio
    K_values = [0.001, 0.01, 0.1]
    restored_constant = []
    
    for K in K_values:
        restored = wiener_filter_constant_ratio(degraded, psf, K)
        restored_constant.append(restored)
        print(f"Wiener filtering with constant ratio K={K} completed")
    
    # Restore using Wiener filter with autocorrelation
    restored_autocorr = wiener_filter_autocorr(degraded, psf, noise_var)
    print("Wiener filtering with autocorrelation completed")
    
    # Display results
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # PSF
    plt.subplot(2, 4, 2)
    plt.imshow(psf, cmap='hot')
    plt.title('Motion Blur PSF')
    plt.axis('off')
    
    # Degraded image
    plt.subplot(2, 4, 3)
    plt.imshow(degraded, cmap='gray')
    plt.title('Degraded Image')
    plt.axis('off')
    
    # Wiener filter with constant ratios
    for i, (restored, K) in enumerate(zip(restored_constant, K_values)):
        plt.subplot(2, 4, 4 + i)
        plt.imshow(restored, cmap='gray')
        plt.title(f'Wiener (K={K})')
        plt.axis('off')
    
    # Wiener filter with autocorrelation
    plt.subplot(2, 4, 8)
    plt.imshow(restored_autocorr, cmap='gray')
    plt.title('Wiener (Autocorr)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics
    print(f"\nQuality Metrics (MSE):")
    mse_degraded = np.mean((image.astype(np.float32) - degraded.astype(np.float32))**2)
    print(f"Original vs Degraded: {mse_degraded:.2f}")
    
    for i, (restored, K) in enumerate(zip(restored_constant, K_values)):
        mse = np.mean((image.astype(np.float32) - restored.astype(np.float32))**2)
        print(f"Original vs Wiener (K={K}): {mse:.2f}")
    
    mse_autocorr = np.mean((image.astype(np.float32) - restored_autocorr.astype(np.float32))**2)
    print(f"Original vs Wiener (Autocorr): {mse_autocorr:.2f}")

if __name__ == "__main__":
    main()