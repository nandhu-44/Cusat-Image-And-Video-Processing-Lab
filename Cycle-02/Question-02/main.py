import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def add_noise_to_image(image, noise_type='gaussian', amount=0.1):
    """Add different types of noise to an image"""
    # Convert BGR to RGB for skimage
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Normalize to [0, 1] for skimage
    image_normalized = image_rgb.astype(np.float64) / 255.0
    
    if noise_type == 'gaussian':
        noisy = random_noise(image_normalized, mode='gaussian', var=amount)
    elif noise_type == 'salt_pepper':
        noisy = random_noise(image_normalized, mode='s&p', amount=amount)
    elif noise_type == 'poisson':
        noisy = random_noise(image_normalized, mode='poisson')
    elif noise_type == 'speckle':
        noisy = random_noise(image_normalized, mode='speckle', var=amount)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Convert back to [0, 255] and uint8
    noisy_image = (noisy * 255).astype(np.uint8)
    
    # Convert back to BGR if original was BGR
    if len(image.shape) == 3:
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    
    return noisy_image

def averaging_filter(image, kernel_size):
    """Apply averaging filter with specified kernel size"""
    return cv2.blur(image, (kernel_size, kernel_size))

def reduce_noise_by_averaging(image, factors):
    """Reduce noise by averaging with different factors"""
    results = {}
    
    for factor in factors:
        # Apply averaging filter
        filtered = averaging_filter(image, factor)
        results[factor] = filtered
    
    return results

def calculate_image_quality_metrics(original, processed):
    """Calculate quality metrics between original and processed images"""
    # Convert to grayscale for calculations if needed
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        proc_gray = processed
    
    # Mean Squared Error (MSE)
    mse = np.mean((orig_gray.astype(np.float64) - proc_gray.astype(np.float64)) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Structural Similarity Index (SSIM) - simplified version
    # Using correlation coefficient as a simplified SSIM
    orig_flat = orig_gray.flatten().astype(np.float64)
    proc_flat = proc_gray.flatten().astype(np.float64)
    
    if np.std(orig_flat) > 0 and np.std(proc_flat) > 0:
        correlation = np.corrcoef(orig_flat, proc_flat)[0, 1]
    else:
        correlation = 0
    
    return mse, psnr, correlation

def display_noise_reduction_results(original, noisy, filtered_results, factors):
    """Display all noise reduction results"""
    num_results = len(factors)
    cols = min(4, num_results + 2)  # Original, noisy, + filtered results
    rows = (num_results + 4) // cols  # Ensure enough rows
    
    plt.figure(figsize=(16, 4 * rows))
    
    # Original image
    plt.subplot(rows, cols, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Noisy image
    plt.subplot(rows, cols, 2)
    if len(noisy.shape) == 3:
        plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(noisy, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    # Filtered results
    for i, factor in enumerate(factors):
        plt.subplot(rows, cols, i + 3)
        result = filtered_results[factor]
        if len(result.shape) == 3:
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result, cmap='gray')
        
        # Calculate quality metrics
        mse, psnr, corr = calculate_image_quality_metrics(original, result)
        plt.title(f'Averaging {factor}x{factor}\nPSNR: {psnr:.2f}dB')
        plt.axis('off')
    
    plt.suptitle('Noise Reduction by Averaging with Different Kernel Sizes', fontsize=16)
    plt.tight_layout()
    plt.show()

def display_quality_comparison(original, noisy, filtered_results, factors):
    """Display quality metrics comparison"""
    metrics = {
        'Factor': [],
        'MSE': [],
        'PSNR': [],
        'Correlation': []
    }
    
    # Calculate metrics for noisy image
    mse_noisy, psnr_noisy, corr_noisy = calculate_image_quality_metrics(original, noisy)
    
    print("Quality Metrics Comparison:")
    print("=" * 60)
    print(f"{'Method':<15} {'MSE':<10} {'PSNR (dB)':<12} {'Correlation':<12}")
    print("-" * 60)
    print(f"{'Noisy':<15} {mse_noisy:<10.2f} {psnr_noisy:<12.2f} {corr_noisy:<12.3f}")
    
    # Calculate metrics for each filtered result
    for factor in factors:
        result = filtered_results[factor]
        mse, psnr, corr = calculate_image_quality_metrics(original, result)
        
        metrics['Factor'].append(f"{factor}x{factor}")
        metrics['MSE'].append(mse)
        metrics['PSNR'].append(psnr)
        metrics['Correlation'].append(corr)
        
        print(f"{'Avg ' + str(factor) + 'x' + str(factor):<15} {mse:<10.2f} {psnr:<12.2f} {corr:<12.3f}")
    
    # Plot quality metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE plot
    ax1.bar(metrics['Factor'], metrics['MSE'])
    ax1.set_title('Mean Squared Error (Lower is Better)')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # PSNR plot
    ax2.bar(metrics['Factor'], metrics['PSNR'])
    ax2.set_title('Peak Signal-to-Noise Ratio (Higher is Better)')
    ax2.set_ylabel('PSNR (dB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Correlation plot
    ax3.bar(metrics['Factor'], metrics['Correlation'])
    ax3.set_title('Correlation with Original (Higher is Better)')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Find the best result
    best_psnr_idx = np.argmax(metrics['PSNR'])
    best_corr_idx = np.argmax(metrics['Correlation'])
    best_mse_idx = np.argmin(metrics['MSE'])
    
    print("\nBest Results:")
    print(f"Best PSNR: {metrics['Factor'][best_psnr_idx]} averaging ({metrics['PSNR'][best_psnr_idx]:.2f} dB)")
    print(f"Best Correlation: {metrics['Factor'][best_corr_idx]} averaging ({metrics['Correlation'][best_corr_idx]:.3f})")
    print(f"Lowest MSE: {metrics['Factor'][best_mse_idx]} averaging ({metrics['MSE'][best_mse_idx]:.2f})")

def create_test_image():
    """Create a simple test image with clear features"""
    # Create a test image with geometric shapes
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Background
    img[:, :] = [50, 50, 50]
    
    # Rectangle
    cv2.rectangle(img, (50, 50), (100, 100), (255, 0, 0), -1)
    
    # Circle
    cv2.circle(img, (150, 150), 30, (0, 255, 0), -1)
    
    # Triangle (using lines)
    pts = np.array([[150, 50], [120, 100], [180, 100]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))
    
    # Add some texture
    for i in range(0, 200, 20):
        cv2.line(img, (i, 0), (i, 200), (100, 100, 100), 1)
        cv2.line(img, (0, i), (200, i), (100, 100, 100), 1)
    
    return img

def main():
    # Path to the image
    image_path = "../images/glucose-strip.jpg"
    
    try:
        # Try to read the image
        print("Reading image...")
        original_image = read_image(image_path)
        print("Image read successfully!")
    except Exception as e:
        print(f"Could not read image: {e}")
        print("Creating test image...")
        original_image = create_test_image()
    
    try:
        print(f"Image dimensions: {original_image.shape}")
        
        # Add noise to the image
        print("\nAdding noise to the image...")
        
        # You can experiment with different noise types and amounts
        noise_types = [
            ('gaussian', 0.05),
            ('salt_pepper', 0.1),
            ('speckle', 0.1)
        ]
        
        print("Available noise types:")
        for i, (noise_type, amount) in enumerate(noise_types):
            print(f"{i+1}. {noise_type.replace('_', ' ').title()} (amount: {amount})")
        
        # Use Gaussian noise by default
        selected_noise = noise_types[0]
        noisy_image = add_noise_to_image(original_image, selected_noise[0], selected_noise[1])
        
        print(f"Applied {selected_noise[0]} noise with amount {selected_noise[1]}")
        
        # Define averaging factors as specified in the question
        averaging_factors = [2, 8, 16, 32, 128]
        
        print(f"\nApplying averaging filters with factors: {averaging_factors}")
        
        # Apply noise reduction with different averaging factors
        filtered_results = reduce_noise_by_averaging(noisy_image, averaging_factors)
        
        # Display results
        print("\nDisplaying results...")
        display_noise_reduction_results(original_image, noisy_image, filtered_results, averaging_factors)
        
        # Display quality comparison
        print("\nAnalyzing quality metrics...")
        display_quality_comparison(original_image, noisy_image, filtered_results, averaging_factors)
        
        print("\nNoise reduction analysis completed!")
        print("\nObservations:")
        print("- Small averaging kernels (2x2, 8x8) preserve details but may not remove all noise")
        print("- Large averaging kernels (32x32, 128x128) remove noise but blur the image significantly")
        print("- Medium-sized kernels (16x16) often provide the best trade-off")
        print("- The optimal kernel size depends on the noise level and desired detail preservation")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
