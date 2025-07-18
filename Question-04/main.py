import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def local_histogram_equalization(image, tile_size=(8, 8), clip_limit=2.0):
    """
    Perform local histogram equalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    # Apply CLAHE
    equalized = clahe.apply(gray_image)
    
    return gray_image, equalized

def manual_local_histogram_equalization(image, window_size=64):
    """
    Manual implementation of local histogram equalization using sliding window
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    height, width = gray_image.shape
    result = np.zeros_like(gray_image)
    
    # Half window size for padding
    half_window = window_size // 2
    
    # Pad the image
    padded_image = np.pad(gray_image, half_window, mode='reflect')
    
    print(f"Processing {height}x{width} image with {window_size}x{window_size} local windows...")
    
    for i in range(height):
        for j in range(width):
            # Extract local window
            local_window = padded_image[i:i+window_size, j:j+window_size]
            
            # Calculate local histogram
            hist, _ = np.histogram(local_window.flatten(), bins=256, range=[0, 256])
            
            # Calculate CDF
            cdf = hist.cumsum()
            
            # Normalize CDF
            cdf_normalized = cdf * 255 / (window_size * window_size)
            
            # Apply transformation to center pixel
            center_pixel = gray_image[i, j]
            result[i, j] = cdf_normalized[center_pixel]
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{height} rows")
    
    return result.astype(np.uint8)

def adaptive_local_equalization(image, block_size=64, overlap=0.5):
    """
    Perform adaptive local histogram equalization with overlapping blocks
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    height, width = gray_image.shape
    result = np.zeros_like(gray_image, dtype=np.float64)
    count_matrix = np.zeros_like(gray_image, dtype=np.float64)
    
    step_size = int(block_size * (1 - overlap))
    
    print(f"Processing with block size {block_size}x{block_size} and {overlap*100}% overlap...")
    
    for i in range(0, height - block_size + 1, step_size):
        for j in range(0, width - block_size + 1, step_size):
            # Extract block
            block = gray_image[i:i+block_size, j:j+block_size]
            
            # Perform histogram equalization on block
            equalized_block = cv2.equalizeHist(block)
            
            # Add to result with overlap handling
            result[i:i+block_size, j:j+block_size] += equalized_block.astype(np.float64)
            count_matrix[i:i+block_size, j:j+block_size] += 1
    
    # Average overlapping regions
    count_matrix[count_matrix == 0] = 1  # Avoid division by zero
    result = result / count_matrix
    
    return result.astype(np.uint8)

def compare_methods(image):
    """Compare different local histogram equalization methods"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Global histogram equalization for comparison
    global_eq = cv2.equalizeHist(gray_image)
    
    # CLAHE with different parameters
    clahe_small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_large = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    
    local_eq_small = clahe_small.apply(gray_image)
    local_eq_large = clahe_large.apply(gray_image)
    
    # Clean comparison plot
    plt.figure(figsize=(16, 8))
    
    images = [gray_image, global_eq, local_eq_small, local_eq_large]
    titles = ['Original', 'Global Equalization', 'CLAHE (8x8 tiles)', 'CLAHE (16x16 tiles)']
    
    # Display images
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        
        # Display corresponding histogram
        plt.subplot(2, 4, i + 5)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='darkblue')
        plt.title(f'{title} - Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Local vs Global Histogram Equalization Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return local_eq_small, local_eq_large

def simple_comparison_display(gray_original, clahe_result, title="CLAHE Result"):
    """Simple side-by-side comparison"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray_original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(clahe_result, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    plt.suptitle('Local Histogram Equalization (CLAHE)', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    # Path to the image
    image_path = "../images/glucose-strip.jpg"
    
    try:
        # Read the image
        print("Reading image...")
        image = read_image(image_path)
        print("Image read successfully!")
        
        # Basic local histogram equalization using CLAHE
        print("Performing local histogram equalization using CLAHE...")
        gray_original, clahe_result = local_histogram_equalization(image, tile_size=(8, 8), clip_limit=2.0)
        
        # Simple display
        simple_comparison_display(gray_original, clahe_result, "CLAHE Result")
        
        # Compare different methods
        print("Comparing different local equalization methods...")
        local_small, local_large = compare_methods(image)
        
        print("Local histogram equalization completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
