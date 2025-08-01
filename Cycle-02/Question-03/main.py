import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def create_test_image():
    """Create a test image with various features"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(200):
        for j in range(200):
            img[i, j] = [i//2, j//2, 100]
    
    # Add shapes with different intensities
    cv2.rectangle(img, (50, 50), (100, 100), (255, 255, 255), -1)
    cv2.circle(img, (150, 150), 25, (0, 0, 0), -1)
    cv2.rectangle(img, (120, 30), (180, 70), (128, 128, 128), -1)
    
    # Add some lines for edge detection
    cv2.line(img, (0, 100), (200, 100), (255, 0, 0), 2)
    cv2.line(img, (100, 0), (100, 200), (0, 255, 0), 2)
    
    return img

# =============================================================================
# SMOOTHING/LOW-PASS FILTERS
# =============================================================================

def averaging_filter(image, kernel_size):
    """Average/Box filter - simplest smoothing filter"""
    return cv2.blur(image, (kernel_size, kernel_size))

def gaussian_filter(image, kernel_size, sigma_x=0, sigma_y=0):
    """Gaussian filter - weighted smoothing with Gaussian weights"""
    if sigma_x == 0:
        sigma_x = kernel_size / 6  # Rule of thumb
    if sigma_y == 0:
        sigma_y = sigma_x
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigma_y)

def median_filter(image, kernel_size):
    """Median filter - good for salt and pepper noise"""
    return cv2.medianBlur(image, kernel_size)

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter - edge-preserving smoothing"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def morphological_opening(image, kernel_size):
    """Morphological opening - erosion followed by dilation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def morphological_closing(image, kernel_size):
    """Morphological closing - dilation followed by erosion"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# =============================================================================
# SHARPENING/HIGH-PASS FILTERS
# =============================================================================

def laplacian_filter(image):
    """Laplacian filter - edge enhancement"""
    # Convert to grayscale for Laplacian
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Convert back to 3 channels
        laplacian = cv2.convertScaleAbs(laplacian)
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    else:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

def laplacian_sharpening(image, alpha=1.0):
    """Laplacian sharpening - original + alpha * laplacian"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(image, 1.0, laplacian_3ch, alpha, 0)
    else:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        sharpened = cv2.addWeighted(image, 1.0, laplacian, alpha, 0)
    return sharpened

def unsharp_masking(image, kernel_size=5, sigma=1.0, alpha=1.5, beta=-0.5):
    """Unsharp masking - enhances edges while preserving overall appearance"""
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    # Create sharpened image: original + alpha*(original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, beta, 0)
    return sharpened

def high_boost_filter(image, kernel_size=5, boost_factor=1.5):
    """High boost filter - amplifies high frequency components"""
    # Create low-pass filtered version
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # High boost: (boost_factor * original) - low_pass
    high_boost = cv2.addWeighted(image, boost_factor, low_pass, -1.0, 0)
    return high_boost

# =============================================================================
# EDGE DETECTION FILTERS
# =============================================================================

def sobel_filter(image):
    """Sobel edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobel X and Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(sobel_magnitude, cv2.COLOR_GRAY2BGR)
    return sobel_magnitude

def prewitt_filter(image):
    """Prewitt edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    
    prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt_magnitude = cv2.convertScaleAbs(prewitt_magnitude)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(prewitt_magnitude, cv2.COLOR_GRAY2BGR)
    return prewitt_magnitude

def roberts_filter(image):
    """Roberts cross-gradient edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Roberts kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    roberts_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    
    roberts_magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    roberts_magnitude = cv2.convertScaleAbs(roberts_magnitude)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(roberts_magnitude, cv2.COLOR_GRAY2BGR)
    return roberts_magnitude

def canny_filter(image, low_threshold=50, high_threshold=150):
    """Canny edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges

# =============================================================================
# CUSTOM CONVOLUTION FILTERS
# =============================================================================

def custom_convolution(image, kernel):
    """Apply custom convolution kernel"""
    return cv2.filter2D(image, -1, kernel)

def create_custom_kernels():
    """Create various custom kernels"""
    kernels = {}
    
    # Edge detection kernels
    kernels['horizontal_edge'] = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    kernels['vertical_edge'] = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
    kernels['diagonal_edge_1'] = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)
    kernels['diagonal_edge_2'] = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
    
    # Emboss kernels
    kernels['emboss'] = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    
    # Sharpening kernels
    kernels['sharpen_1'] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kernels['sharpen_2'] = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    
    # Motion blur kernels
    motion_blur = np.zeros((9, 9), dtype=np.float32)
    motion_blur[4, :] = 1/9  # Horizontal motion blur
    kernels['motion_blur_horizontal'] = motion_blur
    
    motion_blur_diag = np.zeros((9, 9), dtype=np.float32)
    for i in range(9):
        motion_blur_diag[i, i] = 1/9  # Diagonal motion blur
    kernels['motion_blur_diagonal'] = motion_blur_diag
    
    return kernels

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_smoothing_filters(image):
    """Display all smoothing filters"""
    plt.figure(figsize=(16, 12))
    
    # Original
    plt.subplot(3, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Smoothing filters
    filters = [
        (averaging_filter(image, 5), 'Average Filter (5x5)'),
        (gaussian_filter(image, 5, 1.0), 'Gaussian Filter (σ=1)'),
        (median_filter(image, 5), 'Median Filter (5x5)'),
        (bilateral_filter(image), 'Bilateral Filter'),
        (morphological_opening(image, 5), 'Morphological Opening'),
        (morphological_closing(image, 5), 'Morphological Closing'),
        (gaussian_filter(image, 15, 3.0), 'Heavy Gaussian Blur'),
        (averaging_filter(image, 15), 'Heavy Average Filter')
    ]
    
    for i, (filtered_img, title) in enumerate(filters):
        plt.subplot(3, 4, i + 2)
        if len(filtered_img.shape) == 3:
            plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(filtered_img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Info panel
    plt.subplot(3, 4, 10)
    plt.text(0.1, 0.8, 'Smoothing Filters:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, '• Remove noise', fontsize=10)
    plt.text(0.1, 0.5, '• Blur details', fontsize=10)
    plt.text(0.1, 0.4, '• Low-pass filtering', fontsize=10)
    plt.text(0.1, 0.3, '• Reduce high frequencies', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle('Linear Spatial Filters - Smoothing/Low-Pass Filters', fontsize=16)
    plt.tight_layout()
    plt.show()

def display_sharpening_filters(image):
    """Display all sharpening filters"""
    plt.figure(figsize=(16, 10))
    
    # Original
    plt.subplot(2, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Sharpening filters
    filters = [
        (laplacian_filter(image), 'Laplacian Filter'),
        (laplacian_sharpening(image, 0.5), 'Laplacian Sharpening'),
        (unsharp_masking(image), 'Unsharp Masking'),
        (high_boost_filter(image, 5, 1.5), 'High Boost Filter'),
        (laplacian_sharpening(image, 1.0), 'Strong Laplacian Sharp'),
        (unsharp_masking(image, 3, 1.0, 2.0), 'Strong Unsharp Mask')
    ]
    
    for i, (filtered_img, title) in enumerate(filters):
        plt.subplot(2, 4, i + 2)
        if len(filtered_img.shape) == 3:
            plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(filtered_img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Info panel
    plt.subplot(2, 4, 8)
    plt.text(0.1, 0.8, 'Sharpening Filters:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, '• Enhance edges', fontsize=10)
    plt.text(0.1, 0.5, '• Increase contrast', fontsize=10)
    plt.text(0.1, 0.4, '• High-pass filtering', fontsize=10)
    plt.text(0.1, 0.3, '• Amplify details', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle('Linear Spatial Filters - Sharpening/High-Pass Filters', fontsize=16)
    plt.tight_layout()
    plt.show()

def display_edge_detection_filters(image):
    """Display all edge detection filters"""
    plt.figure(figsize=(16, 8))
    
    # Original
    plt.subplot(2, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Edge detection filters
    filters = [
        (sobel_filter(image), 'Sobel Edge Detection'),
        (prewitt_filter(image), 'Prewitt Edge Detection'),
        (roberts_filter(image), 'Roberts Edge Detection'),
        (canny_filter(image, 50, 150), 'Canny Edge Detection'),
        (laplacian_filter(image), 'Laplacian (Zero Crossing)'),
        (canny_filter(image, 100, 200), 'Canny (High Threshold)')
    ]
    
    for i, (filtered_img, title) in enumerate(filters):
        plt.subplot(2, 4, i + 2)
        if len(filtered_img.shape) == 3:
            plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(filtered_img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Info panel
    plt.subplot(2, 4, 8)
    plt.text(0.1, 0.8, 'Edge Detection:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, '• Find boundaries', fontsize=10)
    plt.text(0.1, 0.5, '• Detect gradients', fontsize=10)
    plt.text(0.1, 0.4, '• Feature extraction', fontsize=10)
    plt.text(0.1, 0.3, '• Object recognition', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle('Linear Spatial Filters - Edge Detection Filters', fontsize=16)
    plt.tight_layout()
    plt.show()

def display_custom_filters(image):
    """Display custom convolution filters"""
    kernels = create_custom_kernels()
    
    plt.figure(figsize=(16, 12))
    
    # Original
    plt.subplot(3, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Custom filters
    filter_names = [
        'horizontal_edge', 'vertical_edge', 'diagonal_edge_1', 'diagonal_edge_2',
        'emboss', 'sharpen_1', 'sharpen_2', 'motion_blur_horizontal',
        'motion_blur_diagonal'
    ]
    
    for i, filter_name in enumerate(filter_names):
        if i < 11:  # Limit to available subplot positions
            plt.subplot(3, 4, i + 2)
            kernel = kernels[filter_name]
            filtered_img = custom_convolution(image, kernel)
            
            if len(filtered_img.shape) == 3:
                plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(filtered_img, cmap='gray')
            plt.title(filter_name.replace('_', ' ').title())
            plt.axis('off')
    
    # Info panel
    plt.subplot(3, 4, 12)
    plt.text(0.1, 0.8, 'Custom Kernels:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, '• Directional filters', fontsize=10)
    plt.text(0.1, 0.5, '• Special effects', fontsize=10)
    plt.text(0.1, 0.4, '• Custom convolution', fontsize=10)
    plt.text(0.1, 0.3, '• Feature detection', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle('Linear Spatial Filters - Custom Convolution Filters', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # Path to the image
    image_path = "../images/glucose-strip.jpg"
    
    try:
        # Try to read the image
        print("Reading image...")
        image = read_image(image_path)
        print("Image read successfully!")
    except Exception as e:
        print(f"Could not read image: {e}")
        print("Creating test image...")
        image = create_test_image()
    
    try:
        print(f"Image dimensions: {image.shape}")
        print("\nImplementing all types of linear spatial filters...")
        
        # Display all filter categories
        print("\n1. Displaying smoothing/low-pass filters...")
        display_smoothing_filters(image)
        
        print("2. Displaying sharpening/high-pass filters...")
        display_sharpening_filters(image)
        
        print("3. Displaying edge detection filters...")
        display_edge_detection_filters(image)
        
        print("4. Displaying custom convolution filters...")
        display_custom_filters(image)
        
        print("\nAll linear spatial filters implemented and displayed!")
        
        print("\nFilter Categories Summary:")
        print("=" * 60)
        print("1. SMOOTHING FILTERS (Low-pass):")
        print("   - Average/Box filter")
        print("   - Gaussian filter")
        print("   - Median filter")
        print("   - Bilateral filter")
        print("   - Morphological opening/closing")
        print()
        print("2. SHARPENING FILTERS (High-pass):")
        print("   - Laplacian filter")
        print("   - Laplacian sharpening")
        print("   - Unsharp masking")
        print("   - High boost filter")
        print()
        print("3. EDGE DETECTION FILTERS:")
        print("   - Sobel filter")
        print("   - Prewitt filter")
        print("   - Roberts filter")
        print("   - Canny edge detector")
        print("   - Laplacian edge detector")
        print()
        print("4. CUSTOM CONVOLUTION FILTERS:")
        print("   - Directional edge filters")
        print("   - Emboss filters")
        print("   - Motion blur filters")
        print("   - Custom kernels")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
