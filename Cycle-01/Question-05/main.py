import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def brightness_enhancement(image, brightness_value=50):
    """Enhance brightness by adding a constant value"""
    if len(image.shape) == 3:
        enhanced = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * brightness_value)
    else:
        enhanced = cv2.add(image, brightness_value)
    return enhanced

def contrast_enhancement(image, alpha=1.5):
    """Enhance contrast using linear transformation: new_pixel = alpha * old_pixel"""
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return enhanced

def complement_image(image):
    """Create complement (negative) of an image"""
    complement = 255 - image
    return complement

def bi_level_contrast_enhancement(image, threshold=127):
    """Binary contrast enhancement - convert to bi-level (black/white)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def brightness_slicing(image, min_range=100, max_range=200, highlight_value=255):
    """Highlight pixels in a specific brightness range"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create mask for pixels in the specified range
    mask = cv2.inRange(gray, min_range, max_range)
    
    # Create result image
    result = gray.copy()
    result[mask > 0] = highlight_value
    
    return result

def low_pass_filtering(image, kernel_size=15):
    """Apply low-pass filter (blur) to remove high-frequency noise"""
    # Gaussian blur as low-pass filter
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return filtered

def high_pass_filtering(image, kernel_size=15):
    """Apply high-pass filter to enhance edges"""
    # Apply low-pass filter first
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # High-pass = Original - Low-pass
    if len(image.shape) == 3:
        high_pass = cv2.subtract(image, low_pass)
    else:
        high_pass = cv2.subtract(image, low_pass)
    
    # Add 128 to make it visible (since subtraction can give negative values)
    high_pass = cv2.add(high_pass, 128)
    
    return high_pass

def display_enhancement_results(original, enhanced, title):
    """Display original and enhanced images side by side"""
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Enhanced image
    plt.subplot(1, 2, 2)
    if len(enhanced.shape) == 3:
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(enhanced, cmap='gray')
    plt.title(f'Enhanced: {title}')
    plt.axis('off')
    
    plt.suptitle(f'Image Enhancement: {title}', fontsize=14)
    plt.tight_layout()
    plt.show()

def display_all_enhancements(image):
    """Display all enhancement operations in a grid"""
    # Convert to grayscale for some operations
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply all enhancements
    brightness_enhanced = brightness_enhancement(image, 50)
    contrast_enhanced = contrast_enhancement(image, 1.5)
    complement = complement_image(image)
    bi_level = bi_level_contrast_enhancement(image, 127)
    brightness_sliced = brightness_slicing(image, 100, 200, 255)
    low_pass = low_pass_filtering(image, 15)
    high_pass = high_pass_filtering(image, 15)
    
    # Create a comprehensive display
    plt.figure(figsize=(16, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Brightness enhancement
    plt.subplot(3, 3, 2)
    if len(brightness_enhanced.shape) == 3:
        plt.imshow(cv2.cvtColor(brightness_enhanced, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(brightness_enhanced, cmap='gray')
    plt.title('Brightness Enhanced')
    plt.axis('off')
    
    # Contrast enhancement
    plt.subplot(3, 3, 3)
    if len(contrast_enhanced.shape) == 3:
        plt.imshow(cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(contrast_enhanced, cmap='gray')
    plt.title('Contrast Enhanced')
    plt.axis('off')
    
    # Complement
    plt.subplot(3, 3, 4)
    if len(complement.shape) == 3:
        plt.imshow(cv2.cvtColor(complement, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(complement, cmap='gray')
    plt.title('Complement (Negative)')
    plt.axis('off')
    
    # Bi-level
    plt.subplot(3, 3, 5)
    plt.imshow(bi_level, cmap='gray')
    plt.title('Bi-level (Binary)')
    plt.axis('off')
    
    # Brightness slicing
    plt.subplot(3, 3, 6)
    plt.imshow(brightness_sliced, cmap='gray')
    plt.title('Brightness Slicing')
    plt.axis('off')
    
    # Low-pass filtering
    plt.subplot(3, 3, 7)
    if len(low_pass.shape) == 3:
        plt.imshow(cv2.cvtColor(low_pass, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(low_pass, cmap='gray')
    plt.title('Low-pass Filter (Blur)')
    plt.axis('off')
    
    # High-pass filtering
    plt.subplot(3, 3, 8)
    if len(high_pass.shape) == 3:
        plt.imshow(cv2.cvtColor(high_pass, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(high_pass, cmap='gray')
    plt.title('High-pass Filter (Edges)')
    plt.axis('off')
    
    # Leave last subplot empty or add a summary
    plt.subplot(3, 3, 9)
    plt.text(0.5, 0.5, 'Image Enhancement\nOperations\nCompleted!', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=14, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.suptitle('All Image Enhancement Operations', fontsize=16)
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
        
        print("\nPerforming image enhancement operations...")
        
        # a. Brightness enhancement
        print("a. Brightness enhancement...")
        brightness_enhanced = brightness_enhancement(image, 50)
        display_enhancement_results(image, brightness_enhanced, "Brightness Enhancement")
        
        # b. Contrast enhancement
        print("b. Contrast enhancement...")
        contrast_enhanced = contrast_enhancement(image, 1.5)
        display_enhancement_results(image, contrast_enhanced, "Contrast Enhancement")
        
        # c. Complement of an image
        print("c. Complement of an image...")
        complement = complement_image(image)
        display_enhancement_results(image, complement, "Complement (Negative)")
        
        # d. Bi-level or binary contrast enhancement
        print("d. Bi-level contrast enhancement...")
        bi_level = bi_level_contrast_enhancement(image, 127)
        display_enhancement_results(image, bi_level, "Bi-level (Binary)")
        
        # e. Brightness slicing
        print("e. Brightness slicing...")
        brightness_sliced = brightness_slicing(image, 100, 200, 255)
        display_enhancement_results(image, brightness_sliced, "Brightness Slicing")
        
        # f. Low-pass filtering
        print("f. Low-pass filtering...")
        low_pass = low_pass_filtering(image, 15)
        display_enhancement_results(image, low_pass, "Low-pass Filter")
        
        # g. High-pass filtering
        print("g. High-pass filtering...")
        high_pass = high_pass_filtering(image, 15)
        display_enhancement_results(image, high_pass, "High-pass Filter")
        
        # Display all enhancements in one comprehensive view
        print("\nDisplaying all enhancements together...")
        display_all_enhancements(image)
        
        print("All image enhancement operations completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
