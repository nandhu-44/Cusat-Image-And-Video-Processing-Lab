import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def histogram_equalization_grayscale(image):
    """Perform histogram equalization on grayscale image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray_image)
    
    return gray_image, equalized

def histogram_equalization_color(image):
    """Perform histogram equalization on color image using different methods"""
    
    # Method 1: Equalize each channel separately
    channels = cv2.split(image)
    equalized_channels = []
    for channel in channels:
        equalized_channel = cv2.equalizeHist(channel)
        equalized_channels.append(equalized_channel)
    equalized_bgr = cv2.merge(equalized_channels)
    
    # Method 2: Convert to YUV, equalize Y channel, convert back
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    equalized_yuv = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    
    return equalized_bgr, equalized_yuv

def plot_histogram_comparison(original, equalized, title="Histogram Comparison"):
    """Plot clean comparison of histograms before and after equalization"""
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(2, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Equalized image
    plt.subplot(2, 2, 2)
    if len(equalized.shape) == 3:
        plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(equalized, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    # Original histogram
    plt.subplot(2, 2, 3)
    if len(original.shape) == 3:
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        plt.legend()
    else:
        hist = cv2.calcHist([original], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Equalized histogram
    plt.subplot(2, 2, 4)
    if len(equalized.shape) == 3:
        colors = ('blue', 'green', 'red')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([equalized], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()} channel')
        plt.legend()
    else:
        hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def manual_histogram_equalization(image):
    """Manual implementation of histogram equalization"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Get image dimensions
    height, width = gray_image.shape
    total_pixels = height * width
    
    # Calculate histogram
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
    
    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF
    cdf_normalized = cdf * 255 / total_pixels
    
    # Create lookup table
    lookup_table = np.round(cdf_normalized).astype(np.uint8)
    
    # Apply lookup table to image
    equalized = lookup_table[gray_image]
    
    return equalized

def main():
    # Path to the image
    image_path = "../images/glucose-strip.jpg"
    
    try:
        # Read the image
        print("Reading image...")
        image = read_image(image_path)
        print("Image read successfully!")
        
        # Grayscale histogram equalization
        print("Performing grayscale histogram equalization...")
        gray_original, gray_equalized = histogram_equalization_grayscale(image)
        plot_histogram_comparison(gray_original, gray_equalized, "Grayscale Histogram Equalization")
        
        # Color histogram equalization
        print("Performing color histogram equalization...")
        equalized_bgr, equalized_yuv = histogram_equalization_color(image)
        
        plot_histogram_comparison(image, equalized_bgr, "Color Histogram Equalization (BGR)")
        plot_histogram_comparison(image, equalized_yuv, "Color Histogram Equalization (YUV)")
        
        # Manual histogram equalization
        print("Performing manual histogram equalization...")
        manual_equalized = manual_histogram_equalization(image)
        plot_histogram_comparison(gray_original, manual_equalized, "Manual Histogram Equalization")
        
        print("Histogram equalization completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
