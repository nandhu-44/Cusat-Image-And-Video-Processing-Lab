import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def plot_histogram_grayscale(image):
    """Plot histogram for grayscale image"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

def plot_histogram_color(image):
    """Plot histogram for color image (BGR channels)"""
    colors = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title('Original Image')
    plt.axis('off')
    
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        
        plt.subplot(1, 4, i + 2)
        plt.plot(hist, color=color)
        plt.title(f'{name} Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

def plot_combined_histogram(image):
    """Plot combined histogram showing all channels"""
    colors = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=name, alpha=0.7)
    
    plt.title('Combined Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    image_path = "../images/glucose-strip.jpg"
    
    try:
        print("Reading image...")
        image = read_image(image_path)
        print("Image read successfully!")
        
        print("Plotting grayscale histogram...")
        plot_histogram_grayscale(image)
        
        print("Plotting color histograms...")
        plot_histogram_color(image)
        
        print("Plotting combined histogram...")
        plot_combined_histogram(image)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
