import cv2
import numpy as np
import os

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def get_image_info(image):
    """Get basic information about the image"""
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    data_type = image.dtype
    size_bytes = image.nbytes
    
    info = {
        'height': height,
        'width': width,
        'channels': channels,
        'data_type': data_type,
        'size_bytes': size_bytes,
        'shape': image.shape
    }
    
    return info

def calculate_compression_ratio(original_path, compressed_path):
    """Calculate compression ratio between original and compressed image"""
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size
    return compression_ratio

def create_negative(image):
    """Create negative of an image"""
    if len(image.shape) == 3:
        negative = 255 - image
    else:
        negative = 255 - image
    return negative

def resize_for_display(image, max_width=800, max_height=600):
    """Resize image for display while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    else:
        return image

def main():
    image_path = "../images/glucose-strip.jpg"
    
    try:
        # a. Read an image
        print("Reading image...")
        image = read_image(image_path)
        print("Image read successfully!")
        
        # b. Get image information
        print("\nGetting image information...")
        info = get_image_info(image)
        print("Image Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # c. Calculate compression ratio (file size vs memory size)
        print("\nCalculating compression ratio...")
        file_size = os.path.getsize(image_path)
        memory_size = image.nbytes
        compression_ratio = memory_size / file_size
        compression_level = ((memory_size - file_size) / memory_size) * 100
        
        print(f"File size on disk: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"Size in memory: {memory_size:,} bytes ({memory_size / 1024 / 1024:.2f} MB)")
        print(f"Compression ratio (memory/file): {compression_ratio:.2f}x")
        print(f"Compression level: {compression_level:.2f}% space saved")
        
        # d. Display negative of the image
        print("\nCreating negative image...")
        negative_image = create_negative(image)
        
        # Resize images for display
        display_original = resize_for_display(image)
        display_negative = resize_for_display(negative_image)
        
        # Display images at normal size
        cv2.imshow('Original Image', display_original)
        cv2.imshow('Negative Image', display_negative)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
