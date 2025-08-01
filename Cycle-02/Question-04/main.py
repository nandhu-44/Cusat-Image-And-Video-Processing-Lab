import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def read_kernel_from_file(file_path):
    """Read convolution kernel from ASCII text file"""
    kernel = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                row = [float(x) for x in line.split()]
                kernel.append(row)
    return np.array(kernel, dtype=np.float32)

def convolution(image, kernel):
    """Perform convolution operation"""
    if len(image.shape) == 3:
        # Convert to grayscale for simplicity
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Add padding
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize output
    output = np.zeros_like(image, dtype=np.float64)
    
    # Perform convolution (flip kernel for true convolution)
    flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(roi * flipped_kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)

def correlation(image, kernel):
    """Perform correlation operation"""
    if len(image.shape) == 3:
        # Convert to grayscale for simplicity
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Add padding
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Initialize output
    output = np.zeros_like(image, dtype=np.float64)
    
    # Perform correlation (no kernel flipping)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(roi * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)

def test_kernels(image):
    """Test the three required averaging kernels"""
    kernel_files = [
        'kernel_3x3_avg.txt',
        'kernel_7x7_avg.txt',
        'kernel_11x11_avg.txt'
    ]
    
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for i, kernel_file in enumerate(kernel_files):
        # Read kernel
        kernel = read_kernel_from_file(kernel_file)
        print(f"\nKernel from {kernel_file}:")
        print(f"Size: {kernel.shape}")
        print(f"Sum: {np.sum(kernel):.6f}")
        
        # Apply convolution and correlation
        conv_result = convolution(image, kernel)
        corr_result = correlation(image, kernel)
        
        # Display convolution result
        plt.subplot(2, 4, i + 2)
        plt.imshow(conv_result, cmap='gray')
        plt.title(f'Convolution\n{kernel.shape[0]}×{kernel.shape[1]} Avg')
        plt.axis('off')
        
        # Display correlation result
        plt.subplot(2, 4, i + 6)
        plt.imshow(corr_result, cmap='gray')
        plt.title(f'Correlation\n{kernel.shape[0]}×{kernel.shape[1]} Avg')
        plt.axis('off')
    
    plt.suptitle('Image Convolution and Correlation with Averaging Kernels', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    # Read elephant image
    try:
        print("Reading elephant image...")
        image = read_image("../images/elephant.jpg")
        print("Image loaded successfully!")
    except:
        print("Could not read elephant image. Please ensure elephant.jpg is in ../images/ folder")
        return
    
    print(f"Image dimensions: {image.shape}")
    
    # Test with the required kernels
    test_kernels(image)
    
    print("\nConvolution vs Correlation completed!")
    print("Note: For symmetric kernels like averaging, convolution and correlation give similar results.")

if __name__ == "__main__":
    main()
