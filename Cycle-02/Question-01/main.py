import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def resize_images_to_match(img1, img2):
    """Resize both images to the same dimensions (smaller of the two)"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use the smaller dimensions
    min_height = min(h1, h2)
    min_width = min(w1, w2)
    
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))
    
    return img1_resized, img2_resized

def image_addition(img1, img2):
    """Add two images"""
    # Method 1: Using cv2.add (handles overflow properly)
    result_cv2 = cv2.add(img1, img2)
    
    # Method 2: Using numpy addition with clipping
    result_np = np.clip(img1.astype(np.float64) + img2.astype(np.float64), 0, 255).astype(np.uint8)
    
    # Method 3: Weighted addition (50% each image)
    result_weighted = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    return result_cv2, result_np, result_weighted

def image_subtraction(img1, img2):
    """Subtract two images"""
    # Method 1: Using cv2.subtract (handles underflow properly)
    result_cv2 = cv2.subtract(img1, img2)
    
    # Method 2: Using numpy subtraction with clipping
    result_np = np.clip(img1.astype(np.float64) - img2.astype(np.float64), 0, 255).astype(np.uint8)
    
    # Method 3: Absolute difference
    result_abs = cv2.absdiff(img1, img2)
    
    return result_cv2, result_np, result_abs

def image_multiplication(img1, img2):
    """Multiply two images"""
    # Method 1: Element-wise multiplication (normalized)
    result_normalized = cv2.multiply(img1, img2, scale=1/255.0)
    
    # Method 2: Direct multiplication with clipping
    result_direct = np.clip((img1.astype(np.float64) * img2.astype(np.float64)) / 255.0, 0, 255).astype(np.uint8)
    
    # Method 3: Bitwise AND (for binary-like operations)
    result_bitwise = cv2.bitwise_and(img1, img2)
    
    return result_normalized, result_direct, result_bitwise

def image_division(img1, img2):
    """Divide two images"""
    # Method 1: Safe division with small epsilon to avoid division by zero
    epsilon = 1e-10
    result_safe = np.clip((img1.astype(np.float64) / (img2.astype(np.float64) + epsilon)) * 255, 0, 255).astype(np.uint8)
    
    # Method 2: Division with zero handling
    img2_safe = img2.copy()
    img2_safe[img2_safe == 0] = 1  # Replace zeros with 1
    result_zero_handle = np.clip((img1.astype(np.float64) / img2_safe.astype(np.float64)) * 255, 0, 255).astype(np.uint8)
    
    # Method 3: Normalized division
    result_normalized = np.clip((img1.astype(np.float64) / 255.0) / (img2.astype(np.float64) / 255.0 + epsilon) * 255, 0, 255).astype(np.uint8)
    
    return result_safe, result_zero_handle, result_normalized

def display_operation_results(img1, img2, results, operation_name):
    """Display results of an arithmetic operation"""
    plt.figure(figsize=(15, 10))
    
    # Original images
    plt.subplot(2, 3, 1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    # Results
    titles = ['Method 1', 'Method 2', 'Method 3']
    for i, (result, title) in enumerate(zip(results, titles)):
        plt.subplot(2, 3, i + 4)
        if len(result.shape) == 3:
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result, cmap='gray')
        plt.title(f'{operation_name} - {title}')
        plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.text(0.5, 0.5, f'{operation_name}\nOperation', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=16, weight='bold', transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.suptitle(f'Image Arithmetic: {operation_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

def display_all_operations(img1, img2):
    """Display all four arithmetic operations in one comprehensive view"""
    # Perform all operations
    add_results = image_addition(img1, img2)
    sub_results = image_subtraction(img1, img2)
    mul_results = image_multiplication(img1, img2)
    div_results = image_division(img1, img2)
    
    # Create comprehensive display (using first method of each operation)
    plt.figure(figsize=(16, 12))
    
    # Original images
    plt.subplot(3, 4, 1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    # Operations
    operations = [
        (add_results[0], 'Addition (cv2.add)'),
        (sub_results[0], 'Subtraction (cv2.subtract)'),
        (mul_results[0], 'Multiplication (normalized)'),
        (div_results[0], 'Division (safe)'),
        (add_results[2], 'Weighted Addition'),
        (sub_results[2], 'Absolute Difference'),
        (mul_results[2], 'Bitwise AND'),
        (div_results[1], 'Division (zero handled)')
    ]
    
    for i, (result, title) in enumerate(operations):
        plt.subplot(3, 4, i + 5)
        if len(result.shape) == 3:
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Info panel
    plt.subplot(3, 4, 3)
    plt.text(0.1, 0.8, 'Arithmetic Operations:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, '• Addition: I1 + I2', fontsize=10)
    plt.text(0.1, 0.5, '• Subtraction: I1 - I2', fontsize=10)
    plt.text(0.1, 0.4, '• Multiplication: I1 × I2', fontsize=10)
    plt.text(0.1, 0.3, '• Division: I1 ÷ I2', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.text(0.5, 0.5, 'Image\nArithmetic\nOperations\nCompleted!', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=14, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.suptitle('All Image Arithmetic Operations', fontsize=16)
    plt.tight_layout()
    plt.show()

def create_test_images():
    """Create simple test images if glucose strip image is not available"""
    # Create a simple gradient image
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        img1[i, :] = [i * 255 // 200, 100, 150]
    
    # Create a circular pattern image
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    center = (100, 100)
    for i in range(200):
        for j in range(200):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            intensity = int(255 * (1 - min(dist / 100, 1)))
            img2[i, j] = [intensity, intensity // 2, intensity // 3]
    
    return img1, img2

def main():
    # Path to the images
    image_path1 = "../images/glucose-strip.jpg"
    
    try:
        # Try to read the primary image
        print("Reading image...")
        img1 = read_image(image_path1)
        print("Image read successfully!")
        
        # Create a modified version of the same image as second image
        # This could be the same image with different processing
        img2 = cv2.GaussianBlur(img1, (15, 15), 0)  # Blurred version
        
        # Alternatively, create a simple test pattern
        # img2 = np.full_like(img1, 100)  # Uniform gray image
        
        print("Using glucose strip image and its blurred version...")
        
    except Exception as e:
        print(f"Could not read glucose strip image: {e}")
        print("Creating test images...")
        img1, img2 = create_test_images()
    
    try:
        # Ensure both images have the same size
        img1, img2 = resize_images_to_match(img1, img2)
        
        print(f"Image dimensions: {img1.shape}")
        print("\nPerforming arithmetic operations between images...")
        
        # Perform and display each operation individually
        print("1. Addition...")
        add_results = image_addition(img1, img2)
        display_operation_results(img1, img2, add_results, "Addition")
        
        print("2. Subtraction...")
        sub_results = image_subtraction(img1, img2)
        display_operation_results(img1, img2, sub_results, "Subtraction")
        
        print("3. Multiplication...")
        mul_results = image_multiplication(img1, img2)
        display_operation_results(img1, img2, mul_results, "Multiplication")
        
        print("4. Division...")
        div_results = image_division(img1, img2)
        display_operation_results(img1, img2, div_results, "Division")
        
        # Display all operations together
        print("\nDisplaying all operations together...")
        display_all_operations(img1, img2)
        
        print("All arithmetic operations completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
