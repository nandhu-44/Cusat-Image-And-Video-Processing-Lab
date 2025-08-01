import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path):
    """Read an image from the given path"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return image

def image_translation(image, tx, ty):
    """
    Translate an image by tx pixels horizontally and ty pixels vertically
    """
    height, width = image.shape[:2]
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return translated

def image_rotation(image, angle, center=None):
    """
    Rotate an image by given angle (in degrees)
    """
    height, width = image.shape[:2]
    
    # If center not specified, use image center
    if center is None:
        center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated

def image_scaling(image, scale_x, scale_y):
    """
    Scale an image by scale_x and scale_y factors
    """
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    
    # Apply scaling
    scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return scaled

def image_skewing(image, skew_x=0.2, skew_y=0.1):
    """
    Apply skewing transformation to an image
    """
    height, width = image.shape[:2]
    
    # Create skewing transformation matrix
    # [1, skew_x, 0]
    # [skew_y, 1, 0]
    skew_matrix = np.float32([[1, skew_x, 0], [skew_y, 1, 0]])
    
    # Calculate new dimensions to avoid cropping
    new_width = int(width + abs(skew_x * height))
    new_height = int(height + abs(skew_y * width))
    
    # Apply skewing
    skewed = cv2.warpAffine(image, skew_matrix, (new_width, new_height))
    
    return skewed

def advanced_rotation_with_resize(image, angle):
    """
    Rotate image and adjust canvas size to fit the entire rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation with new dimensions
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated

def display_transformation_results(original, transformed, title):
    """Display original and transformed images side by side"""
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Transformed image
    plt.subplot(1, 2, 2)
    if len(transformed.shape) == 3:
        plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(transformed, cmap='gray')
    plt.title(f'Transformed: {title}')
    plt.axis('off')
    
    plt.suptitle(f'Geometrical Transformation: {title}', fontsize=14)
    plt.tight_layout()
    plt.show()

def display_all_transformations(image):
    """Display all geometric transformations in a grid"""
    # Resize image for better display
    display_image = cv2.resize(image, (300, 300))
    
    # Apply all transformations
    translated = image_translation(display_image, 50, 30)
    rotated = image_rotation(display_image, 45)
    scaled = image_scaling(display_image, 0.8, 1.2)
    skewed = image_skewing(display_image, 0.3, 0.1)
    
    # Advanced rotation that fits entire image
    rotated_full = advanced_rotation_with_resize(display_image, 30)
    
    # Create comprehensive display
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    if len(display_image.shape) == 3:
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(display_image, cmap='gray')
    plt.title('Original (300x300)')
    plt.axis('off')
    
    # Translation
    plt.subplot(3, 3, 2)
    if len(translated.shape) == 3:
        plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(translated, cmap='gray')
    plt.title('Translation (50, 30)')
    plt.axis('off')
    
    # Rotation
    plt.subplot(3, 3, 3)
    if len(rotated.shape) == 3:
        plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(rotated, cmap='gray')
    plt.title('Rotation (45°)')
    plt.axis('off')
    
    # Scaling
    plt.subplot(3, 3, 4)
    if len(scaled.shape) == 3:
        plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(scaled, cmap='gray')
    plt.title('Scaling (0.8x, 1.2y)')
    plt.axis('off')
    
    # Skewing
    plt.subplot(3, 3, 5)
    if len(skewed.shape) == 3:
        plt.imshow(cv2.cvtColor(skewed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(skewed, cmap='gray')
    plt.title('Skewing (0.3, 0.1)')
    plt.axis('off')
    
    # Advanced rotation
    plt.subplot(3, 3, 6)
    if len(rotated_full.shape) == 3:
        plt.imshow(cv2.cvtColor(rotated_full, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(rotated_full, cmap='gray')
    plt.title('Full Rotation (30°)')
    plt.axis('off')
    
    # Combined transformation example
    plt.subplot(3, 3, 7)
    # Apply multiple transformations
    combined = image_translation(display_image, 20, 20)
    combined = image_rotation(combined, 15)
    combined = image_scaling(combined, 0.9, 0.9)
    
    if len(combined.shape) == 3:
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(combined, cmap='gray')
    plt.title('Combined Transforms')
    plt.axis('off')
    
    # Show transformation matrix example
    plt.subplot(3, 3, 8)
    plt.text(0.1, 0.8, 'Transformation Matrices:', fontsize=12, weight='bold')
    plt.text(0.1, 0.6, 'Translation:\n[1, 0, tx]\n[0, 1, ty]', fontsize=10, fontfamily='monospace')
    plt.text(0.1, 0.3, 'Rotation:\n[cos θ, -sin θ, 0]\n[sin θ, cos θ, 0]', fontsize=10, fontfamily='monospace')
    plt.text(0.1, 0.1, 'Scaling:\n[sx, 0, 0]\n[0, sy, 0]', fontsize=10, fontfamily='monospace')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Summary
    plt.subplot(3, 3, 9)
    plt.text(0.5, 0.5, 'Geometrical\nTransformations\nCompleted!', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=14, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.suptitle('All Geometrical Transformations', fontsize=16)
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
        
        print("\nPerforming geometrical transformations...")
        
        # a. Translation
        print("a. Translation...")
        translated = image_translation(image, 100, 50)
        display_transformation_results(image, translated, "Translation (100, 50)")
        
        # b. Rotation
        print("b. Rotation...")
        rotated = image_rotation(image, 30)
        display_transformation_results(image, rotated, "Rotation (30°)")
        
        # c. Scaling
        print("c. Scaling...")
        scaled = image_scaling(image, 0.7, 1.3)
        display_transformation_results(image, scaled, "Scaling (0.7x, 1.3y)")
        
        # d. Skewing
        print("d. Skewing...")
        skewed = image_skewing(image, 0.2, 0.1)
        display_transformation_results(image, skewed, "Skewing (0.2, 0.1)")
        
        # Additional: Advanced rotation that shows full image
        print("Advanced rotation (full image visible)...")
        rotated_full = advanced_rotation_with_resize(image, 45)
        display_transformation_results(image, rotated_full, "Advanced Rotation (45°)")
        
        # Display all transformations together
        print("\nDisplaying all transformations together...")
        display_all_transformations(image)
        
        print("All geometrical transformations completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()