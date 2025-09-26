import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/lena.jpg')

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to float for calculations
image_float = image.astype(np.float32)

# Define Laplacian kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float32)

# Apply Laplacian filter
laplacian = cv2.filter2D(image_float, -1, kernel)

# Sharpened image: original + Laplacian
sharpened = image_float - laplacian  # Note: subtracting for sharpening effect

# Clip to valid range and convert back to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Display the original and sharpened images
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
