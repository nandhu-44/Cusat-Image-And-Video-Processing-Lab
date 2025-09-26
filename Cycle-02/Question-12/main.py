import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Define kernels for line detection
horizontal_kernel = np.array([[-1, -1, -1],
                             [ 2,  2,  2],
                             [-1, -1, -1]], dtype=np.float32)

vertical_kernel = np.array([[-1,  2, -1],
                           [-1,  2, -1],
                           [-1,  2, -1]], dtype=np.float32)

diagonal_kernel = np.array([[ 2, -1, -1],
                           [-1,  2, -1],
                           [-1, -1,  2]], dtype=np.float32)

# Apply filters
horizontal_lines = cv2.filter2D(image, -1, horizontal_kernel)
vertical_lines = cv2.filter2D(image, -1, vertical_kernel)
diagonal_lines = cv2.filter2D(image, -1, diagonal_kernel)

# Normalize for display
horizontal_lines = cv2.normalize(horizontal_lines, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
vertical_lines = cv2.normalize(vertical_lines, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
diagonal_lines = cv2.normalize(diagonal_lines, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Horizontal Lines', horizontal_lines)
cv2.imshow('Vertical Lines', vertical_lines)
cv2.imshow('Diagonal Lines', diagonal_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
