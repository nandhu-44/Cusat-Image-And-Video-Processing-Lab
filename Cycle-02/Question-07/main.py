import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/elephant.jpg')

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Define a linear spatial filter kernel (e.g., averaging filter 3x3)
kernel = np.ones((3, 3), np.float32) / 9

# Apply the filter
smoothed_image = cv2.filter2D(image, -1, kernel)

# Display the original and smoothed images
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
