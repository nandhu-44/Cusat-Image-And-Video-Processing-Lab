import cv2
import numpy as np

# Load the image and resize to 256x256 if necessary
image = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to float32
image_float = np.float32(image)

# Function to perform DFT filtering
def dft_filter(img, pad=False):
    if pad:
        # Pad to 512x512
        padded = cv2.copyMakeBorder(img, 0, 256, 0, 256, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = img

    # Perform DFT
    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a low-pass filter mask (simple circular)
    rows, cols = padded.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 50  # radius
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1

    # Apply mask
    fshift = dft_shift * mask

    # Inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    if pad:
        # Crop back to 256x256
        img_back = img_back[:256, :256]

    return img_back

# Without padding
filtered_no_pad = dft_filter(image_float, pad=False)

# With padding
filtered_pad = dft_filter(image_float, pad=True)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Filtered without padding', filtered_no_pad)
cv2.imshow('Filtered with padding', filtered_pad)
cv2.waitKey(0)
cv2.destroyAllWindows()
