import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

# Check if image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to float32
image_float = np.float32(image)

# Perform DFT
dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = image.shape
crow, ccol = rows//2, cols//2

def ideal_lowpass(dft_shift, cutoff):
    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= cutoff**2
    mask[mask_area] = 1
    return dft_shift * mask

def butterworth_lowpass(dft_shift, cutoff, n=2):
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mask = 1 / (1 + (dist / cutoff)**(2*n))
    return dft_shift * mask[:, :, np.newaxis]

def gaussian_lowpass(dft_shift, cutoff):
    x, y = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - crow)**2 + (y - ccol)**2)
    mask = np.exp(-(dist**2) / (2 * (cutoff**2)))
    return dft_shift * mask[:, :, np.newaxis]

# Apply filters
cutoff = 50

ideal_filtered = ideal_lowpass(dft_shift, cutoff)
butter_filtered = butterworth_lowpass(dft_shift, cutoff)
gauss_filtered = gaussian_lowpass(dft_shift, cutoff)

# Inverse DFT function
def inverse_dft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# Get filtered images
ideal_img = inverse_dft(ideal_filtered)
butter_img = inverse_dft(butter_filtered)
gauss_img = inverse_dft(gauss_filtered)

# Display
cv2.imshow('Original', image)
cv2.imshow('Ideal Low-pass', ideal_img)
cv2.imshow('Butterworth Low-pass', butter_img)
cv2.imshow('Gaussian Low-pass', gauss_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
