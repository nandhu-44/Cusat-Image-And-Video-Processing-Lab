import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# 1. Create Degradation Function (Motion Blur PSF)
def get_motion_blur_psf(size=15, angle=45):
    psf = np.zeros((size, size))
    center = size // 2
    slope = np.tan(np.radians(angle))
    
    # Simple line drawing for kernel
    # For better accuracy we could use cv2.line or similar, but let's do a simple one
    # Actually, cv2.warpAffine is good for rotating a horizontal line
    kernel = np.zeros((size, size))
    kernel[center, :] = 1
    
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    psf = cv2.warpAffine(kernel, M, (size, size))
    
    # Normalize
    psf = psf / np.sum(psf)
    return psf

psf = get_motion_blur_psf(15, 45)

# 2. Degrade Image
# Convolution
img_float = img.astype(np.float32) / 255.0
blurred = cv2.filter2D(img_float, -1, psf)

# Add Noise
noise_var = 0.001
noise = np.random.normal(0, np.sqrt(noise_var), img.shape)
degraded = blurred + noise
degraded = np.clip(degraded, 0, 1)

# 3. Direct Inverse Filtering
def inverse_filter(degraded_img, psf, epsilon=1e-3):
    rows, cols = degraded_img.shape
    
    # Pad PSF to image size
    psf_padded = np.zeros((rows, cols))
    kh, kw = psf.shape
    
    # Place PSF in center
    # Assuming odd kernel size
    r_start = rows//2 - kh//2
    c_start = cols//2 - kw//2
    psf_padded[r_start : r_start+kh, c_start : c_start+kw] = psf
    
    # Shift PSF so center is at (0,0) for FFT
    psf_shifted = np.fft.ifftshift(psf_padded)
    
    # FFT
    H = np.fft.fft2(psf_shifted)
    G = np.fft.fft2(degraded_img)
    
    # Inverse Filtering: F = G / H
    # Avoid division by zero or small numbers
    # Simple thresholding
    H_abs = np.abs(H)
    H_safe = H.copy()
    H_safe[H_abs < epsilon] = epsilon # Avoid division by zero
    
    F_hat = G / H_safe
    
    # Inverse FFT
    f_hat = np.real(np.fft.ifft2(F_hat))
    f_hat = np.clip(f_hat, 0, 1)
    return f_hat

restored = inverse_filter(degraded, psf, epsilon=0.1) # High epsilon because direct inverse is unstable

# Convert back to uint8 for display
degraded_uint8 = (degraded * 255).astype(np.uint8)
restored_uint8 = (restored * 255).astype(np.uint8)

results = [
    ('Original', img),
    ('PSF', psf),
    ('Degraded (Blur+Noise)', degraded_uint8),
    ('Restored (Inverse)', restored_uint8)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    if title == 'PSF':
        plt.imshow(result, cmap='gray')
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()