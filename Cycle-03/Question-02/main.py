import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# 1. Create Degradation Function (Motion Blur PSF)
def get_motion_blur_psf(size=15, angle=45):
    psf = np.zeros((size, size))
    center = size // 2
    kernel = np.zeros((size, size))
    kernel[center, :] = 1
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    psf = cv2.warpAffine(kernel, M, (size, size))
    psf = psf / np.sum(psf)
    return psf

psf = get_motion_blur_psf(15, 45)

# 2. Degrade Image
img_float = img.astype(np.float32) / 255.0
blurred = cv2.filter2D(img_float, -1, psf)
noise_var = 0.001
noise = np.random.normal(0, np.sqrt(noise_var), img.shape)
degraded = blurred + noise
degraded = np.clip(degraded, 0, 1)

# Helper: Pad and FFT PSF
def get_H(image_shape, psf):
    rows, cols = image_shape
    psf_padded = np.zeros((rows, cols))
    kh, kw = psf.shape
    r_start = rows//2 - kh//2
    c_start = cols//2 - kw//2
    psf_padded[r_start : r_start+kh, c_start : c_start+kw] = psf
    psf_shifted = np.fft.ifftshift(psf_padded)
    H = np.fft.fft2(psf_shifted)
    return H

# 3. Wiener Filter (Constant Ratio)
def wiener_filter_constant(degraded_img, psf, K=0.01):
    H = get_H(degraded_img.shape, psf)
    G = np.fft.fft2(degraded_img)
    
    # F = (H* / (|H|^2 + K)) * G
    H_conj = np.conj(H)
    H_mag2 = np.abs(H)**2
    
    W = H_conj / (H_mag2 + K)
    F_hat = W * G
    
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 1)

# 4. Wiener Filter (Auto Correlation / Power Spectrum)
def wiener_filter_autocorr(degraded_img, psf, original_img, noise_var):
    H = get_H(degraded_img.shape, psf)
    G = np.fft.fft2(degraded_img)
    
    # Estimate Power Spectra
    # S_xx = |DFT(f)|^2
    # S_nn = |DFT(n)|^2 ~ rows*cols * noise_var (White noise approximation)
    
    F_orig = np.fft.fft2(original_img.astype(np.float32)/255.0)
    S_xx = np.abs(F_orig)**2
    
    rows, cols = degraded_img.shape
    S_nn = rows * cols * noise_var
    
    # NSR = S_nn / S_xx
    # Handle division by zero in S_xx
    S_xx[S_xx < 1e-6] = 1e-6
    NSR = S_nn / S_xx
    
    H_conj = np.conj(H)
    H_mag2 = np.abs(H)**2
    
    W = H_conj / (H_mag2 + NSR)
    F_hat = W * G
    
    f_hat = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_hat, 0, 1)

# Apply Filters
res_const = wiener_filter_constant(degraded, psf, K=0.01)
res_auto = wiener_filter_autocorr(degraded, psf, img, noise_var)

# Convert to uint8
degraded_u8 = (degraded * 255).astype(np.uint8)
res_const_u8 = (res_const * 255).astype(np.uint8)
res_auto_u8 = (res_auto * 255).astype(np.uint8)

results = [
    ('Original', img),
    ('Degraded', degraded_u8),
    ('Wiener (K=0.01)', res_const_u8),
    ('Wiener (AutoCorr)', res_auto_u8)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()