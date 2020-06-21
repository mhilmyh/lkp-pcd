import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt
from skimage import img_as_float
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet, estimate_sigma


def distance(x, y):
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def gaussian_low_pass(size):
    rows, cols, *_ = size
    mask = np.zeros(size, dtype=np.uint8)
    center = (rows // 2, cols // 2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = exp(((-distance((y, x), center)**2)/(2*(50**2))))
    return mask


def generate_mask(size):
    row, col, *_ = size
    crow, ccol = row // 2, col // 2
    mask = np.zeros(size, dtype=np.uint8)
    r = 30
    x, y = np.ogrid[:row, :col]
    filter_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r ** 2
    mask[filter_area] = 1
    return mask


def fourier_transform(image):
    image_fft = np.fft.fft2(image.astype("float32"))
    image_fshifted = np.fft.fftshift(image_fft)
    mask = generate_mask(image.shape)
    image_fshifted = image_fshifted * mask
    image_ishifted = np.fft.ifftshift(image_fshifted)
    image_back = np.fft.ifft2(image_ishifted)

    result = [np.log(1+np.abs(image_fft)), np.log(1+np.abs(image_fshifted)),
              np.log(1+np.abs(image_ishifted)), np.abs(image_back)]
    return result


def do_fourier(image):
    image_r = fourier_transform(image[:, :, 0])
    image_g = fourier_transform(image[:, :, 1])
    image_b = fourier_transform(image[:, :, 2])

    plt.subplot(3, 4, 1), plt.imshow(image_r[0])
    plt.subplot(3, 4, 5), plt.imshow(image_g[0])
    plt.subplot(3, 4, 9), plt.imshow(image_b[0])

    plt.subplot(3, 4, 2), plt.imshow(image_r[1])
    plt.subplot(3, 4, 6), plt.imshow(image_g[1])
    plt.subplot(3, 4, 10), plt.imshow(image_b[1])

    plt.subplot(3, 4, 3), plt.imshow(image_r[2])
    plt.subplot(3, 4, 7), plt.imshow(image_g[2])
    plt.subplot(3, 4, 11), plt.imshow(image_b[2])

    plt.subplot(3, 4, 4), plt.imshow(image_r[3])
    plt.subplot(3, 4, 8), plt.imshow(image_g[3])
    plt.subplot(3, 4, 12), plt.imshow(image_b[3])

    plt.show()

    image[:, :, 0] = image_r[3]
    image[:, :, 1] = image_g[3]
    image[:, :, 2] = image_g[3]

    return image


def do_wavelet(image):
    image = img_as_float(image)
    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
    result = denoise_wavelet(image, multichannel=True, convert2ycbcr=True,
                             method='VisuShrink', mode='soft',
                             sigma=sigma_est/4, rescale_sigma=True)
    return result


if __name__ == "__main__":
    plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

    image = cv2.imread("image-soalno-3.jpg")

    result_fourier = do_fourier(image.copy())
    result_wavelet = do_wavelet(image.copy())

    cv2.imshow("Real Image Fourier", image)
    cv2.imshow("Result Fourier", result_fourier)
    cv2.imshow("Result Wavelet", result_wavelet)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
