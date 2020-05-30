import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def generateFilter(blueprint, mode='HIGH', threshold=100):
    row, col, *ch = blueprint.shape
    center = (row // 2, col // 2)
    result = np.zeros(blueprint.shape, np.float32) if mode == 'LOW' else np.ones(blueprint.shape, np.float32)

    for x in range(row):
        for y in range(col):
            if distance((x, y), center) < threshold:
                result[x, y] = 1.0000000000 if mode == 'LOW' else 0.0000000001
            else:
                result[x, y] = 0.0000000001 if mode == 'LOW' else 1.0000000000

    return result


def main():
    image = cv.imread('leaf-spot-disease-control-in-michigan.jpg')

    row, col, *ch = image.shape

    image_red = np.zeros((row, col), np.uint8)
    image_green = np.zeros((row, col), np.uint8)
    image_blue = np.zeros((row, col), np.uint8)

    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            image_red[i, j] = r
            image_green[i, j] = g
            image_blue[i, j] = b

    fourier = np.fft.fft2(image_red)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude_spectrum = 20 * np.log(np.abs(fourier_shift))

    plt.subplot(331)
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Input Image')

    plt.subplot(332)
    plt.imshow(image_red, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Red Channel')

    plt.subplot(333)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Magnitude Spectrum')

    low_pass_filter = generateFilter(fourier_shift, mode='LOW', threshold=25)
    high_pass_filter = generateFilter(fourier_shift, mode='HIGH', threshold=25)

    custom_filter = np.copy(fourier_shift)
    outrange = 50
    inrange = 5
    custom_filter[row // 2 - inrange: row // 2 + inrange, col // 2 - inrange: col // 2 + inrange] = 0.0000000001
    custom_filter[:, : col // 2 - outrange] = 0.0000000001
    custom_filter[:, col // 2 + outrange:] = 0.0000000001
    custom_filter[: row // 2 - outrange, :] = 0.0000000001
    custom_filter[row // 2 + outrange:, :] = 0.0000000001

    magnitude_spectrum_low = 20 * np.log(np.abs(low_pass_filter))
    magnitude_spectrum_high = 20 * np.log(np.abs(high_pass_filter))
    magnitude_spectrum_custom = 20 * np.log(np.abs(custom_filter))

    plt.subplot(334)
    plt.imshow(magnitude_spectrum_low, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Low Pass Filter')

    plt.subplot(335)
    plt.imshow(magnitude_spectrum_high, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('High Pass Filter')

    plt.subplot(336)
    plt.imshow(magnitude_spectrum_custom, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Custom Filter')

    fourier_invers_shift_low = np.fft.ifftshift(fourier_shift * low_pass_filter)
    image_invers_low = np.abs(np.fft.ifft2(fourier_invers_shift_low))

    fourier_invers_shift_high = np.fft.ifftshift(fourier_shift * high_pass_filter)
    image_invers_high = np.abs(np.fft.ifft2(fourier_invers_shift_high))

    fourier_invers_shift_custom = np.fft.ifftshift(custom_filter)
    image_invers_custom = np.abs(np.fft.ifft2(fourier_invers_shift_custom))

    plt.subplot(337)
    plt.imshow(image_invers_low, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Image Low')

    plt.subplot(338)
    plt.imshow(image_invers_high, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Image High')

    plt.subplot(339)
    plt.imshow(image_invers_custom, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Image Custom')

    plt.savefig('example1.png')


if __name__ == '__main__':
    main()
