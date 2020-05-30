import cv2
import numpy as np


# Subtract Image
def subtract_image(source, subtractor):
    row, col, ch = source.shape
    print("Channel : {}".format(ch))
    canvas = np.zeros((row, col, 3), np.uint8)
    for i in range(row):
        for j in range(col):
            if subtractor[i, j] > 0:
                canvas[i, j] = source[i, j]

    return canvas


# Low Pass Filter
def low_pass_filter(image):
    row, col, ch = image.shape
    canvas = np.zeros((row, col, 1), np.uint8)
    for i in range(row):
        for j in range(col):
            if (1 <= i < row - 1) and (1 <= j < col - 1):
                value = [image[i - 1, j - 1] * 1 / 9, image[i - 1, j] * 1 / 9, image[i - 1, j + 1] * 1 / 9,
                         image[i, j - 1] * 1 / 9, image[i, j] * 1 / 9, image[i, j + 1] * 1 / 9,
                         image[i + 1, j - 1] * 1 / 9, image[i + 1, j] * 1 / 9, image[i + 1, j + 1] * 1 / 9]
                canvas[i, j] = sum(value)
            else:
                canvas[i, j] = image[i, j]
    return canvas


# Median Filter
def median_filter(image, size=3):
    row, col, ch = image.shape
    mid = int(size * size / 2)
    canvas = np.zeros((row, col, 1), np.uint8)
    for i in range(row):
        for j in range(col):
            if size <= i < row - size and size <= j < col - size:
                pixel_list = []
                for a in range(size):
                    for b in range(size):
                        pixel_list.append(image[i + a - 1, j + b - 1])

                pixel_list = sorted(pixel_list)
                canvas[i, j] = pixel_list[mid]
            else:
                canvas[i, j] = image[i, j]

    return canvas


# Threshold to Red
def do_threshold_white(image):
    row, col, ch = image.shape
    canvas = np.zeros((row, col, 1), np.uint8)

    for i in range(row):
        for j in range(col):
            if image[i, j] < 50:
                canvas[i, j] = 0
            else :
                canvas[i, j] = 255

    return canvas


# Threshold to Red
def do_threshold_red(image):
    row, col, ch = image.shape
    canvas = np.zeros((row, col, 1), np.uint8)

    for i in range(row):
        for j in range(col):
            h, s, v = image[i, j]
            if (0 < h < 40) and (30 < s < 160) and (150 < v < 255):
                canvas[i, j] = 255

    return canvas


# RGB to HSV
def rgb_to_hsv(image):
    row, col, ch = image.shape
    canvas = np.zeros((row, col, ch), np.uint8)
    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            b, g, r = b / 255, g / 255, r / 255
            cmax = max(b, g, r)
            cmin = min(b, g, r)
            delta = cmax - cmin
            h = 0
            if delta == 0:
                h = 0
            elif r == cmax:
                h = 60 * (((g - b) / delta) % 6)
            elif g == cmax:
                h = 60 * (((b - r) / delta) + 2)
            elif b == cmax:
                h = 60 * (((r - g) / delta) + 4)

            s = 0 if cmax == 0 else delta / cmax
            v = cmax
            canvas[i, j] = h, s * 255, v * 255

    return canvas


def main():
    image = cv2.imread('FACE DETECTION.png')
    image_hsv = rgb_to_hsv(image)
    image_threshold_red = do_threshold_red(image_hsv)
    image_median = median_filter(image_threshold_red)
    image_low_pass = low_pass_filter(image_median)
    image_threshold_white = do_threshold_white(image_low_pass)
    image_filtered = median_filter(image_threshold_white)
    image_result = subtract_image(image, image_filtered)

    cv2.imwrite("rgb_to_hsv.png", image_hsv)
    cv2.imwrite("threshold_red.png", image_threshold_red)
    cv2.imwrite("median_filter1.png", image_median)
    cv2.imwrite("low_pass_filter.png", image_low_pass)
    cv2.imwrite("threshold_white.png", image_threshold_white)
    cv2.imwrite("median_filter2.png", image_filtered)
    cv2.imwrite("hasil.png", image_result)


if __name__ == '__main__':
    main()
