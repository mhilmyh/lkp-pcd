import numpy as np
import cv2


def gaussian_kernel(size, sigma=1):
    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2))) * normal
    return g


def convolution(gambar, kernel):
    heigth, width, *_ = kernel.shape
    h = heigth // 2
    w = width // 2

    gambar = np.pad(gambar, ((h, h), (w, w)), 'mean')
    hasil = np.copy(gambar)
    row, col, *ch = hasil.shape

    for i in range(h, row - h):
        for j in range(w, col - w):
            x = gambar[i - h: i + h + 1, j - w: j + w + 1]
            hasil[i, j] = np.sum(x * kernel)

    return hasil[h:-h, w:-w]


def gaussian_filter(gambar, size=5, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    filtered = convolution(gambar, kernel)
    return filtered


def sobel_filter(gambar):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], np.float32)

    dx = cv2.filter2D(np.float32(gambar), -1, Kx)
    dy = cv2.filter2D(np.float32(gambar), -1, Ky)
    cv2.imwrite("images/ipb_sobel_x.png", dx)
    cv2.imwrite("images/ipb_sobel_y.png", dy)
    g = np.sqrt(dx * dx + dy * dy)
    g = np.float32(g / np.max(g))
    theta = np.arctan2(dy, dx)

    return g, theta


def non_max_suppression(gambar, D):
    M, N = gambar.shape
    Z = np.zeros((M, N), np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gambar[i, j + 1]
                r = gambar[i, j - 1]
            # angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = gambar[i + 1, j - 1]
                r = gambar[i - 1, j + 1]
            # angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = gambar[i + 1, j]
                r = gambar[i - 1, j]
            # angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = gambar[i - 1, j - 1]
                r = gambar[i + 1, j + 1]

            if (gambar[i, j] >= q) and (gambar[i, j] >= r):
                Z[i, j] = gambar[i, j]
            else:
                Z[i, j] = 0

    return Z


def double_threshold(gambar, weak=100, strong=255, lower=0.05, upper=0.09):
    row, col, *_ = gambar.shape
    hasil = np.zeros((row, col), np.uint8)

    upperbound = upper * gambar.max()
    lowerbound = lower * upperbound

    weak = np.uint8(weak)
    strong = np.uint8(strong)

    strong_i, strong_j = np.where(gambar >= upperbound)
    weak_i, weak_j = np.where((gambar <= upperbound) & (gambar >= lowerbound))

    hasil[strong_i, strong_j] = strong
    hasil[weak_i, weak_j] = weak

    return hasil, weak, strong


def hysteresis(gambar, weak=100, strong=255):
    hasil = np.copy(gambar)
    row, col, *_ = hasil.shape
    for i in range(1, row-1):
        for j in range(1, col-1):
            if hasil[i, j] == weak:

                if hasil[i + 1, j + 1] == strong:
                    hasil[i, j] = strong
                elif hasil[i + 1, j] == strong:
                    hasil[i, j] = strong
                elif hasil[i + 1, j - 1] == strong:
                    hasil[i, j] = strong
                elif hasil[i, j + 1] == strong:
                    hasil[i, j] = strong
                elif hasil[i, j] == strong:
                    hasil[i, j] = strong
                elif hasil[i, j - 1] == strong:
                    hasil[i, j] = strong
                elif hasil[i - 1, j + 1] == strong:
                    hasil[i, j] = strong
                elif hasil[i - 1, j] == strong:
                    hasil[i, j] = strong
                elif hasil[i - 1, j - 1] == strong:
                    hasil[i, j] = strong
                else:
                    hasil[i, j] = 0

    return hasil


def main():
    ipb = cv2.imread('images/IPB_Exam2.jpg', 0)

    # noise reduction dengan gaussian
    ipb_blur = gaussian_filter(gambar=ipb, size=11)
    cv2.imshow("Gaussian Filter", ipb_blur)

    # gradient calculation dengan sobel
    ipb_sobel, theta = sobel_filter(ipb_blur)
    cv2.imshow("Sobel", ipb_sobel)
    # cv2.imwrite("images/ipb_sobel.png", np.uint8(ipb_sobel * 255))

    # non max suppression
    ipb_thin_line = non_max_suppression(ipb_sobel, theta)
    cv2.imshow("Non Max Suppresion", ipb_thin_line)
    cv2.imwrite("images/ipb_non_max_suppression.png", np.uint8(ipb_thin_line * 255))

    # double threshold
    ipb_threshold, weak, strong = double_threshold(ipb_thin_line)
    cv2.imshow("Threshold", ipb_threshold)
    # cv2.imwrite("images/ipb_double_threshold.png", ipb_threshold)

    # hysteresis
    ipb_hysteresis = hysteresis(ipb_threshold, weak, strong)
    cv2.imshow("Hysteresis", ipb_hysteresis)
    # cv2.imwrite("images/ipb_hysteresis.png", ipb_hysteresis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
