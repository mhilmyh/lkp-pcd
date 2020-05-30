import numpy as np
import cv2
import matplotlib.pyplot as plt


def rgb2hsv(gambar):
    row, col, _ = gambar.shape
    hasil = np.zeros_like(gambar)

    for i in range(row):
        for j in range(col):
            b, g, r = gambar[i, j]
            r, g, b = r / 255, g / 255, b / 255

            h = 0
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx - mn

            if mx == r:
                h = (60 * ((g - b) / df) + 360) % 360
            elif mx == g:
                h = (60 * ((b - r) / df) + 120) % 360
            elif mx == b:
                h = (60 * ((r - g) / df) + 240) % 360
            if mx == 0:
                s = 0
            else:
                s = df / mx

            v = mx
            h = h * 255 / 360
            s *= 255
            v *= 255

            hasil[i, j] = h, s, v

    return hasil, hasil[:, :, 0], hasil[:, :, 1], hasil[:, :, 2]


def rgb2yuv(gambar):
    row, col, _ = gambar.shape
    hasil = np.zeros_like(gambar)

    for i in range(row):
        for j in range(col):
            b, g, r = gambar[i, j]
            y = min(255, max(0, 0.29900 * r + 0.58700 * g + 0.11400 * b))
            u = min(255, max(0, -0.147108 * r - 0.288804 * g + 0.435912 * b + 127.5))
            v = min(255, max(0, 0.614777 * r - 0.514799 * g - 0.099978 * b + 127.5))
            hasil[i, j] = [y, u, v]

    return hasil, hasil[:, :, 0], hasil[:, :, 1], hasil[:, :, 2]


def main():
    tomat_rgb = cv2.imread('images/tomato_2.jpg')

    # konversi ke color space YUV
    tomat_yuv, *chanel_yuv = rgb2yuv(tomat_rgb)

    # pilih chanel V
    tomat_v = chanel_yuv[-1]
    cv2.imwrite("images/tomato_2_gray.jpg", tomat_v)

    # histogram
    plt.hist(tomat_v.ravel(), 256, [0, 256])
    plt.show()

    # threshold gambar tomat
    threshold = 155
    mask = np.copy(tomat_v)
    mask[tomat_v <= threshold] = False
    mask[tomat_v > threshold] = True

    # masking gambar tomat asli
    tomat_rgb = cv2.bitwise_and(tomat_rgb, tomat_rgb, mask=mask)
    cv2.imshow("Hasil", tomat_rgb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
