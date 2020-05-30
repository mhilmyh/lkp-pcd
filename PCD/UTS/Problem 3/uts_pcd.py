import numpy as np
import cv2


def normalize(gambar, width=1):
    row, col, *_ = gambar.shape
    hasil = np.zeros(gambar.shape, np.float32)
    for i in range(row):
        for j in range(col):
            hasil[i, j] = (gambar[i, j] - gambar.min()) / (gambar.max() - gambar.min()) * width

    return hasil


def main():
    flowers = cv2.imread('images/flowers.tif')
    template = cv2.imread('images/flower-template.tif')

    # crop gambar template
    template = template[0:99, 0:99]

    # convert ke grayscale
    flowers_gray = cv2.cvtColor(flowers, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # cari template di gambar flowers_gray dengan cross correlation
    hasil_cross = cv2.matchTemplate(flowers_gray, template_gray, cv2.TM_CCORR_NORMED)
    print(hasil_cross)
    h, w = template_gray.shape

    # cari local maximum
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hasil_cross)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(flowers, top_left, bottom_right, (0, 0, 255), 4)

    cv2.imshow("Matching Correlation", hasil_cross)
    cv2.imwrite("images/hasil_cross_correlation.png", hasil_cross * 255)

    # cari template dengan convolution
    kernel = normalize(template_gray, 100)
    hasil_convo = cv2.filter2D(flowers_gray, -1, np.uint8(kernel))

    cv2.imshow("Template", template)
    cv2.imshow("Matching Convolution", hasil_convo)
    cv2.imwrite("images/hasil_convolution.png", hasil_convo)
    cv2.imshow("Flowers", flowers)
    cv2.imwrite("images/hasil.png", flowers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
