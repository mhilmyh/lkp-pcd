import cv2
import numpy as np


class Kernel:
    def __init__(self, size, matrix=np.zeros((3, 3), np.uint8)):
        self.size = size
        self.border = int(self.size / 2)
        self.matrix = matrix
        self.total = self.matrix.sum()
        self.normalize = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if self.total > 0:
                    self.normalize[i, j] = self.matrix[i, j] / self.total
                else:
                    self.normalize[i, j] = self.matrix[i, j]

    def set_size(self, value):
        self.size = value
        self.border = int(self.size / 2)

    def get_size(self):
        return self.size

    def set_matrix(self, value):
        self.matrix = value
        self.total = self.matrix.sum()
        for i in range(self.size):
            for j in range(self.size):
                self.normalize[i, j] = self.matrix[i, j] / self.total

    def get_matrix(self):
        return self.matrix

    def find_median(self, data, x, y):
        result = np.array([])
        for i in range(self.size):
            for j in range(self.size):
                result = np.append(result, data[(x + i - self.border), (y + j - self.border)])
        n = self.size * self.size
        result = sorted(result)
        return int(result[int(n / 2) + 1]) if n % 2 == 0 else int((result[int(n / 2)] + result[int(n / 2) + 1]) / 2)

    def median_filter(self, image):
        row, col = image.shape
        canvas = np.zeros((row, col), np.uint8)
        for i in range(row):
            for j in range(col):
                if (self.border < i < row - self.border) and (self.border < j < col - self.border):
                    canvas[i, j] = self.find_median(image, i, j)
                else:
                    canvas[i, j] = image[i, j]
        return canvas

    def low_pass_calculation(self, data, x, y):
        result = np.array([], np.int32)
        for i in range(self.size):
            for j in range(self.size):
                result = np.append(result, data[(x + i - self.border), (y + j - self.border)] * self.normalize[i, j])
        result = result.astype('uint8')
        return sum(result)

    def low_pass_filter(self, image):
        row, col = image.shape
        canvas = np.zeros((row, col), np.uint8)
        for i in range(row):
            for j in range(col):
                if (self.border < i < row - self.border) and (self.border < j < col - self.border):
                    canvas[i, j] = self.low_pass_calculation(image, i, j)
                else:
                    canvas[i, j] = image[i, j]
        return canvas


def main():
    image_raw = cv2.imread('LennaInput.png', 0)
    kernel = Kernel(size=3, matrix=np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]))
    image_low_pass_filter = kernel.low_pass_filter(image=image_raw)
    image_median_filter = kernel.median_filter(image=image_raw)
    cv2.imwrite('LennaOutput_median.png', image_median_filter)
    cv2.imwrite('LennaOutput_low_pass.png', image_low_pass_filter)


if __name__ == '__main__':
    main()
