import pywt
import cv2 as cv
import matplotlib.pyplot as plt

# import image
image = cv.imread('fusarium-patogen.png', cv.IMREAD_GRAYSCALE)

# do threshold
otsus, threshold = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)

# masking
image = cv.bitwise_and(image, image, mask=threshold)

# dwt
LL, (LH, HL, HH) = pywt.dwt2(image, 'db1')

titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("fusarium.png", dpi=300)
