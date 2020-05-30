import cv2
import numpy as np
import matplotlib.pyplot as plt

tomato = cv2.imread('tomato.jpg')
tomato_revert = cv2.cvtColor(tomato, cv2.COLOR_BGR2RGB)
converted = cv2.cvtColor(tomato_revert, cv2.COLOR_RGB2LUV)

gray = converted[:, :, 1]

_, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)

dilasi = cv2.dilate(threshold, kernel, iterations=1)
erosi = cv2.erode(dilasi, kernel, iterations=3)
closing = cv2.morphologyEx(erosi, cv2.MORPH_CLOSE, kernel, iterations=1)

mask = cv2.medianBlur(closing, 3)

masked = cv2.bitwise_and(tomato, tomato, mask=mask)
titles = ['tomato', 'converted', 'gray', 'threshold', 'closing', 'mask']
images = [tomato_revert, converted, gray, threshold, closing, mask]

for i in range(len(images)):
    plt.subplot(len(images) / 3 + 1, min(3, len(images)), i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.savefig('result.png', dpi=300)
plt.show()

cv2.imshow('masked', masked)
cv2.imwrite('hasil.png', masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
