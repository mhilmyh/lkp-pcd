import numpy as np
import os
import cv2
from sklearn.cluster import KMeans

if __name__ == "__main__":

    path_file = "image-soalno-4.bmp"
    image = cv2.imread(path_file)

    print("Dimensi gambar :", image.shape)
    print("Ukuran gambar asli : {} kb".format(
        os.stat(path_file).st_size / 1024))

    original = np.array(image, dtype=np.float64).reshape((-1, 1))

    kmeans = KMeans(n_clusters=16, init='random')
    kmeans.fit(original)
    centers = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    compressed = []
    for label in labels:
        compressed.append(centers[label])

    compressed = np.asarray(compressed, dtype=np.uint8).reshape(image.shape)

    psnr = cv2.PSNR(image, compressed)
    print("Nilai Peak Signal to Noise Ratio :", psnr)
    cv2.imshow("Image", image)
    cv2.imshow("Compressed", compressed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
