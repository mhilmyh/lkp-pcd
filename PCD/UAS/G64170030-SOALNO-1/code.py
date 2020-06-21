import cv2 as cv

image = cv.imread("image-soalno-1.jpeg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(gray, 255//2, 255, cv.THRESH_BINARY)

circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, 1,
                          image.shape[0] // 8, param1=100, param2=20, minRadius=0, maxRadius=40)

if circles is not None:
    for (x, y, r) in circles[0, :]:
        cv.circle(image, (x, y), r, (255, 0, 0), 2)
        cv.circle(image, (x, y), 2, (0, 255, 0), 2)
        print("(x = {}, y = {}) -> r = {}".format(x, y, r))

cv.imshow("Result", image)
cv.imwrite("result.png", image)
cv.waitKey(0)
cv.destroyAllWindows()
