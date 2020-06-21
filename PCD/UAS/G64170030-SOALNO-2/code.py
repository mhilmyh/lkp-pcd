from math import cos, sin, radians
from scipy import interpolate
import matplotlib.pyplot as plt
from numpy import linspace


def calculateRadius(x, y, t):
    n = len(min(x, y))
    print("x :", x)
    print("y :", y)
    print("t :", t)
    print("n :", n)
    print("x\t\ty\t\t-45\t\t0\t\t45\t\t90")
    print("="*100)
    ret = []
    for i in range(n):
        tmp = [x[i], y[i]]
        for j in t:
            r = x[i] * cos(radians(j)) + y[i] * sin(radians(j))
            tmp.append(round(r, 3))

        ret.append(tmp)
        print("\t\t".join(str(s) for s in tmp))

    return ret


def plotHoughTransform(data, t, smooth=True):
    _, ax = plt.subplots()
    for datum in data:
        x, y, *th = datum
        if smooth:
            new_t = linspace(t[0], t[-1], 100)
            a_spline = interpolate.make_interp_spline(t, th)
            new_th = a_spline(new_t)
            plt.plot(new_t, new_th, label="(" + str(x) + "," + str(y) + ")")
        else:
            plt.plot(t, th, label="(" + str(x) + "," + str(y) + ")")

    ax.set(xlabel="theta", ylabel="r value", title="HOUGH LINE CURVE")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == "__main__":

    x = [1, 2, 4, 3, 2, 1, 4, 2, 2]
    y = [1, 1, 1, 2, 3, 4, 4, 5, 6]
    theta = [-45, 0, 45, 90]

    radius = calculateRadius(x, y, theta)
    plotHoughTransform(radius, theta, False)
