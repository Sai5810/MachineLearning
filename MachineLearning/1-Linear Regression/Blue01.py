import numpy as np
import matplotlib.pyplot as plt


def reg(x, y):
    q, r = np.linalg.qr(x)
    qty = q.T @ y
    b0 = qty[1] / r[1, 1]
    b1 = (qty[0] - r[0, 1] * b0) / r[0, 0]
    return b0, b1


def main():
    data = np.loadtxt('test.csv', delimiter=',', skiprows=0)
    n = data.shape[0]
    x = np.ones((n, 2))
    x[:, 0] = data[:, 0]
    b0, b1 = reg(x, data[:, 1])
    rss = sum((y - (b1 + b0 * x)) ** 2 for x, y in zip(data[:, 0], data[:, 1]))
    print(b0, b1, rss)
    b0s = np.arange(b0 - 4, b0 + 4, .1)
    b1s = np.arange(b1 - 4, b1 + 4, .1)
    c = []
    for _0 in b0s:
        cs = []
        for _1 in b1s:
            cs.append(sum((y - (_1 + _0 * x)) ** 2 for x, y in zip(data[:, 0], data[:, 1])))
        c.append(cs)
    c = np.array(c)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    a, b = np.meshgrid(b0s, b1s)
    ax.plot_surface(a, b, c)
    ax.contour(a, b, c + 1, 10)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(a, b, c)
    plt.show()


if __name__ == "__main__":
    main()
