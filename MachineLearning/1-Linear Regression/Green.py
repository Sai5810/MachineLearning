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
    x = np.ones((n, 2))
    x[:, 0] = data[:, 1]
    b0p, b1p = reg(x, data[:, 0])
    print(f'B1:{b1: .3f}, B1\':{b1p: .3f}, r^2:{b1 * b1p: .3f}')
    M = data[:, 0:2]
    S = np.cov(M, rowvar=False, bias=True)
    print(f'Sxy:{S[0, 1]: .3f}, Sy^2:{S[1, 1]: .3f}')
    np.random.shuffle(data)
    xsmp = data[:50, 0]
    ysmp = data[:50, 1]
    plt.plot(xsmp, ysmp, 'o')
    plt.plot(xsmp, b0 + b1 * xsmp, '-')
    for x, y in zip(xsmp, ysmp):
        plt.plot([x, x], [y, b0 + b1 * x], linestyle='dotted')
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()


if __name__ == "__main__":
    main()
