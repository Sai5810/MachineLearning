import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl


def gaussian(x1, x2, m, s):
    dx = (np.array([x1, x2]) - m).reshape(-1, 1)
    sinv = np.linalg.inv(s)
    p = m.size
    sf = np.power(2.0 * np.pi, p / 2.0) * np.sqrt(np.linalg.det(s))
    return 1.0 / sf * np.exp(-0.5 * ((dx.T @ sinv) @ dx))


def main():
    data = np.loadtxt('mouse.csv', delimiter=',', skiprows=1)[1:]
    data = np.delete(data, 0, 1)
    n, m = len(data), len(data[0])
    k = 3
    means = np.array([[-1.5, 1.5], [.5, 5], [1.5, 1.5]])
    variances = np.array([[[0.25, 0], [0, 0.25]]] * k)
    gnks = np.empty((n, k))
    mixers = np.array([.5] * 3)
    for i, (xi, yi) in enumerate(data):
        for j, (mi, si, pi) in enumerate(zip(means, variances, mixers)):
            gnks[i][j] = pi * gaussian(xi, yi, mi, si)
    gnks = gnks / gnks.sum(axis=1)[:, np.newaxis]
    # plt.scatter(data[:, 0], data[:, 1], c=gnks)
    for _ in range(2):
        for i, _ in enumerate(means):
            means[i] = np.average(data, weights=gnks[:, i], axis=0)
        for i, mean in enumerate(means):
            variances[i] = np.array([[0, 0], [0, 0]])
            for j, (x, y) in enumerate(data):
                dx = x - mean[0]
                dy = y - mean[1]
                gamma = gnks[j][i] * np.array([[dx ** 2, dx * dy], [dx * dy, dy ** 2]])
                variances[i] += gamma
        variances /= n
        for i, _ in enumerate(means):
            mixers[i] = np.mean(gnks[:, i])
        for i, (xi, yi) in enumerate(data):
            for j, (mi, si, pi) in enumerate(zip(means, variances, mixers)):
                gnks[i][j] = pi * gaussian(xi, yi, mi, si)
        gnks = gnks / gnks.sum(axis=1)[:, np.newaxis]
    plt.scatter(data[:, 0], data[:, 1], c=gnks)
    print(means)
    print(variances)
    plt.show()


if __name__ == "__main__":
    main()
