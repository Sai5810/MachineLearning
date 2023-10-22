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
    data = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
    n, m = len(data), len(data[0])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    means = np.array([[-1.25, 1], [1.25, -0.75]])
    variances = np.array([[[0.25, 0], [0, 0.25]], [[0.25, 0], [0, 0.25]]])
    gnks = np.empty((n, m))
    mixers = np.array([.5, .5])
    for i, (xi, yi) in enumerate(data):
        for j, (mi, si, pi) in enumerate(zip(means, variances, mixers)):
            gnks[i][j] = pi * gaussian(xi, yi, mi, si)
    gnks = gnks / gnks.sum(axis=1)[:, np.newaxis]
    colors = np.append(gnks, np.zeros([len(gnks), 1]), 1)
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    for _ in range(1000):
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

    for m, v in zip(means, variances):
        xyz = np.empty(shape=(3, 100, 100))
        xyz[0], xyz[1] = np.meshgrid(np.linspace(-2.0, 2.0, 100), np.linspace(-2.0, 2.0, 100))
        for idx1, (xi, yi) in enumerate(zip(xyz[0], xyz[1])):
            for idx2, (xii, yii) in enumerate(zip(xi, yi)):
                xyz[2][idx1][idx2] = gaussian(xii, yii, m, v)
        plt.contour(xyz[0], xyz[1], xyz[2], [.5])
    print(means)
    print(variances)
    plt.show()


if __name__ == "__main__":
    main()
