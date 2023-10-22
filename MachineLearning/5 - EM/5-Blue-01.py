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
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, y = data[:, 0], data[:, 1]
    means = np.array([[-1.25, 1], [1.25, -0.75]])
    variances = np.array([[[0.25, 0], [0, 0.25]], [[0.25, 0], [0, 0.25]]])
    gnks = np.empty((272, 3))
    mix_coef = .5
    for i, (xi, yi) in enumerate(zip(x, y)):
        for j, (mi, si) in enumerate(zip(means, variances)):
            gnks[i][j] = mix_coef * gaussian(xi, yi, mi, si)
        gnks[i][2] = 0
    row_sums = gnks.sum(axis=1)
    gnks = gnks / row_sums[:, np.newaxis]
    plt.scatter(x, y, c=gnks)
    for m, v in zip(means, variances):
        xyz = np.empty(shape=(3, 100, 100))
        xyz[0], xyz[1] = np.meshgrid(np.linspace(-2.0, 2.0, 100), np.linspace(-2.0, 2.0, 100))
        for idx1, (xi, yi) in enumerate(zip(xyz[0], xyz[1])):
            for idx2, (xii, yii) in enumerate(zip(xi, yi)):
                xyz[2][idx1][idx2] = gaussian(xii, yii, m, v)
        plt.contour(xyz[0], xyz[1], xyz[2], [.5])
    plt.show()


if __name__ == "__main__":
    main()
