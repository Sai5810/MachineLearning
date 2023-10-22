import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl


def main():
    data = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, y = data[:, 0], data[:, 1]
    plt.scatter(x, y, c="green")

    x1 = np.linspace(-2.0, 2.0, 100)
    x2 = np.linspace(-2.0, 2.0, 101)
    X1, X2 = np.meshgrid(x2, x1)
    m = np.array([-1, 1])
    s = np.array([[0.01, 0], [0, 0.01]])

    def f(x1, x2):
        dx = (np.array([x1, x2]) - m).reshape(-1, 1)
        sinv = np.linalg.inv(s)
        p = m.size
        sf = np.power(2.0 * np.pi, p / 2.0) * np.sqrt(np.linalg.det(s))
        return 1.0 / sf * np.exp(-0.5 * ((dx.T @ sinv) @ dx))

    vf = np.vectorize(f)
    Y1 = vf(X1, X2)
    m = np.array([1, -1])
    vf = np.vectorize(f)
    Y2 = vf(X1, X2)
    Y = Y1 + Y2
    plt.scatter(X1, X2, c=Y, cmap="Greens")
    plt.show()


if __name__ == "__main__":
    main()
