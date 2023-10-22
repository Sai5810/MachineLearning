from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def color(x, y, c1, c2):
    red, black = [], []
    for xi, yi in zip(x, y):
        d1 = (xi - c1[0]) ** 2 + (yi - c1[1]) ** 2
        d2 = (xi - c2[0]) ** 2 + (yi - c2[1]) ** 2
        if d1 > d2:
            red.append([xi, yi])
        else:
            black.append([xi, yi])
    return np.array(red), np.array(black)


def center(arr):
    return [sum(arr[:, i]) / len(arr[:, i]) for i in range(arr.shape[1])]


def main():
    data = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, y = data[:, 0], data[:, 1]
    centers_old = [[0, 0], [0, 0]]
    centers = [[1, -1], [-1, 1]]
    while sorted(centers) != sorted(centers_old):
        print(centers)
        red, black = color(x, y, centers[0], centers[1])
        centers_old = centers.copy()
        centers[0], centers[1] = center(red), center(black)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.plot(*zip(*red), '.', color='red')
    plt.plot(*zip(*black), '.', color='black')
    plt.show()


if __name__ == "__main__":
    main()
