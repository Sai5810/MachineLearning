from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def main():
    data = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, y = data[:, 0], data[:, 1]
    red, black = [], []
    for xi, yi in zip(x, y):
        d1 = (xi + 1) ** 2 + (yi - 1) ** 2
        d2 = (xi - 1) ** 2 + (yi + 1) ** 2
        if d1 > d2:
            red.append([xi, yi])
        else:
            black.append([xi, yi])
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.plot(*zip(*red), '.', color='red')
    plt.plot(*zip(*black), '.', color='black')
    print(f'(1, -1): {red}')
    print(f'(-1, 1): {black}')
    plt.show()


if __name__ == "__main__":
    main()
