import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main():
    data = np.loadtxt('data.csv', delimiter=',', skiprows=0)
    d1 = np.array([i for i in data if i[2] == 0])
    d2 = np.array([i for i in data if i[2] == 1])
    plt.plot(d1[:, 0], d1[:, 1], '.')
    plt.plot(d2[:, 0], d2[:, 1], '*')
    plt.show()


if __name__ == "__main__":
    main()
