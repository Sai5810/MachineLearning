import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    data = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x, y = data[:, 0], data[:, 1]
    cmap = mpl.colormaps['Greens']
    cval = cmap(0.5, bytes=True)
    val = [random.choice(cval) for _ in x]
    plt.scatter(x, y, c=val, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    main()
