import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics


def main():
    data = np.genfromtxt('Default.csv', dtype='str', delimiter=',', skip_header=1)
    data[data == '"No"'] = "0"
    data[data == '"Yes"'] = "1"
    data = data.astype('float')
    y = data[:, 0]
    x = data[:, 3].reshape(-1, 1)
    n = x.size
    clf = LinearDiscriminantAnalysis().fit(x, y)
    m = clf.means_
    s2 = 1 / (n - 2) * (sum((xi - m[1]) ** 2 for xi, yi in zip(x, y) if yi == 1) + sum(
        (xi - m[0]) ** 2 for xi, yi in zip(x, y) if yi == 0))
    print(f'm: {m}')
    print(f's2: {s2}')
    pred = clf.predict(x)
    cm = metrics.confusion_matrix(y, pred)
    gx = []
    gy = []
    for xi, yi, pi in zip(x, y, pred):
        if yi != pi:
            gx.append(xi)
            gy.append(pi)
    gx, gy = zip(*sorted(zip(gx, gy)))
    plt.plot(gx, gy, '.')
    plt.plot(gx, gy, '-')
    plt.show()


if __name__ == "__main__":
    main()
