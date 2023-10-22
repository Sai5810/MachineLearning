import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from numpy.linalg import inv

def update(m, cov, s2, x, y):
    xt = np.array([1, x])
    cov_old = cov.copy()
    cov_new = inv(inv(cov_old) + (1 / s2) * xt * x)
    m = cov_new * (inv(cov_old) * m + (1 / s2) * xt * y)
    return m, cov_new

def main():
    data = np.loadtxt('xy.txt', delimiter=',', skiprows=1)
    x, y = data[:, 1], data[:, 2]
    s2 = .04
    n = x.size
    m = np.array([0, 0])
    cov = np.array([[.5, 0], [0, .5]])
    m, cov = update(m, cov, s2, x[0], y[0])
    bs = np.random.multivariate_normal(m, cov, 6)
    fig, ax = plt.subplots()
    for b in bs:
        ax.axline((0, b[0]), slope=b[1])
    plt.plot(x[:2], y[:2], '.')
    plt.show()


if __name__ == "__main__":
    main()
