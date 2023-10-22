import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def reg(x, y):
    q, r = np.linalg.qr(x)
    qty = q.T @ y
    bs = []
    rcol = r.shape[0]
    for i in range(rcol - 1, -1, -1):
        bs.append((qty[i] + sum(-b * ri for b, ri in zip(bs, r[i][::-1]))) / r[i, i])
    return bs[::-1]


def findRSS(x, y, b):
    return sum((yi - sum(xj * bi for xj, bi in zip(xi, b))) ** 2 for xi, yi in zip(x, y))


def findTSS(y):
    return sum((yi - np.average(y)) ** 2 for yi in y)


def findStats(x, y, n):
    b = reg(x, y)
    rss = findRSS(x, y, b)
    input_n = x.shape[1] - 1
    rse = math.sqrt(rss / (n - input_n - 1))
    se = [rse / math.sqrt(findTSS(x[:, i])) for i in range(0, 3)]
    t = [bi / sei for bi, sei in zip(b, se)]
    p = [stats.t.sf(abs(ti), df=n - input_n - 1) * 2.0 for ti in t]
    ci = [(bi - 2 * sei, bi + 2 * sei) for bi, sei in zip(b, se)]
    return {'b': b, 'rss': rss, 'r^2': 1 - rss / findTSS(y), 'rse': rse, 'se': se, 't': t, 'p': p, 'CI': ci}


def main():
    data = np.loadtxt('Advertising.csv', delimiter=',', skiprows=1)
    n = data.shape[0]
    y = data[:, 4]
    for i in range(1, 4):
        x = np.ones((n, 2))
        x[:, 0] = data[:, i]
        b = reg(x, y)
        print(f"Variable {i} Model\nb: {b}\nrss: {findRSS(x, y, b)}")
    x = np.ones((n, 4))
    x[:, 0:3] = data[:, 1:4]
    stat_dict = findStats(x, y, n)
    print(f"3-1 Model")
    for k, v in stat_dict.items():
        print(f'{k}: {v}')
    x = np.ones((n, 3))
    x[:, 0:2] = data[:, 1:3]
    b = reg(x, y)
    rss = findRSS(x, y, b)
    print(f"2-1 Model\nb: {b}\nrss: {rss}\nr^2: {1 - rss / findTSS(y)}")
    x = np.ones((n, 4))
    x[:, 0:2] = data[:, 1:3]
    x[:, 2] = x[:, 0] * x[:, 1]
    stat_dict = findStats(x, y, n)
    print(f"X1, X2, X1*X2 Model")
    for k, v in stat_dict.items():
        print(f'{k}: {v}')


if __name__ == "__main__":
    main()
