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


def find_tss(y):
    return sum((yi - np.average(y)) ** 2 for yi in y)


def find_stats(x, y, n):
    b = reg(x, y)
    rss = sum((yi - sum(xj * bi for xj, bi in zip(xi, b))) ** 2 for xi, yi in zip(x, y))
    input_n = x.shape[1] - 1
    rse = math.sqrt(rss / (n - input_n - 1))
    se = [rse / math.sqrt(find_tss(x[:, i])) for i in range(0, input_n)]
    t = [bi / sei for bi, sei in zip(b, se)]
    p = [stats.t.sf(abs(ti), df=n - input_n - 1) * 2.0 for ti in t]
    ci = [(bi - 2 * sei, bi + 2 * sei) for bi, sei in zip(b, se)]
    return {'b': b, 'rss': rss, 'r^2': 1 - rss / find_tss(y), 'rse': rse, 'se': se, 't': t, 'p': p, 'CI': ci}


def regularize(x, y, lam, terms):
    stack = lam * np.eye(terms)
    stack[terms - 1][terms - 1] = 0
    return np.vstack((x, stack)), np.concatenate((y, [0] * terms))


def main():
    data = np.loadtxt('test3.csv', delimiter=',', skiprows=0)
    n = data.shape[0]
    y = data[:, -1]
    terms = 1
    x = np.ones((n, terms))
    xcol = data[:, 1]
    for i in range(terms - 1):
        x[:, i] = xcol ** (terms - i - 1)
    # x, yreg = regularize(x, y, .001, terms)
    # stat_dict = find_stats(x, yreg, n)
    stat_dict = find_stats(x, y, n)
    for k, v in stat_dict.items():
        print(f'{k}: {v}')
    plt.plot(xcol, y, 'o')
    xlin = np.linspace(np.min(xcol), np.max(xcol))
    ylin = []
    for xi in xlin:
        ylin.append(sum(bi * xi ** ti for ti, bi in zip(range(terms - 1, -1, -1), stat_dict["b"])))
    # plt.ylim(-10, 10)
    plt.plot(xlin, ylin, 'g')
    plt.show()


if __name__ == "__main__":
    main()
