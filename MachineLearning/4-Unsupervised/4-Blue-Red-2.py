import random
from math import inf

import matplotlib.pyplot as plt


def calc_dist(pix, r, c, m):
    return sum((pix[r, c][i] - m[i]) ** 2 for i in range(3))


def main():
    k = 6
    data = []
    day = []
    with open("t365.txt") as file:
        while line := file.readline().rstrip():
            if line == 'QuantityMagnitude[Missing["NotAvailable"]]':
                day.append(day[-1])
            else:
                day.append(int(line))
            if len(day) == 144:
                data.append(sum(day[66:90]) / 24)
                day = []
    means = [data[random.randint(0, 365)] for _ in range(k)]
    prev_means = []
    ctr = 0
    while prev_means != means:
        prev_means = means
        clusters = [[] for _ in range(k)]
        for di in data:
            bdist = inf
            bidx = 0
            for idx, mi in enumerate(means):
                dist = abs(mi - di)
                if dist < bdist:
                    bidx = idx
                    bdist = dist
            clusters[bidx].append(di)
        means = [sum(c) / len(c) for idx, c in enumerate(clusters)]
        ctr += 1
    print(means)
    print(ctr)
    for c in clusters:
        plt.plot(c, '.')
    plt.show()


if __name__ == '__main__':
    main()
