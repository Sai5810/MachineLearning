import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def sigmoid(xi, b):
    return math.exp(b[0] * xi + b[1]) / (1 + math.exp(b[0] * xi + b[1]))

def main():
    data = np.genfromtxt('Default.csv', dtype='str', delimiter=',', skip_header=1)
    data[data == '"No"'] = "0"
    data[data == '"Yes"'] = "1"
    data = data.astype('float')
    y = data[:, 0]
    x = data[:, 2]
    x = x.reshape(-1, 1)
    print(f"num of x:{x.size}")
    clf = LogisticRegression(random_state=0).fit(x, y)
    b = [clf.coef_[0][0], clf.intercept_[0]]
    print(f"b:{b}")
    fneg = 0
    fpos = 0
    for xi, yi in zip(x, y):
        ypred = round(sigmoid(xi, b))
        if ypred == 0 and yi == 1:
            fneg += 1
        elif ypred == 1 and yi == 0:
            fpos += 1
    print(f"fneg: {fneg}, fpos: {fpos}")
    plt.plot(x, y, '.')
    xlin = np.linspace(np.min(x), np.max(x))
    print(f"max x:{np.max(x)}")
    ylin = []
    for xi in xlin:
        ylin.append(sigmoid(xi, b))
    plt.plot(xlin, ylin, 'g')
    plt.show()


if __name__ == "__main__":
    main()
