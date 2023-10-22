import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    data = np.genfromtxt('Default.csv', dtype='str', delimiter=',', skip_header=1)
    data[data == '"No"'] = "0"
    data[data == '"Yes"'] = "1"
    data = data.astype('float')
    y = data[:, 0]
    x = data[:, 2]
    x = x.reshape(-1, 1)
    clf = LogisticRegression(random_state=0).fit(x, y)
    print(clf.score(x, y))
    print(clf.intercept_, clf.coef_)


if __name__ == "__main__":
    main()
