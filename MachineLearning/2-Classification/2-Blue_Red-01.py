import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def main():
    data = np.genfromtxt('Default.csv', dtype='str', delimiter=',', skip_header=1)
    data[data == '"No"'] = "0"
    data[data == '"Yes"'] = "1"
    data = data.astype('float')
    y = data[:, 0]
    x = data[:, 1:4]
    x[:, 2] = x[:, 2] / 1000
    print(f"num of x:{x.size}")
    clf = LogisticRegression(random_state=0).fit(x, y)
    pred = clf.predict(x)
    cm = metrics.confusion_matrix(y, pred)
    b = np.append(clf.coef_[0], clf.intercept_[0])
    metrics.RocCurveDisplay.from_estimator(clf, x, y)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print(f"AUC: {metrics.auc(fpr, tpr)}")
    print(f"b:{b}")
    plt.show()


if __name__ == "__main__":
    main()
