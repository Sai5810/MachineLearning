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
    x = data[:, 1:]
    n = x.size
    clf = LinearDiscriminantAnalysis().fit(x, y)
    pred = clf.predict(x)
    cm = metrics.confusion_matrix(y, pred)
    print(f"cm: {cm}")
    threshold = 0.2
    y_pred = (clf.predict_proba(x)[:, 1] > threshold).astype('float')
    cm = metrics.confusion_matrix(y, y_pred)
    print(f"cm: {cm}")


if __name__ == "__main__":
    main()
