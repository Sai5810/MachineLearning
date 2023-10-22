import numpy as np
import matplotlib.pyplot as plt


def main():
    x1 = np.linspace(-1.0, 2.0, 100)
    x2 = np.linspace(-1.0, 2.0, 101)
    X1, X2 = np.meshgrid(x2, x1)
    m = np.array([0.4, 0.7])
    s = np.array([[0.05, 0.04], [0.04, 0.04]])

    def f(x1, x2):
        dx = (np.array([x1, x2]) - m).reshape(-1, 1)
        sinv = np.linalg.inv(s)
        p = m.size
        sf = np.power(2.0 * np.pi, p / 2.0) * np.sqrt(np.linalg.det(s))
        return 1.0 / sf * np.exp(-0.5 * ((dx.T @ sinv) @ dx))

    vf = np.vectorize(f)
    Y = vf(X1, X2)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X1, X2, Y)
    plt.show()


if __name__ == "__main__":
    main()
