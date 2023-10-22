import numpy as np
import matplotlib.pyplot as plt
M1 = np.array([[-1.25, 1]])
M2 = np.array([[1.25, -0.75]])
S1, S2 = 0.25 * np.identity(2), 0.25 * np.identity(2)
P1, P2= 0.5, 0.5
D = np.loadtxt('faithful.csv', delimiter=',', skiprows=1)
D = (D-np.mean(D, axis=0))/np.std(D, axis=0)
rval1, gval1, bval1, rval2, gval2, bval2 = 223, 61, 26, 16, 19, 217
def f1(x1, x2):
   dx = (np.array([x1, x2]) - M1).reshape(-1, 1 )
   sinv = np.linalg.inv(S1)
   p = M1.size
   sf = np.power(2.0*np.pi, p/2.0) * np.sqrt(np.linalg.det(S1))
   return 1.0/sf * np.exp(-0.5 * ((dx.T@sinv)@dx))
def f2(x1, x2):
   dx = (np.array([x1, x2]) - M2).reshape(-1, 1 )
   sinv = np.linalg.inv(S2)
   p = M2.size
   sf = np.power(2.0*np.pi, p/2.0) * np.sqrt(np.linalg.det(S2))
   return 1.0/sf * np.exp(-0.5 * ((dx.T@sinv)@dx))
vf = np.vectorize(f1)
vf2 = np.vectorize(f2)
x1 = np.linspace(-2.0, 2.0, 100)
x2 = np.linspace(-2.0, 2.0, 100)
X1, X2 = np.meshgrid(x2, x1)
fig, ax = plt.subplots()
D = np.hstack((D, np.zeros((D.shape[0], 2))))
for i in range(D.shape[0]):
    D[i, 2] = P1 * f1(D[i, 0], D[i, 1])
    D[i, 3] = P2 * f2(D[i, 0], D[i, 1])
    D[i, 2], D[i, 3] = D[i, 2]/(D[i, 2] + D[i, 3]), D[i, 3]/(D[i, 2] + D[i, 3])
M1, M2 = np.reshape(np.average(D[:, 0:2], weights=D[:, 2], axis=0), (-1, 2)), np.reshape(np.average(D[:, 0:2], weights=D[:, 3], axis=0), (-1, 2))
S1, S2 = np.zeros((2, 2)), np.zeros((2, 2))
for i in range(D.shape[0]):
    S1 += D[i, 2]*(D[i, 0:2]-M1)*(D[i, 0:2]-M1).T
    S2 += D[i, 3]*(D[i, 0:2]-M2)*(D[i, 0:2]-M2).T
S1 /= np.sum(D[:, 2])
S2 /= np.sum(D[:, 3])
P1, P2 = np.sum(D[:, 2])/np.sum(D[:, 2:4]), np.sum(D[:, 3])/np.sum(D[:, 2:4])
for i in range(D.shape[0]):
    rval, gval, bval = int(D[i, 2]*rval1+D[i, 3]*rval2), int(D[i, 2]*gval1+D[i, 3]*gval2), int(D[i, 2]*bval1+D[i, 3]*bval2)
    gval, bval = bval, gval
    mycolor = f'#{rval:02x}{gval:02x}{bval:02x}'
    ax.scatter([D[i, 0]], [D[i, 1]], color=mycolor)
Y = vf(X1, X2)
Y2 = vf2(X1, X2)
ax.contour(X1, X2, Y, levels=1, colors="black")
ax.contour(X1, X2, Y2, levels=1, colors="black")
plt.show()