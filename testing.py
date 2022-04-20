from math_utils import vec3, rotation_matrix
import numpy as np
import matplotlib.pyplot as plt

v = vec3(1,1,1)

xs = []
ys = []
zs = []

thetas = np.linspace(0, 2*np.pi, 1000)

for t in thetas:
	R = rotation_matrix(0,0,t)
	vf = R @ v

	xs.append(vf[0])
	ys.append(vf[1])
	zs.append(vf[2])

plt.plot(thetas, xs, c='r', label='x')
plt.plot(thetas, ys, c='g', label='y')
plt.plot(thetas, zs, c='b', label='z')
plt.axhline(0, c='k', ls='--')
plt.legend()
plt.show()