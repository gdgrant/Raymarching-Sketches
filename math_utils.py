import numpy as np
from numba import njit, float32, prange, literal_unroll
from numba.experimental import jitclass
from PIL import Image

from parameters import par


# @jitclass([ ('x', float32), ('y', float32), ('z', float32) ])
# class Vec3:

# 	def __init__(self, x, y, z):
# 		self.x = x
# 		self.y = y
# 		self.z = z

# 	def __repr__(self):
# 		return 'Vec3({},{},{})'.format(self.x, self.y, self.z)

# 	def xy(self):
# 		return self.x, self.y


@njit
def vec3(x, y, z):
	return np.array([x,y,z], dtype=np.float32)

@njit
def norm(p):
	return np.sqrt(np.sum(np.square(p)))

@njit
def unit(x):
	return x / norm(x)

@njit
def dist(p1, p2):
	return norm(p2 - p1)

@njit
def dot(a, b):
	return np.sum(a * b)

@njit
def min_zero(x):
	return np.maximum(x, 0)

@njit
def pos_dot(a, b):
	return min_zero(dot(a, b))

@njit
def clip(x):
	return np.maximum(np.minimum(x, 0), 1)

@njit
def cross(a, b):
	s1 = a[1]*b[2] - a[2]*b[1]
	s2 = a[2]*b[0] - a[0]*b[2]
	s3 = a[0]*b[1] - a[1]*a[0]
	return vec3(s1,s2,s3)

@njit
def rotation_matrix(a=0., b=0., c=0.):
	# a = yaw,   about z
	# b = pitch, about y
	# c = roll,  about x

	ca = np.cos(a)
	cb = np.cos(b)
	cc = np.cos(c)

	sa = np.sin(a)
	sb = np.sin(b)
	sc = np.sin(c)

	r11 = ca * cb
	r12 = ca * sb * sc - sa * cc
	r13 = ca * sb * cc + sa * sc
	r21 = sa * cb
	r22 = sa * sb * sc + ca * cc
	r23 = sa * sb * cc - ca * sc
	r31 = - sb
	r32 = cb * sc
	r33 = cb * cc

	return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], dtype=np.float32)


@njit(parallel=True)
def find_normal(p, sdf, eps=1e-6):

	vnorm = vec3(0.,0.,0.)
	for i in prange(3):

		ax = vec3(0.,0.,0.)
		ax[i] = 1

		vnorm[i] = sdf(p + eps*ax) - sdf(p - eps*ax)

	return unit(vnorm)

@njit
def interpolate(a, b, step):

	inter = []
	num_steps = int(dist(a, b)/step)
	for i in range(num_steps):
		v = float(i) / float(num_steps)
		inter.append((a * (1-v) + b * v).astype(np.float32))

	return inter


def pixels_to_image(pixels, shape_x, shape_y):

	# Convert to image structure
	pixels = np.array(pixels, dtype=np.float32)
	pixels = pixels.reshape([shape_x, shape_y, 3])

	# Clip color limits
	pixels = np.clip(pixels, 0., 1.)

	# Flip y axis
	pixels = np.flip(pixels, axis=1)

	# Transpose to screen direction
	pixels = np.transpose(pixels, [1,0,2])

	# Scale color for image
	img_arr = (255*pixels).astype(np.uint8)

	# Make image object and return
	return Image.fromarray(img_arr)

