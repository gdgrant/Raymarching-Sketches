import numpy as np
from numba import jit, njit, float32, prange, literal_unroll
from numba.experimental import jitclass
from PIL import Image

from parameters import par
from math_utils import *	# Particularly: vec3, norm


### The goal here is that each function contributes to an overall SDF.
### The functions each produce an object such that at the end, once
### all adjustments are made to the entire scene, that a function
### called something like "build_sdf" is run on the provided outputs,
### and the sdf 'function' itself is built and provided to the renderer



##################
### Primitives ###
##################


def sphere():

	@njit
	def sdf(p):
		return norm(p) - 1.

	return sdf

def cube():

	@njit('float32(float32[:])')
	def sdf(p):
		return norm(np.maximum(np.abs(p) - 1., 0.))

	return sdf



#######################
### Transformations ###
#######################


def translate(d, sdf1):

	@njit
	def sdf(p):
		return sdf1(p - d)

	return sdf

def scale(s, sdf1):

	@njit
	def sdf(p):
		return s * sdf1(p/s)

	return sdf

def scale_aniso(s, sdf1):

	@njit
	def sdf(p):
		return min(s)*sdf1(p/s)

	return sdf


def repeat(c, sdf1):

	@njit
	def sdf(p):
		q = np.mod(p+0.5*c, c) - 0.5*c
		return sdf1(q)

	return sdf


def op_round(r, sdf1):

	@njit
	def sdf(p):
		return sdf1(p) - r

	return sdf

def mirror_x(sdf1):

	@njit
	def sdf(p):
		q = 1.*p
		q[0] = np.abs(p[0])
		return sdf1(q)

	return sdf

def mirror_y(sdf1):

	@njit
	def sdf(p):
		q = 1.*p
		q[1] = np.abs(p[1])
		return sdf1(q)

	return sdf

def mirror_z(sdf1):

	@njit
	def sdf(p):
		q = 1.*p
		q[2] = np.abs(p[2])
		return sdf1(q)

	return sdf

def rotate_x(a, sdf1):

	r = np.array([ \
		[1, 0, 0],
		[0, np.cos(a), -np.sin(a)],
		[0, np.sin(a), np.cos(a)]], dtype=np.float32)

	rinv = np.transpose(r)

	# q = rinv @ vec3(1., 2., 3.)
	# print(q)
	# print(sdf1(q))
	# print(q.dtype)

	# q = np.dot(rinv, vec3(1., 2., 3.))
	# print(q)
	# print(sdf1(q))
	# print(q.dtype)
	# quit()

	@njit('float32(float32[:])')
	def sdf(p):
		q = np.dot(rinv, p)
		return sdf1(q)

	return sdf


#########################
### Binary Operations ###
#########################

def union(sdf1, sdf2):

	@njit
	def sdf(p):
		return np.minimum(sdf1(p), sdf2(p))

	return sdf

def intersect(sdf1, sdf2):

	@njit
	def sdf(p):
		return np.maximum(sdf1(p), sdf2(p))

	return sdf

def subtract(sdf1, sdf2):

	@njit
	def sdf(p):
		return np.maximum(sdf1(p), -sdf2(p))

	return sdf



#########################
### Composite Objects ###
#########################

def infinite_cross():

	b1 = scale_aniso(vec3(1,1,np.inf), cube())
	b2 = scale_aniso(vec3(1,np.inf,1), cube())
	b3 = scale_aniso(vec3(np.inf,1,1), cube())

	cross = union(b1, union(b2, b3))

	@njit
	def sdf(p):
		return cross(p)

	return sdf


def menger(n=3):

	box = cube()
	cross = infinite_cross()

	@njit
	def sdf(p):

		d = box(p)

		s = 1.
		for i in range(n):
			a = np.mod(p * s, 2.) - 1.
			s *= 3.
			r = 1. - 3*np.abs(a)

			c = cross(r)/s
			d = np.maximum(d, c)
			
		return d

	return sdf