import numpy as np
from numba import njit, float32, prange, literal_unroll
from numba.experimental import jitclass
from PIL import Image

from parameters import par
from math_utils import *


def translate(sdf, d):
	# Translate a sdf by some distance d

	@njit
	def translated_sdf(p):
		return sdf(p - d)

	return translated_sdf


def scale(sdf, d):
	# Scale (stretch/compress) a sdf by some amount d

	@njit
	def scaled_sdf(p):
		return sdf(p / d)

	return scaled_sdf


def union(*sdfs):
	# Take the union of a set of sdfs

	@njit
	def union_sdf(p):
		minval = np.inf
		for s in literal_unroll(sdfs):
			minval = np.minimum(minval, s(p))
		return minval

	return union_sdf


def subtract(a, b):
	# Subtract sdf b from a

	@njit
	def subtract_sdf(p):
		return np.maximum(a(p), -b(p))

	return subtract_sdf


def intersect(a, b):
	# Take the intersection of sdfs a and b

	@njit
	def intersect_sdf(p):
		return np.maximum(a(p), b(p))

	return intersect_sdf


@njit
def sphere_sdf(p):
	return np.float32(norm(p) - 1.)

@njit
def plane_sdf(p):
	return dot(p, vec3(0,0,1))

@njit
def cube_sdf(p):
	q = np.abs(p)
	return np.float32(max(q[0], q[1], q[2]) - 1.)




def infcross_sdf():

	a = scale(cube_sdf, vec3(1,1,np.inf))
	b = scale(cube_sdf, vec3(1,np.inf,1))
	c = scale(cube_sdf, vec3(np.inf,1,1))
	infcross = union(a, b, c)
	
	@njit
	def infcross_sdf_func(p):
		return infcross(p)

	return infcross_sdf_func


def cross_from_box_sdf():

	a = cube_sdf
	b = scale(infcross_sdf(), vec3(1/3,1/3,1/3))
	cross_from_box = subtract(a, b)

	@njit
	def cross_from_box_func(p):
		return cross_from_box(p)

	return cross_from_box_func

def menger_sponge_sdf():

	infcross = infcross_sdf()

	@njit
	def menger_sponge_func(p):

		d = cube_sdf(p)

		s = 1.
		for i in range(3):

			a = np.fmod(p * s, 2.) - 1.
			r = 1. - 3.*np.abs(a)
			s *= 3.

			c = infcross(r)/s
			d = np.maximum(d, c)

		return np.float32(d)

	return menger_sponge_func


