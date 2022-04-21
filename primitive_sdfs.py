import numpy as np
from numba import njit, float32, prange, literal_unroll
from numba.experimental import jitclass
from PIL import Image

from parameters import par
from math_utils import *	# Particularly: vec3, norm


### It's like there needs to be an SDF class
### Where a method in the class is the primitive
### And then (maybe inheritable) is the ability to
### perform the primitive modifications to the primitive
### sdf shapes?


# @jitclass([ (), () ])
# class Sphere:

# 	def __init__(self):
# 		self.rad = 1.

# 	def sdf(self, p):
# 		return norm(p) - self.rad



@njit
def sdf_sphere(p):
	""" Return the evaluated sdf of a sphere of radius 1
		compared to a point p """
	return norm(p) - 1.

@njit
def sdf_cube(p):
	""" Return the evaluated sdf of a cube of side length 1
		compared to a point p """
	return max(np.abs(p)) - 0.5

@njit
def sdf_plane(p, n, h):
	""" Return the evaluated sdf of a plane of height h
		with normal direction n, compared to a point p """

	# Note that n must be normalized
	return dot(p, n) + h




from primitive_mods import mod_scale, mod_scale_aniso

x = mod_scale(vec3(3.,0.,0.,), vec3(1., 2., 2.), sdf_cube)
print(x)



