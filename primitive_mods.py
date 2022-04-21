import numpy as np
from numba import njit, float32, prange, literal_unroll
from numba.experimental import jitclass
from PIL import Image

from parameters import par
from math_utils import *	# Particularly: vec3, norm


@njit
def mod_union(p, *sdfs):
	""" Given a point and two sdfs, evaluate the union
		of the two sdfs. """
	# return min(sdf1(p), sdf2(p))
	
	minval = np.inf
	for s in literal_unroll(sdfs):
		minval = np.minimum(minval, s(p))
	return minval


@njit
def mod_scale(p, s, sdf):
	""" Given a point p and an sdf, make the sdf object
		larger by a factor of s """

	# Scale internal distance by 1/s
	# But then also apply scaling to exterior, e.g.
	# a sphere of size 2 would have distance of 1
	# from the point (3,0,0), not a distance of 1/2

	# It's better to underestimate the distance
	# than to overestimate it

	return min(s) * sdf(p / s)

@njit
def mod_scale_aniso(p, s, sdf):
	""" Given a point p and an sdf, make the sdf object
		larger by a factor of s (which is in 3D) """

	# Scale internal distance by 1/s
	# But then also apply scaling to exterior, e.g.
	# a sphere of size 2 would have distance of 1
	# from the point (3,0,0), not a distance of 1/2

	# It's better to underestimate the distance
	# than to overestimate it

	return min(s) * sdf(p / s)

