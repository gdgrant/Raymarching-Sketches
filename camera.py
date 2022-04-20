import numpy as np
from numba import njit, float32
from numba.experimental import jitclass

from math_utils import *

@jitclass([ ('ori', float32[:]), ('dir', float32[:]) ])
class Ray:

	def __init__(self, origin, direction):
		self.ori = origin
		self.dir = direction

	def advance(self, distance):
		self.ori = self.ori + np.float32(distance) * self.dir

@njit
def generate_rays(x_screen, y_screen, cam_pos, p0, m, l):

	camera_rays = []
	for cx in x_screen:
		for cy in y_screen:
			p1 = cam_pos + cx*m + cy*l

			R = Ray(p1, unit(p1 - p0))
			camera_rays.append(R)

	return camera_rays

# @njit
def camera_rays(cam_pos, look_pos, cam_width, cam_height, cam_depth, pixels_x, pixels_y):


	# n = direction camera is facing
	n = unit(look_pos - cam_pos)

	# m = horizontal vector of screen
	m = unit(vec3(n[1], -n[0], 0))

	# l = vertical vector of screen
	l = unit(cross(m, n))

	x_screen = np.linspace(-cam_width/2, cam_width/2, pixels_x, dtype=np.float32)
	y_screen = np.linspace(-cam_height/2, cam_height/2, pixels_y, dtype=np.float32)

	p0 = cam_pos - n*cam_depth

	return generate_rays(x_screen, y_screen, cam_pos, p0, m, l)


