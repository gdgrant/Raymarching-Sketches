import numpy as np
from numba import njit, float32, prange
from numba.experimental import jitclass

from camera import Ray
from math_utils import *

@njit
def compute_shadow(r, target, sdf):

	em3 = np.float32(1e-3)

	k = np.float32(4.)			# Penumbra sharpness
	penumbra = np.float32(1.)	# How shadowed to make the point
	travel = np.float32(0.)		# Accumulate distance casted

	r1 = np.inf

	counts = 0
	while counts < 10000:
		counts += 1

		# If the light target is hit, return
		# full illumination minus the penumbra
		to_target = dist(target, r.ori)
		if to_target < em3:
			# print(penumbra)
			return np.float32(penumbra)

		# Compute the SDF distance, and compare to
		# stepping to the light target
		sdf_step = sdf(r.ori)
		step = np.minimum(sdf_step, to_target)

		# If the SDF is hit, return no illumination
		# Otherwise, advance to the next step.
		if step < em3:
			return np.float32(0.)
		else:
			r.advance(step)

			if counts > 0:

				# Accumulate travel
				travel = travel + step

				# Set the minimum penumbra darkness to last value,
				# or to the scene distance divided by the total travel distance
				penumbra = np.minimum(penumbra, k*sdf_step/travel)

	else:
		raise Exception('Too many iterations in shadowcasting.')



@njit(parallel=True)
def compute_lighting(ray, sdf, lights):
	""" Phong lighting algorithm

		ray = incoming ray, splits into:
			p = current position
			v = direction from viewer
		sdf = environment
		lights = list of scene lights
	"""

	# Unpack reference point and viewing direction
	p, v = ray.ori, ray.dir

	# Invert viewing direction to point at viewer
	v = -v

	# Backstep reference  slightly
	p = p + v*np.float32(1e-3)

	# Ambient, specular, and diffuse intensity as color
	i_am = vec3(0.9,0.9,1)
	i_sp = vec3(0.9,1,0.9)
	i_di = vec3(1,0.9,0.9)

	# Ambient, specular, and diffuse reflectance
	k_am = np.float32(0.05)
	k_sp = np.float32(0.5)
	k_di = np.float32(0.3)

	# Shininess (higher = shinier)
	alpha = np.float32(1.2)

	# Reflectivity (0 = none, 1 = total)
	# reflectivity = 0.2

	# Find the surface normal from the sdf
	n = find_normal(p, sdf)

	# Ambient term
	light_val = k_am * i_am

	num_lights = len(lights)
	for light_id in prange(num_lights):
	# for light_loc in literal_unroll(lights):
		light_loc = lights[light_id]

		# Direction from position to light
		l = unit(light_loc - p)

		# Direction of perfect reflection
		r = 2 * dot(l, n) * n - l

		# Compute whether shadowed
		shadowed = compute_shadow(Ray(p, l), light_loc, sdf)
		if shadowed < 1e-3:
			continue
		
		# Diffuse term
		light_diff = k_di * pos_dot(l, n) * i_di

		# Specular term
		light_spec = k_sp * pos_dot(r, v)**alpha * i_sp

		# Combine light
		light_val += shadowed*(light_diff + light_spec).astype(np.float32)


	return light_val


