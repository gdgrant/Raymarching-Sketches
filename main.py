import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import time

from numba import jit, njit, prange
# https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#feature-typed-list
from numba.typed import List as nbList

from parameters import par
from math_utils import *
from sdfs import *
from lighting import *
from camera import camera_rays



from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)







cam_pos = vec3(par['cam_x'], par['cam_y'], par['cam_z'])
look_pos = vec3(par['look_x'], par['look_y'], par['look_z'])
rays = camera_rays(cam_pos, look_pos, \
	par['cam_width'], par['cam_height'], par['cam_depth'], \
	par['pixels_x'], par['pixels_y'])


@njit
def compute_ray(ray, lights, sdf, raystep, cam_pos):

	keep_going = True

	while keep_going:
		if dist(cam_pos, ray.ori) > 200:
			px_out = vec3(0,0,0)
			keep_going = False

		step = sdf(ray.ori)
		if step < 1e-3:
			px_out = compute_lighting(ray, sdf, lights)
			keep_going = False
		else:
			ray.advance(raystep * step)

	return px_out

@njit
def compute_rays(rays, lights, sdf, raystep, cam_pos):

	num_rays = len(rays)

	cs = np.empty((num_rays, 3), dtype=np.float32)
	for i in prange(num_rays):
		cs[i,:] = compute_ray(rays[i], lights, sdf, raystep, cam_pos)

	return cs



# sdf0 = translate(sphere_sdf, vec3(0, 0, -1))
# sdf1 = translate(sphere_sdf, vec3(5, 0, 2))
# sdf2 = translate(scale(sphere_sdf, vec3(2, 1, 4)), vec3(0, 5, 2))
# sph_sdf = union(sdf0, sdf1, sdf2)
# sdf = union(plane_sdf, translate(sph_sdf, vec3(0,0,2)))
# sdf = union(plane_sdf, sdf2, translate(cube_sdf, vec3(0,0,1)))

# a = scale(cube_sdf, vec3(1,1,np.inf))
# b = scale(cube_sdf, vec3(1,np.inf,1))
# c = scale(cube_sdf, vec3(np.inf,1,1))
# sdf = union(a, b, c)

# sdf = infcross_sdf()
# sdf = union(scale(cube_sdf, vec3(3,3,3)), infcross_sdf())
# sdf = cross_from_box_sdf()
sdf = scale(menger_sponge_sdf(), vec3(5,5,5))




lights = ( vec3(-10, 10, 10), vec3(0, -20, 50) )

nb_rays = nbList()
[nb_rays.append(r) for r in rays]

t0 = time.time()
cs = compute_rays(nb_rays, lights, sdf, par['ray_step_mult'], cam_pos)
print('Time to run: {:.1f} sec'.format(time.time() - t0))

img = pixels_to_image(cs, par['pixels_x'], par['pixels_y'])
img.show()
