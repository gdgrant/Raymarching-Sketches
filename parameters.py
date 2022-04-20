
par = {
	'pixels_x'	:	400,
	'pixels_y'	:	300,
	'cam_width' :	5.,

	'cam_x'		:	-10.,
	'cam_y'		:	-8.,
	'cam_z'		:	5.,
	'look_x'	:	0.,
	'look_y'	:	0.,
	'look_z'	:	2.,
	'cam_depth'	:	5.,
	'ray_step_mult'	:	1 - 1e-6,
}

par['screen_aspect'] = par['pixels_y'] / par['pixels_x']
par['cam_height'] = par['cam_width'] * par['screen_aspect']