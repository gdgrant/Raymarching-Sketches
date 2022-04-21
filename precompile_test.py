

if False:
	from numba.pycc import CC

	cc = CC('sdf_module')


	def make_sdf(func):

		# @cc.export('p_sphere', 'f4(f4[:])')
		# def sdf(*args, **kwargs):
		# 	return lambda p : func(p, *args, **kwargs)
		
		def sdf(p):
			return func(p)
			
		return sdf

	@make_sdf#(name='p_sphere', sig='f4(f4[:])')
	def sphere(p):
		return norm(p) - 1.

	@make_sdf#('t_scale', 'f4(f4[:], f4, )')
	def scale(p, s, sdf):
		return s * sdf(p/s)

	# sdf = scale(2., sphere())
	@cc.export('run_sdf', 'f4(f4[:])')
	def run_sdf(p):
		return sphere(p)

	cc.compile()

	import sdf_module
	print(sdf_module.run_sdf(vec3(2.,0.,0.)))


	# print(sdf(vec3(1.,0.,0.,)))
	# print(sdf(vec3(2.,0.,0.,)))
	# print(sdf(vec3(3.,0.,0.,)))

