import dolfin

type_dict = {
	'd': 'double',
	'u': 'size_t'
}

def solve(AA, uu, bb, *, krylov_args = ('cg', 'hypre_amg'), direct_args = ('mumps',)):
	try:
		solver = dolfin.KrylovSolver(*krylov_args)
		it = solver.solve(AA, uu.vector(), bb)
		print(f'krylov iterations: {it}')
	except:
		dolfin.solve(AA, uu.vector(), bb, *direct_args)
		print(f'direct solve after krylov failure')
		it = -1
	return it