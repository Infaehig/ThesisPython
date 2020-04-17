import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin
import numpy

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.computation.problems as problems

mesh = dolfin.UnitCubeMesh(4, 4, 4)
domains = dolfin.MeshFunction('size_t', mesh, 3, 0)
half = dolfin.AutoSubDomain(lambda xx, on: xx[0] > 0.5)
half.mark(domains, 1)
facets = dolfin.MeshFunction('size_t', mesh, 2, 0)
left = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], 0))
right = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], 1))
front = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], 0))
back = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], 1))
bottom = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], 0))
top = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], 1))
left.mark(facets, 1)
right.mark(facets, 2)
front.mark(facets, 3)
back.mark(facets, 4)
bottom.mark(facets, 5)
top.mark(facets, 6)

h5file = io.H5File('cube', 'w')
h5file.set_mesh(mesh)
h5file.add_attribute(domains.array(), 'sides')

problem = problems.PoissonProblem(mesh, domains = domains, facets = facets)
ff = dolfin.Constant(-1.)
uu = problem.assemble_and_solve(ff, [(dolfin.Constant(1), 1), (dolfin.Constant(-1), 2)])
h5file.add_function_group([uu], 'p_constant')
del problem, ff, uu

coeff = problems.PoissonProblem.default_coefficient
problem = problems.PoissonProblem(mesh, coefficients = [coeff(), coeff(aa = 100)], domains = domains, facets = facets)
ff = dolfin.Constant(-1.)
uu = problem.assemble_and_solve(ff, [(dolfin.Constant(1), 1), (dolfin.Constant(-1), 2)])
h5file.add_function_group([uu], 'p_variable')
del coeff, problem, ff, uu

problem = problems.LinearElasticityProblem(mesh, domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'e_constant')
del problem, ff, uu

coeff = problems.LinearElasticityProblem.default_coefficient
problem = problems.LinearElasticityProblem(mesh, coefficients = [
	coeff(), coeff(EE=1e4, nu=0.45)
], domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'e_variable')
del coeff, problem, ff, uu

problem = problems.OrthotropicMaterialProblem(mesh, domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'o_constant')
del problem, ff, uu

coeff = problems.OrthotropicMaterialProblem.default_coefficient
problem = problems.OrthotropicMaterialProblem(mesh, coefficients = [
	coeff(angle=0), coeff(angle=numpy.pi/2)
], domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'o_variable')
del problem, ff, uu

h5file.write_xdmf()
h5file.close()