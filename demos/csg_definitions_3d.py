import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin
import numpy

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.meshing.csg3d as csg3d
import sa_thesis.computation.problems as problems

cube = csg3d.OrthoBrick([-20, -5, -1], [20, 5, 1])
hole = csg3d.Cylinder([0, 0, -2], [0, 0, 2], 1, eps = 1e-1)
layers = csg3d.Layers(cube-hole, [0, 0, -1], [0, 0, 1], 3, elements_per_layer = 3)
layers.generateMesh(0.3)

mesh = layers.mesh
domains = layers.domains
facets = layers.facets

h5file = io.H5File('3layers', 'w')
h5file.set_mesh(mesh)
h5file.add_attribute(domains.array(), 'layers')

problem = problems.PoissonProblem(mesh, domains = domains, facets = facets)
ff = dolfin.Constant(-1.)
uu = problem.assemble_and_solve(ff, [(dolfin.Constant(1), 5), (dolfin.Constant(-1), 6)])
h5file.add_function_group([uu], 'p_constant')
del problem, ff, uu

coeff = problems.PoissonProblem.default_coefficient
problem = problems.PoissonProblem(mesh, coefficients = [coeff(aa = 100), coeff(), coeff(aa = 100)], domains = domains, facets = facets)
ff = dolfin.Constant(-1.)
uu = problem.assemble_and_solve(ff, [(dolfin.Constant(1), 5), (dolfin.Constant(-1), 6)])
h5file.add_function_group([uu], 'p_variable')
del coeff, problem, ff, uu

problem = problems.LinearElasticityProblem(mesh, domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'e_constant')
del problem, ff, uu

coeff = problems.LinearElasticityProblem.default_coefficient
problem = problems.LinearElasticityProblem(mesh, coefficients = [
	coeff(), coeff(EE=1e4, nu=0.45), coeff()
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
	coeff(angle=0), coeff(angle=numpy.pi/4), coeff(angle=numpy.pi/2)
], domains = domains, facets = facets)
ff = dolfin.Constant((0., 0., -10.))
uu = problem.assemble_and_solve(ff, [(dolfin.Constant((-0.1, -0.1, 0)), 1), (dolfin.Constant((0.1, 0.1, 0)), 2)], [(dolfin.Constant((0, -100, 0)), 3), (dolfin.Constant((0, 100, 0)), 4)])
h5file.add_function_group([uu], 'o_variable')
del problem, ff, uu

h5file.write_xdmf()
h5file.close()