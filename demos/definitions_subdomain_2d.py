import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin
import numpy

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.computation.problems as problems

mesh = dolfin.UnitSquareMesh(4, 4)
domains = dolfin.MeshFunction('size_t', mesh, 2, 0)
half = dolfin.AutoSubDomain(lambda xx, on: xx[0] > 0.5)
half.mark(domains, 1)
facets = dolfin.MeshFunction('size_t', mesh, 1, 0)
left = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], 0))
right = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], 1))
front = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], 0))
back = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], 1))
left.mark(facets, 1)
right.mark(facets, 2)
front.mark(facets, 3)
back.mark(facets, 4)

coeff = problems.PoissonProblem.default_coefficient
problem = problems.PoissonProblem(mesh, coefficients = [coeff(), coeff(aa = 100)], domains = domains, facets = facets)
ff = dolfin.Constant(-1.)
gg = [(dolfin.Constant(1), 1), (dolfin.Constant(-1), 2)]

lower = dolfin.MeshFunction('size_t', mesh, 2, 0)
lower_subdomain = dolfin.AutoSubDomain(lambda xx, on: xx[1] > 0)
lower_subdomain.mark(lower, 1)

h5file = io.H5File('whole', 'w')
h5file.set_mesh(mesh)
h5file.add_attribute(lower.array(), 'lower')
uu = problem.assemble_and_solve(ff, gg)
h5file.add_function_group([uu], 'p_variable')
h5file.write_xdmf()
h5file.close()

h5file = io.H5File('lower', 'w')
submesh = dolfin.SubMesh(mesh, lower, 1)
h5file.set_mesh(submesh)
subfacets = dolfin.MeshFunction('size_t', submesh, 1, 0)
left.mark(subfacets, 1)
right.mark(subfacets, 2)
front.mark(subfacets, 3)
back.mark(subfacets, 4)
uu = problem.assemble_and_solve(ff, gg, mesh = submesh, facets = subfacets)
h5file.add_function_group([uu], 'p_variable')
h5file.write_xdmf()
h5file.close()