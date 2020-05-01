import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.meshing.csg2d as csg2d

maxh = 0.5
eps = 1e-1
head = csg2d.Circle([0, 2], 2, maxh = maxh, eps = eps)
neck = csg2d.Rectangle([-1,-1],[1,2])
mouth = csg2d.Circle([0, 1], 0.5, maxh = maxh, eps = eps)
face = head + neck - mouth
nose = csg2d.Rectangle([-0.2,2], [0.2,3])
eyes = csg2d.Circle([-1, 3], 0.2, maxh = maxh, eps = eps) + csg2d.Circle([1, 3], 0.2, maxh = maxh, eps = eps)

geo = head.getCollection()
geo.add(face)
geo.add(nose, add_boundaries = False)
geo.add(eyes, add_boundaries = False)
geo.generateMesh(0.2)
geo.writeMesh('face')

filea = io.H5File('face', 'r')
print('a cell attributes:', filea.cell_attributes)
print('a facet attributes:', filea.facet_attributes)
domains = filea.read_attribute('domains')
boundaries = filea.read_attribute('boundaries')

space = dolfin.FunctionSpace(filea.mesh, 'CG', 1)
functiona = dolfin.interpolate(dolfin.Expression('x[0]', degree=1), space)
functionb = dolfin.interpolate(dolfin.Expression('x[0]*x[1]', degree=1), space)
vspace = dolfin.VectorFunctionSpace(filea.mesh, 'CG', 1)
vfunctiona = dolfin.interpolate(dolfin.Expression(('x[0]', '0'), degree=1), vspace)
vfunctionb = dolfin.interpolate(dolfin.Expression(('x[1]', '-x[0]'), degree=1), vspace)
filea.close()

fileb = io.H5File('face_copy', 'w')
fileb.set_mesh(filea.mesh)
fileb.add_attribute(domains, 'domains_copy')
fileb.add_attribute(boundaries, 'boundaries_copy')
fileb.add_function_group([functiona, functionb], 'scalars', scale_to_one = True)
fileb.add_function_group([vfunctiona, vfunctionb], 'vectors')
print('b cell attributes:', fileb.cell_attributes)
print('b facet attributes:', fileb.facet_attributes)
print('b scalar groups:', fileb.scalar_groups)
print('b vector groups:', fileb.vector_groups)
fileb.write_xdmf()
fileb.close()
del fileb

fileb = io.H5File('face_copy', 'r')
print('b reload cell attributes:', fileb.cell_attributes)
print('b reload facet attributes:', fileb.facet_attributes)
print('b reload scalar groups:', fileb.scalar_groups)
print('b reload vector groups:', fileb.vector_groups)
functions = fileb.read_function_group('scalars')
vfunctions = fileb.read_function_group('vectors')