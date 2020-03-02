from dolfin import *
import numpy as np

parameters["refinement_algorithm"] = "plaza_with_parent_facets"

oversample = 5

coarse_mesh = UnitSquareMesh(1,1)

meshes = [coarse_mesh]

markers = [[FacetFunction('size_t', meshes[-1], 1)]]
for ii in range(oversample):
    allmarker = CellFunction('bool', meshes[-1], True)
    meshes.append(refine(meshes[-1], allmarker))
    for mark in markers:
        mark.append(adapt(mark[-1], meshes[-1]))
        for facet in facets(meshes[-1]):
            if mark[-1][facet] != 1:
                mark[-1][facet] = 0
    markers.append([FacetFunction('size_t', meshes[-1], 1)])

mesh = meshes[-1]
fine_VV = FunctionSpace(mesh, 'CG', 1)
fine_uu = TrialFunction(fine_VV)
fine_vv = TestFunction(fine_VV)
fine_aa = inner(grad(fine_uu),grad(fine_vv))*dx
fine_LL = Constant(0.)*fine_vv*dx

for ii in range(len(meshes)-1):
    plot(markers[ii][-1], title='marker '+str(ii))
    coarse_mesh = meshes[ii]
    coarse_VV = FunctionSpace(coarse_mesh, 'CG', 1)
    uu = Function(coarse_VV)
    uu.vector()[coarse_VV.dim()/2] = 1
    uu_interp = interpolate(uu, fine_VV)
    bc = DirichletBC(fine_VV, uu, markers[ii][-1], 1)
    res = Function(fine_VV)
    solve(fine_aa == fine_LL, res, bc)
    err = np.sqrt(assemble(div(grad(res))**2*dx(mesh)))
    plot(res, title='r  '+str(err))




