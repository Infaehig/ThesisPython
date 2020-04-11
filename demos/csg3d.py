import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import netgen.csg as csg

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.meshing.csg3d as csg3d

cube = csg3d.OrthoBrick([-20, -5, -1], [20, 5, 1])
hole = csg3d.Cylinder([0, 0, -2], [0, 0, 2], 1, eps = 1e-1)
layers = csg3d.Layers(cube-hole, [0, 0, -1], [0, 0, 1], 3, elements_per_layer = 3)
layers.generateMesh(0.5)
layers.writeMesh('layers')

filea = io.H5File('layers', 'r')
domains = filea.read_attribute('domains')
boundaries = filea.read_attribute('boundaries')

fileb = io.H5File('layers_copy', 'w')
fileb.set_mesh(filea.mesh)
fileb.add_attribute(domains.array(), 'domains_copy')
fileb.add_attribute(boundaries.array(), 'boundaries_copy')
fileb.write_xdmf_cells()
fileb.write_xdmf_facets()