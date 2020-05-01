import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin

import sa_thesis
import sa_thesis.helpers.io as io
import sa_thesis.meshing.csg3d as csg3d
import sa_thesis.helpers.patches as patches

cube = csg3d.OrthoBrick([-20, -5, -1], [20, 5, 1])
hole = csg3d.Cylinder([0, 0, -2], [0, 0, 2], 1, eps = 1e-1)
layers = csg3d.Layers(cube-hole, [0, 0, -1], [0, 0, 1], 3, elements_per_layer = 3)
layers.generateMesh(0.3)

hh = 15
partition = patches.BoxPartition(layers.mesh, hh, weight_degree = 3)
partition.write('layers')
del partition

partition = patches.BoxPartition.read('layers')
partition.write('layers_copy')