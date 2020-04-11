""" Sample script to mesh 3 layered open hole specimen
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import netgen.csg as csg

import sa_thesis
import sa_thesis.meshing.csg3d as csg3d

cube = csg3d.OrthoBrick([-20, -5, -1], [20, 5, 1])
hole = csg3d.Cylinder([0, 0, -2], [0, 0, 2], 1, eps = 1e-1)

layers = csg3d.Layers(cube-hole, [0, 0, -1], [0, 0, 1], 3, elements_per_layer = 3)
layers.generateMesh(0.5)
layers.writeMesh('layers')