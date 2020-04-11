import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import netgen.csg as csg

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