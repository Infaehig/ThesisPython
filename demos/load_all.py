import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_thesis

import sa_thesis.meshing.meshing2d as meshing2d
import sa_thesis.meshing.meshing3d as meshing3d
import sa_thesis.helpers.dimension as dimension

meshing2d.do_stuff()
meshing3d.do_stuff()
test = dimension.Definitions('nomesh')
print(test)