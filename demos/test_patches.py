import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import dolfin
import numpy

import sa_thesis
import sa_thesis.helpers.patches as patches

mesh = dolfin.UnitCubeMesh(20,20,20)
hh = 1/3.
partition = patches.BoxPartition(mesh, hh)