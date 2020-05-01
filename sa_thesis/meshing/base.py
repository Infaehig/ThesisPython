""" Base classes and auxiliary functions for mesh generation
"""

from ..helpers import io

csg_eps = 1e-4

""" Base class for 2d and 3d CSG
"""
class CSG:
	def __init__(self, geometry, *, csg_boundaries = []):
		self.geometry = geometry
		self.csg_boundaries = csg_boundaries

	def __add__(self, other):
		return CSG(self.geometry + other.geometry, csg_boundaries = self.csg_boundaries + other.csg_boundaries)

	def __sub__(self, other):
		return CSG(self.geometry - other.geometry, csg_boundaries = self.csg_boundaries + other.csg_boundaries)

	def __mul__(self, other):
		return CSG(self.geometry * other.geometry, csg_boundaries = self.csg_boundaries + other.csg_boundaries)

	@classmethod
	def getCollection(cls):
		pass

class CSGCollection:
	def writeMesh(self, filename):
		with io.H5File(filename, 'w') as h5_file:
			h5_file.set_mesh(self.mesh)
			h5_file.add_attribute(self.domains, 'domains')
			h5_file.add_attribute(self.boundaries, 'boundaries')
			h5_file.write_xdmf()