""" Base classes and auxiliary functions for mesh generation
"""

from ..helpers import io

csg_eps = 1e-4

""" Base class for 2d and 3d CSG
"""
class CSG:
	def __init__(self, geometry, *, boundaries = []):
		self.geometry = geometry
		self.boundaries = boundaries

	def __add__(self, other):
		return CSG(self.geometry + other.geometry, boundaries = self.boundaries + other.boundaries)

	def __sub__(self, other):
		return CSG(self.geometry - other.geometry, boundaries = self.boundaries + other.boundaries)

	def __mul__(self, other):
		return CSG(self.geometry * other.geometry, boundaries = self.boundaries + other.boundaries)

	@classmethod
	def getCollection(cls):
		pass

class CSGCollection:
	def writeMesh(self, filename):
		with io.H5File(filename, 'w') as h5_file:
			h5_file.set_mesh(self.mesh)
			h5_file.add_attribute(self.domains.array(), 'domains')
			h5_file.write_xdmf_cells()
			h5_file.add_attribute(self.facets.array(), 'boundaries')
			h5_file.write_xdmf_facets()