"""
Helper functions for 2d/3d independent handling of functions spaces etc
"""
import fenics

class Definitions:
	def __init__(self, mesh):
		self.test = f'Test: {mesh}'

	def __str__(self):
		return self.test