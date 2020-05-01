"""
Module for input output

In particular there are functions for handling hdf5 files for storing meshes and computation results.
"""

import os
import dolfin
#import dolfin.MPI as MPI
import h5py
import numpy
from .utils import type_dict

""" Wrapper around h5py with functions to read and write entities from dolfin
"""
class H5File:
	def __init__(self, filename, mode = 'r', *, compression = None):
		self.basename = os.path.splitext(filename)[0]
		self.file = h5py.File(f'{self.basename}.h5', mode)
		self.compression = compression
		self.cell_attributes = []
		self.facet_attributes = []
		self.cell_attribute_groups = []
		self.facet_attribute_groups = []
		self.scalar_groups = []
		self.vector_groups = []
		if mode == 'r':
			xdmf = dolfin.XDMFFile(dolfin.MPI.comm_world, f'{self.basename}_mesh.xdmf')
			self.mesh = dolfin.Mesh()
			xdmf.read(self.mesh)
			self.mesh.init()
			self.basedim = self.mesh.geometric_dimension()
			self.num_vertices = self.mesh.num_vertices()
			self.num_cells = self.mesh.num_cells()
			self.num_facets = self.mesh.num_facets()
			self.dimension_dict = {
				self.num_cells: self.basedim,
				self.num_facets: self.basedim-1
			}
			for key in self.file.keys():
				if key in ['vertices', 'cells', 'facets']:
					continue
				dset = self.file[key]
				if isinstance(dset, h5py.Group):
					data = dset['0']
					if data.shape[0] == self.num_vertices:
						self.scalar_groups.append(key)
					elif data.shape[0] / self.num_vertices == self.basedim:
						self.vector_groups.append(key)
					elif data.shape[0] == self.num_cells:
						self.cell_attribute_groups.append(key)
					elif data.shape[0] == self.num_facets:
						self.facet_attribute_groups.append(key)
				elif dset.shape[0] == self.num_cells:
					self.cell_attributes.append(key)
				elif dset.shape[0] == self.num_facets:
					self.facet_attributes.append(key)

	def close(self):
		if hasattr(self, 'file'):
			self.file.close()
			del self.file

	def __del__(self):
		self.close()

	def __enter__(self):
		return self

	def __exit__(self, exception_type, exception_value, traceback):
		self.close()

	def set_mesh(self, mesh, *, transform = None):
		mesh.init()
		self.mesh = mesh
		self.basedim = mesh.geometric_dimension()
		self.num_vertices = mesh.num_vertices()
		self.num_cells = mesh.num_cells()
		self.num_facets = mesh.num_facets()
		self.dimension_dict = {
			self.num_cells: self.basedim,
			self.num_facets: self.basedim-1
		}
		if transform is None:
			self.file.create_dataset('vertices', (self.num_vertices, self.basedim), data = mesh.coordinates(), compression = self.compression)
		else:
			self.file.create_dataset('vertices', (self.num_vertices, self.basedim), data = transform(mesh.coordinates()), compression = self.compression)
		self.file.create_dataset('cells', (self.num_cells, self.basedim + 1), data = numpy.array(mesh.cells(), dtype = numpy.uintp), compression = self.compression)
		self.file.create_dataset('facets', (self.num_facets, self.basedim), data = numpy.array([facet.entities(0) for facet in dolfin.facets(mesh)], dtype = numpy.uintp), compression = self.compression)

	def add_attribute(self, meshfunction, key):
		data = meshfunction.array()
		shape = data.shape
		self.file.create_dataset(key, shape, data = data, compression = self.compression)
		if shape[0] == self.num_cells:
			self.cell_attributes.append(key)
		elif shape[0] == self.num_facets:
			self.facet_attributes.append(key)

	def read_attribute(self, key):
		try:
			dset = self.file[key]
			dim = self.dimension_dict[dset.shape[0]]
			dtype = type_dict[dset.dtype.kind]
			mf = dolfin.MeshFunction(dtype, self.mesh, dim)
			mf.set_values(dset[:])
			return mf
		except:
			raise Exception(f'Error reading attribute: {key}')
	
	def add_attribute_group(self, meshfunctions, key):
		shape = meshfunctions[0].array().shape
		group = self.file.create_group(key)
		for ii, meshfunction in enumerate(meshfunctions):
			group.create_dataset(f'{ii}', shape, data = meshfunction.array(), compression = self.compression)
		if shape[0] == self.num_cells:
			self.cell_attribute_groups.append(key)
		elif shape[0] == self.num_facets:
			self.facet_attribute_groups.append(key)
	
	def read_attribute_group(self, key):
		try:
			group = self.file[key]
			first = group['0']
			dim = self.dimension_dict[first.shape[0]]
			dtype = type_dict[first.dtype.kind]
			meshfunctions = []
			ii = 0; key = f'{ii}'
			while key in group:
				dset = group[key]
				meshfunctions.append(dolfin.MeshFunction(dtype, self.mesh, dim))
				meshfunctions[-1].set_values(dset[:])
				ii += 1; key = f'{ii}'
			return meshfunctions
		except:
			raise Exception(f'Error reading attribute group: {key}')

	def add_dataset_group(self, datasets, key):
		group = self.file.create_group(key)
		for ii, dataset in enumerate(datasets):
			group.create_dataset(f'{ii}', dataset.shape, data = dataset, compression = self.compression)
	
	def read_dataset_group(self, key):
		try:
			group = self.file[key]
			first = group['0']
			dtype = type_dict[first.dtype.kind]
			datasets = []
			ii = 0; key = f'{ii}'
			while key in group:
				dset = group[key]
				datasets.append(numpy.array(dset))
				ii += 1; key = f'{ii}'
			return datasets
		except:
			raise Exception(f'Error reading attribute group: {key}')

	def add_function_group(self, functions, key, *, scale_to_one = False):
		group = self.file.create_group(key)
		index_map = dolfin.vertex_to_dof_map(functions[0].function_space())
		if scale_to_one:
			for ii, uu in enumerate(functions):
				data = uu.vector()[index_map]
				umin = numpy.min(data); umax = numpy.max(data)
				if umax < -umin:
					data /= -umin
				elif 0 < umax:
					data /= umax
				group.create_dataset(f'{ii}', data.shape, data = data, compression = self.compression)
		else:
			for ii, uu in enumerate(functions):
				data = uu.vector()[index_map]
				group.create_dataset(f'{ii}', data.shape, data = data, compression = self.compression)
		if data.shape[0] == self.num_vertices:
			self.scalar_groups.append(key)
		else:
			assert(data.shape[0] / self.num_vertices == self.basedim)
			self.vector_groups.append(key)

	def read_function_group(self, key):
		group = self.file[key]
		if key in self.scalar_groups:
			space = dolfin.FunctionSpace(self.mesh, 'CG', 1)
		elif key in self.vector_groups:
			space = dolfin.VectorFunctionSpace(self.mesh, 'CG', 1)
		index_map = dolfin.dof_to_vertex_map(space)
		functions = []
		ii = 0; key = f'{ii}'
		while key in group:
			dset = group[key]
			functions.append(dolfin.Function(space, name = 'u'))
			functions[-1].vector().set_local(dset[:][index_map])
			ii += 1; key = f'{ii}'
		return functions
	
	def write_xdmf(self):
		cell_file = open(f'{self.basename}_mesh.xdmf', 'w')
		cell_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "mesh" GridType = "Uniform">
			<Topology NumberOfElements = "{self.num_cells}" TopologyType = "{'Tetrahedron' if self.basedim == 3 else 'Triangle'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_cells} {self.basedim + 1}">{self.basename}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
			</Geometry>''')
		for key in self.cell_attributes:
			cell_file.write(f'''
			<Attribute Name = "{key}" AttributeType = "Scalar" Center = "Cell">
				<DataItem Format = "HDF" Dimensions = "{self.num_cells} 1">{self.basename}.h5:/{key}</DataItem>
			</Attribute>''')
		cell_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
		cell_file.close()
		del cell_file

		for key in self.cell_attribute_groups:
			group = self.file[key]
			group_len = len(group.keys())
			cell_file = open(f'{self.basename}_{key}.xdmf', 'w')
			cell_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "{key}" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{group_len}">
					{' '.join(group.keys())}
				</DataItem>
			</Time>''')
			for ii in range(group_len):
				cell_file.write(f'''
			<Grid Name = "grid_{ii}" GridType = "Uniform">
				<Topology NumberOfElements = "{self.num_cells}" TopologyType = "{'Tetrahedron' if self.basedim == 3 else 'Triangle'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_cells} {self.basedim + 1}">{self.basename}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "{key}" AttributeType = "Scalar" Center = "Cell">
					<DataItem Format = "HDF" Dimensions = "{self.num_cells} 1">{self.basename}.h5:/{key}/{ii}</DataItem>
				</Attribute>
			</Grid>''')
			cell_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
			cell_file.close()
			del cell_file

		for key in self.scalar_groups:
			group = self.file[key]
			group_len = len(group.keys())
			cell_file = open(f'{self.basename}_{key}.xdmf', 'w')
			cell_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "{key}" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{group_len}">
					{' '.join(group.keys())}
				</DataItem>
			</Time>''')
			for ii in range(group_len):
				cell_file.write(f'''
			<Grid Name = "grid_{ii}" GridType = "Uniform">
				<Topology NumberOfElements = "{self.num_cells}" TopologyType = "{'Tetrahedron' if self.basedim == 3 else 'Triangle'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_cells} {self.basedim + 1}">{self.basename}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "u" AttributeType = "Scalar" Center = "Node">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} 1">{self.basename}.h5:/{key}/{ii}</DataItem>
				</Attribute>
			</Grid>''')
			cell_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
			cell_file.close()
			del cell_file

		for key in self.vector_groups:
			group = self.file[key]
			group_len = len(group.keys())
			cell_file = open(f'{self.basename}_{key}.xdmf', 'w')
			cell_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "{key}" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{group_len}">
					{' '.join(group.keys())}
				</DataItem>
			</Time>''')
			for ii in range(group_len):
				cell_file.write(f'''
			<Grid Name = "grid_{ii}" GridType = "Uniform">
				<Topology NumberOfElements = "{self.num_cells}" TopologyType = "{'Tetrahedron' if self.basedim == 3 else 'Triangle'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_cells} {self.basedim + 1}">{self.basename}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "v" AttributeType = "Vector" Center = "Node">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/{key}/{ii}</DataItem>
				</Attribute>''')
				for dim in range(self.basedim):
					cell_file.write(f'''
				<Attribute Name = "u_{dim}" AttributeType = "Scalar" Center = "Node">
					<DataItem ItemType = "HyperSlab" Dimensions = "{self.num_vertices} 1">
						<DataItem Dimensions = "3 1" Format = "XML">
							{dim}
							{self.basedim}
							{self.num_vertices}
						</DataItem>
						<DataItem Format = "HDF" Dimensions = "{self.num_vertices * self.basedim}">{self.basename}.h5:/{key}/{ii}</DataItem>
					</DataItem>
				</Attribute>''')
				cell_file.write('''
			</Grid>''')
			cell_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
			cell_file.close()
			del cell_file
	
		facet_file = open(f'{self.basename}_facets.xdmf', 'w')
		facet_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "facets" GridType = "Uniform">
			<Topology NumberOfElements = "{self.num_facets}" TopologyType = "{'Triangle' if self.basedim == 3 else 'PolyLine" NodesPerElement = "2'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_facets} {self.basedim}">{self.basename}.h5:/facets</DataItem>
			</Topology>
			<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
			</Geometry>''')
		for key in self.facet_attributes:
			facet_file.write(f'''
			<Attribute Name = "{key}" AttributeType = "Scalar" Center = "Cell">
				<DataItem Format = "HDF" Dimensions = "{self.num_facets} 1">{self.basename}.h5:/{key}</DataItem>
			</Attribute>''')
		facet_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
		facet_file.close()
		del facet_file

		for key in self.facet_attribute_groups:
			group = self.file[key]
			group_len = len(group.keys())
			facet_file = open(f'{self.basename}_{key}.xdmf', 'w')
			facet_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "{key}" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{group_len}">
					{' '.join(group.keys())}
				</DataItem>
			</Time>''')
			for ii in range(group_len):
				facet_file.write(f'''
			<Grid Name = "grid_{ii}" GridType = "Uniform">
				<Topology NumberOfElements = "{self.num_facets}" TopologyType = "{'Triangle' if self.basedim == 3 else 'PolyLine" NodesPerElement = "2'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_facets} {self.basedim}">{self.basename}.h5:/facets</DataItem>
				</Topology>
				<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
					<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "{key}" AttributeType = "Scalar" Center = "Cell">
					<DataItem Format = "HDF" Dimensions = "{self.num_facets} 1">{self.basename}.h5:/{key}</DataItem>
				</Attribute>
			</Grid>''')
			facet_file.write('''
		</Grid>
	</Domain>
</Xdmf>''')
			facet_file.close()
			del cell_file

