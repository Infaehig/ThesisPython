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
	def __init__(self, filename, mode='r', *, compression = None):
		self.basename = os.path.splitext(filename)[0]
		self.file = h5py.File(f'{self.basename}.h5', mode)
		self.compression = compression
		self.cell_attributes = []
		self.facet_attributes = []
		if mode == 'r':
			xdmf = dolfin.XDMFFile(dolfin.MPI.comm_world, f'{self.basename}_cells.xdmf')
			self.mesh = dolfin.Mesh()
			xdmf.read(self.mesh)
			self.mesh.init()
			self.basedim = self.mesh.geometry().dim()
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
					continue
				if dset.shape[0] == self.num_cells:
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
		self.basedim = mesh.geometry().dim()
		self.num_vertices = mesh.num_vertices()
		self.num_cells = mesh.num_cells()
		self.num_facets = mesh.num_facets()
		self.dimension_dict = {
			self.num_cells: self.basedim,
			self.num_facets: self.basedim-1
		}
		if transform is None:
			self.file.create_dataset('/vertices', (self.num_vertices, self.basedim), data = mesh.coordinates(), compression = self.compression)
		else:
			self.file.create_dataset('/vertices', (self.num_vertices, self.basedim), data = transform(mesh.coordinates()), compression = self.compression)
		self.file.create_dataset('/cells', (self.num_cells, self.basedim+1), data = numpy.array(mesh.cells(), dtype = numpy.uintp), compression = self.compression)
		self.file.create_dataset('/facets', (self.num_facets, self.basedim), data = numpy.array([facet.entities(0) for facet in dolfin.facets(mesh)], dtype = numpy.uintp), compression = self.compression)

	def add_attribute(self, data, key):
		shape = data.shape
		self.file.create_dataset(f'/{key}', shape, data = data, compression = self.compression)
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

	def write_xdmf_cells(self):
		cell_file = open(f'{self.basename}_cells.xdmf', 'w')
		attributes = ''
		for name in self.cell_attributes:
			attributes += f'''
			<Attribute Name = "{name}" AttributeType = "Scalar" Center = "Cell">
				<DataItem Format = "HDF" Dimensions = "{self.num_cells} 1">{self.basename}.h5:/{name}</DataItem>
			</Attribute>'''
		cell_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "cells" GridType = "Uniform">
			<Topology NumberOfElements = "{self.num_cells}" TopologyType = "{'Tetrahedron' if self.basedim == 3 else 'Triangle'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_cells} {self.basedim+1}">{self.basename}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
			</Geometry>{attributes}
		</Grid>
	</Domain>
</Xdmf>''')
		cell_file.close()
	
	def write_xdmf_facets(self):
		facet_file = open(f'{self.basename}_facets.xdmf', 'w')
		attributes = ''
		for name in self.facet_attributes:
			attributes += f'''
			<Attribute Name = "{name}" AttributeType = "Scalar" Center = "Cell">
				<DataItem Format = "HDF" Dimensions = "{self.num_facets} 1">{self.basename}.h5:/{name}</DataItem>
			</Attribute>'''
		facet_file.write(f'''<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "facets" GridType = "Uniform">
			<Topology NumberOfElements = "{self.num_facets}" TopologyType = "{'Triangle' if self.basedim == 3 else 'PolyLine" NodesPerElement = "2'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_facets} {self.basedim}">{self.basename}.h5:/facets</DataItem>
			</Topology>
			<Geometry GeometryType = "{'XYZ' if self.basedim == 3 else 'XY'}">
				<DataItem Format = "HDF" Dimensions = "{self.num_vertices} {self.basedim}">{self.basename}.h5:/vertices</DataItem>
			</Geometry>{attributes}
		</Grid>
	</Domain>
</Xdmf>''')
		facet_file.close()

"""

def read_dolfin_mesh(basename, dolf = True):
	print('reading ['+basename+'] mesh and markers')
	h5_file = h5py.File(basename+'.h5', 'r')
	h5_vertices = h5_file['/vertices']
	shape_vertices = h5_vertices.shape
	h5_cells = h5_file['/cells']
	shape_cells = h5_cells.shape
	basedim = shape_vertices[1]
	assert(basedim == shape_cells[1]-1)

	h5_cell_function = None
	if '/cell_function' in h5_file:
		h5_cell_function = h5_file['/cell_function']

	h5_cell_coeff = None
	if '/cell_coeff' in h5_file:
		h5_cell_coeff = h5_file['/cell_coeff']

	h5_facet_function = None
	if '/facet_function' in h5_file:
		h5_facet_function = h5_file['/facet_function']
	
	ret = dict()
	if not dolf:
		ret['vertices'] = h5_vertices[:]
		ret['cells'] = h5_cells[:]
		if h5_cell_function is not None:
			ret['cell_function'] = h5_cell_function[:]
		if h5_cell_coeff is not None:
			ret['cell_coeff'] = h5_cell_coeff[:]
		if h5_facet_function is not None:
			ret['facet_function'] = h5_facet_function[:]
	else:
		mesh = Mesh()
		ff = XDMFFile(comm, basename+'_cells.xdmf')
		ff.read(mesh)
		del ff
		ret['mesh'] = mesh
	   
		if h5_cell_function is not None:
			mc = MeshFunction('size_t', mesh, basedim)
			mc.set_values(h5_cell_function[:])
			ret['cell_function'] = mc

		if h5_cell_coeff is not None:
			coeff = MeshFunction('double', mesh, basedim)
			coeff.set_values(h5_cell_coeff[:])
			ret['cell_coeff'] = coeff

		if h5_facet_function is not None:
			mf = MeshFunction('size_t', mesh, basedim-1)
			mf.set_values(h5_facet_function[:])
			ret['facet_function'] = mf
	
	h5_file.close()
	print('read ['+basename+'] mesh and markers')
	return ret

def write_dolfin_scalar_cg1(basename, scalar_functions, scale_to_one = False):
	print('write_dolfin_scalar_cg1({:s})'.format(basename))
	real_dirname = os.path.dirname(basename)
	sa_utils.makedirs_norace(real_dirname)
	real_basename = os.path.basename(basename)

	mesh = scalar_functions[0].function_space().mesh()
	mesh.init()
	basedim = mesh.geometry().dim()
	num_vertices = mesh.num_vertices()
	num_cells = mesh.num_cells()

	h5_file = h5py.File(basename+'.h5', 'w')
	h5_vertices = h5_file.create_dataset('/vertices', (num_vertices, basedim), data = mesh.coordinates(), compression = my_compression)
	h5_cells = h5_file.create_dataset('/cells', (num_cells, basedim+1), data = numpy.array(mesh.cells(), dtype = numpy.uintp), compression = my_compression)
	h5_grp = h5_file.create_group('/scalar')
	num = len(scalar_functions)
	index_map = vertex_to_dof_map(scalar_functions[0].function_space())

	if scale_to_one:
		for ii, uu in enumerate(scalar_functions):
			data = uu.vector()[index_map]
			umin = numpy.min(data); umax = numpy.max(data)
			if umax < -umin:
				data /= -umin
			elif 0 < umax:
				data /= umax
			h5_grp.create_dataset('{:d}'.format(ii), (num_vertices, ), data = data, compression = my_compression)
	else:
		for ii, uu in enumerate(scalar_functions):
			h5_grp.create_dataset('{:d}'.format(ii), (num_vertices, ), data = uu.vector()[index_map], compression = my_compression)
	h5_file.close()
	
	scalar_file = open(basename+'.xdmf', 'w')
	string = r" ""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "cells" GridType = "Uniform">
			<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
			</Geometry>
		</Grid>" "".format(
		num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
		num_cells, basedim+1, real_basename, 
		"XYZ" if basedim == 3 else "XY", 
		num_vertices, basedim, real_basename
	)
	string += " ""
		<Grid Name = "scalar" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{:d}">
					" "".format(num)
	for ii in range(num):
		string += " {:d}".format(ii)
	string += r" ""
				</DataItem>
			</Time>" ""
	for ii in range(num):
		string += r" ""
			<Grid Name = "grid_{:d}" GridType = "Uniform">
				<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "u" AttributeType = "Scalar" Center = "Node">
					<DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/scalar/{:d}</DataItem>
				</Attribute>
			</Grid>" "".format(ii, 
							  num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
							  num_cells, basedim+1, real_basename, 
							  "XYZ" if basedim == 3 else "XY", 
							  num_vertices, basedim, real_basename, 
							  num_vertices, real_basename, ii)
	string += r" ""
		</Grid>
	</Domain>
</Xdmf>" ""
	scalar_file.write(string)
	scalar_file.close()
	print('write_dolfin_scalar_cg1({:s}) written [{:d}] scalar functions'.format(basename, num))

def read_dolfin_scalar_cg1(basename, VV = None, mesh = None, dolf = True):
	print('read_dolfin_scalar_cg1({:s})'.format(basename))
	h5_file = h5py.File(basename+'.h5', 'r')
	grp = h5_file['/scalar']
	ret = []
	
	if not dolf:
		key = '{:d}'.format(ii)
		ii = 0
		while key in grp:
			dset = grp[key]
			ret.append(dset[:])
			ii += 1
			key = '{:d}'.format(ii)
	else:
		if VV is None:
			if mesh is None:
				mesh = Mesh()
				ff = XDMFFile(comm, basename+'.xdmf')
				ff.read(mesh)
				del ff
			VV = FunctionSpace(mesh, 'CG', 1)
		index_map = dof_to_vertex_map(VV)
		ii = 0
		key = '{:d}'.format(ii)
		while key in grp:
			dset = grp[key]
			ret.append(Function(VV, name = 'u'))
			ret[-1].vector().set_local(dset[:][index_map])
			ii += 1
			key = '{:d}'.format(ii)

	h5_file.close()
	print('read_dolfin_scalar_cg1({:s}) read [{:d}] scalar functions'.format(basename, ii))
	return ret

def write_dolfin_vector_cg1(basename, vector_functions, scale_to_one = False):
	print('write_dolfin_vector_cg1({:s})'.format(basename))
	real_dirname = os.path.dirname(basename)
	sa_utils.makedirs_norace(real_dirname)
	real_basename = os.path.basename(basename)

	mesh = vector_functions[0].function_space().mesh()
	mesh.init()
	basedim = mesh.geometry().dim()
	num_vertices = mesh.num_vertices()
	num_cells = mesh.num_cells()

	h5_file = h5py.File(basename+'.h5', 'w')
	h5_vertices = h5_file.create_dataset('/vertices', (num_vertices, basedim), data = mesh.coordinates(), compression = my_compression)
	h5_cells = h5_file.create_dataset('/cells', (num_cells, basedim+1), data = numpy.array(mesh.cells(), dtype = numpy.uintp), compression = my_compression)
	h5_grp = h5_file.create_group('/vector')
	num = len(vector_functions)
	index_map = vertex_to_dof_map(vector_functions[0].function_space())
	if scale_to_one:
		for ii, uu in enumerate(vector_functions):
			data = uu.vector()[index_map]
			umin = numpy.min(data); umax = numpy.max(data)
			if umax < -umin:
				data /= -umin
			elif 0 < umax:
				data /= umax
			h5_grp.create_dataset('{:d}'.format(ii), (num_vertices*basedim, ), data = data, compression = my_compression)
	else:
		for ii, uu in enumerate(vector_functions):
			h5_grp.create_dataset('{:d}'.format(ii), (num_vertices*basedim, ), data = uu.vector()[index_map], compression = my_compression)
	h5_file.close()
	
	vector_file = open(basename+'.xdmf', 'w')
	string = r" ""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "cells" GridType = "Uniform">
			<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
			</Geometry>
		</Grid>" "".format(
		num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
		num_cells, basedim+1, real_basename, 
		"XYZ" if basedim == 3 else "XY", 
		num_vertices, basedim, real_basename
	)
	string += " ""
		<Grid Name = "vector" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{:d}">
					" "".format(num)
	for ii in range(num):
		string += " {:d}".format(ii)
	string += r" ""
				</DataItem>
			</Time>" ""
	for ii in range(num):
		string += r" ""
			<Grid Name = "grid_{:d}" GridType = "Uniform">
				<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
				</Geometry>" "".format(ii, 
									  num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
									  num_cells, basedim+1, real_basename, 
									  "XYZ" if basedim == 3 else "XY", 
									  num_vertices, basedim, real_basename)
		for jj in range(basedim):
			string += r" ""
				<Attribute Name = "u_{:d}" AttributeType = "Scalar" Center = "Node">
					<DataItem ItemType = "HyperSlab" Dimensions = "{:d}">
						<DataItem Dimensions = "3 1" Format = "XML">
							{:d} 
							{:d}
							{:d}
						</DataItem>
						<DataItem Format = "HDF" Dimensions = "{:d}">{:s}.h5:/vector/{:d}</DataItem>
					</DataItem>
				</Attribute>" "".format(jj, 
									   num_vertices, 
									   jj, 
									   basedim, 
									   num_vertices, 
									   num_vertices*basedim, real_basename, ii)
		string += r" ""
			</Grid>" ""
	string += r" ""
		</Grid>
	</Domain>
</Xdmf>" ""
	vector_file.write(string)
	vector_file.close()
	print('write_dolfin_vector_cg1({:s}) written [{:d}] vector functions'.format(basename, num))

def read_dolfin_vector_cg1(basename, VV = None, mesh = None, dolf = True):
	print('read_dolfin_vector_cg1({:s})'.format(basename))
	h5_file = h5py.File(basename+'.h5', 'r')
	grp = h5_file['/vector']
	ret = []
	
	if not dolf:
		key = '{:d}'.format(ii)
		ii = 0
		while key in grp:
			dset = grp[key]
			ret.append(dset[:])
			ii += 1
			key = '{:d}'.format(ii)
	else:
		if VV is None:
			if mesh is None:
				mesh = Mesh()
				ff = XDMFFile(comm, basename+'.xdmf')
				ff.read(mesh)
				del ff
			basedim = mesh.geometry().dim()
			VV = VectorFunctionSpace(mesh, 'CG', 1, basedim)
		index_map = dof_to_vertex_map(VV)
		ii = 0
		key = '{:d}'.format(ii)
		while key in grp:
			dset = grp[key]
			ret.append(Function(VV, name = 'u'))
			ret[-1].vector().set_local(dset[:][index_map])
			ii += 1
			key = '{:d}'.format(ii)

	h5_file.close()
	print('read_dolfin_vector_cg1({:s}) read [{:d}] vector functions'.format(basename, ii))
	print('[{:d}] [{:s}] vector functions read'.format(ii, basename))
	return ret

def write_dolfin_mesh_functions(mesh, basename, *, cell_functions = None, facet_functions = None, transform = None):
	print('write_dolfin_mesh_functions({:s})'.format(basename))
	real_dirname = os.path.dirname(basename)
	sa_utils.makedirs_norace(real_dirname)
	real_basename = os.path.basename(basename)
	mesh.init()
	
	basedim = mesh.geometry().dim()
	num_vertices = mesh.num_vertices()
	num_cells = mesh.num_cells()
	num_facets = mesh.num_facets()

	h5_file = h5py.File(basename+'.h5', 'w')
	if transform is None:
		h5_file.create_dataset('/vertices', (num_vertices, basedim), data = mesh.coordinates(), compression = my_compression)
	else:
		h5_file.create_dataset('/vertices', (num_vertices, basedim), data = transform(mesh.coordinates()), compression = my_compression)
	h5_file.create_dataset('/cells', (num_cells, basedim+1), data = numpy.array(mesh.cells(), dtype = numpy.uintp), compression = my_compression)
	h5_file.create_dataset('/facets', (num_facets, basedim), data = numpy.array([facet.entities(0) for facet in facets(mesh)], dtype = numpy.uintp), compression = my_compression)
	if cell_functions is not None:
		group = h5_file.create_group('/cell_functions')
		for ii, ff in enumerate(cell_functions):
			group.create_dataset('{:d}'.format(ii), (num_cells, 1), data = ff.array(), compression = my_compression)
	if facet_functions is not None:
		group = h5_file.create_group('/facet_functions')
		for ii, ff in enumerate(facet_functions):
			group.create_dataset('{:d}'.format(ii), (num_facets, 1), data = ff.array(), compression = my_compression)
	h5_file.close()
	del h5_file

	if cell_functions is not None:
		print('	writing {:d} cell functions'.format(len(cell_functions)))
		cell_functions_file = open(basename+'_cell_functions.xdmf', 'w')
		cell_functions_file.write(r" ""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "cells" GridType = "Uniform">
			<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
			</Geometry>
		</Grid>" "".format(
			num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
			num_cells, basedim+1, real_basename, 
			"XYZ" if basedim == 3 else "XY", 
			num_vertices, basedim, real_basename
		))
		num = len(cell_functions)
		cell_functions_file.write(r" ""
		<Grid Name = "cell_functions" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{:d}">
					" "".format(num))				
		for ii in range(num):
			cell_functions_file.write(" {:d}".format(ii))
		cell_functions_file.write(r" ""
				</DataItem>
			</Time>" "")
		for ii, ff in enumerate(cell_functions):
			cell_functions_file.write(r" ""
			<Grid Name = "grid_{:d}" GridType = "Uniform">
				<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
				</Topology>
				<Geometry GeometryType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "cell_function" AttributeType = "Scalar" Center = "Cell">
					<DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/cell_functions/{:d}</DataItem>
				</Attribute>
			</Grid>" "".format(
				ii,
				num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
				num_cells, basedim+1, real_basename, 
				"XYZ" if basedim == 3 else "XY", 
				num_vertices, basedim, real_basename,
				num_cells, real_basename, ii
			))
		cell_functions_file.write(r" ""
		</Grid>
	</Domain>
</Xdmf>" "")
		cell_functions_file.close()
		del cell_functions_file
		print('	written {:d} cell functions'.format(len(cell_functions)))

	if facet_functions is not None:
		print('	writing {:d} facet functions'.format(len(facet_functions)))
		facet_functions_file = open(basename+'_facet_functions.xdmf', 'w')
		facet_functions_file.write(r" ""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
	<Domain>
		<Grid Name = "cells" GridType = "Uniform">
			<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
			</Topology>
			<Geometry GeometryType = "{:s}">
				<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
			</Geometry>
		</Grid>" "".format(
			num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
			num_cells, basedim+1, real_basename, 
			"XYZ" if basedim == 3 else "XY", 
			num_vertices, basedim, real_basename
		))
		num = len(facet_functions)
		facet_functions_file.write(r" ""
		<Grid Name = "facet_functions" GridType = "Collection" CollectionType = "Temporal">
			<Time TimeType = "List">
				<DataItem Format = "XML" Dimensions = "{:d}">
					" "".format(num))
		for ii in range(num):
			facet_functions_file.write(" {:d}".format(ii))
		facet_functions_file.write(r" ""
				</DataItem>
			</Time>" "")
		for ii, ff in enumerate(facet_functions):
			facet_functions_file.write(" ""
			<Grid Name = "grid_{:d}" GridType = "Uniform">
				<Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/facets</DataItem>
				</Topology>
				<Geometry GeometryType = "{:s}">
					<DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
				</Geometry>
				<Attribute Name = "facet_function" AttributeType = "Scalar" Center = "Cell">
					<DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/facet_functions/{:d}</DataItem>
				</Attribute>
			</Grid>" "".format(
				ii,
				num_facets, 'Triangle' if basedim == 3 else 'PolyLine" NodesPerElement = "2', 
				num_facets, basedim, real_basename, 
				"XYZ" if basedim == 3 else "XY", 
				num_vertices, basedim, real_basename,
				num_facets, real_basename, ii
			))
		facet_functions_file.write(r" ""
		</Grid>
	</Domain>
</Xdmf>" "")
		facet_functions_file.close()
		del facet_functions_file
		print('	written {:d} cell functions'.format(len(facet_functions)))
	print('write_dolfin_mesh_functions({:s}) end'.format(basename))

def read_dolfin_mesh_functions(basename, dolf = True):
	print('read_dolfin_mesh_functions({:s}) start'.format(basename))
	h5_file = h5py.File(basename+'.h5', 'r')
	h5_vertices = h5_file['/vertices']
	shape_vertices = h5_vertices.shape
	h5_cells = h5_file['/cells']
	shape_cells = h5_cells.shape
	basedim = shape_vertices[1]
	assert(basedim == shape_cells[1]-1)

	ret = dict()
	
	if not dolf:
		ret['vertices'] = h5_vertices[:]
		ret['cells'] = h5_cells[:]
		if '/cell_functions' in h5_file:
			h5_cell_functions = []
			grp = h5_file['/cell_functions']
			ii = 0
			key = '{:d}'.format(ii)
			while key in grp:
				h5_cell_functions.append(grp[key][:])
				ii += 1
				key = '{:d}'.format(ii)

		if '/cell_functions' in h5_file:
			h5_cell_functions = []
			grp = h5_file['/cell_functions']
			ii = 0
			key = '{:d}'.format(ii)
			while key in grp:
				h5_cell_functions.append(grp[key][:])
				ii += 1
				key = '{:d}'.format(ii)
			ret['cell_functions'] = h5_cell_functions

		if '/facet_functions' in h5_file:
			h5_facet_functions = []
			grp = h5_file['/facet_functions']
			ii = 0
			key = '{:d}'.format(ii)
			while key in grp:
				h5_facet_functions.append(grp[key][:])
				ii += 1
				key = '{:d}'.format(ii)
			ret['facet_functions'] = h5_facet_functions
	else:
		mesh = Mesh()
		if '/cell_functions' in h5_file:
			ff = XDMFFile(comm, basename+'_cell_functions.xdmf')
		else:
			ff = XDMFFile(comm, basename+'_facet_functions.xdmf')
		ff.read(mesh)
		del ff
		ret['mesh'] = mesh
		
		if '/cell_functions' in h5_file:
			h5_cell_functions = []
			grp = h5_file['/cell_functions']
			ii = 0
			key = '{:d}'.format(ii)
			while key in grp:
				mf = MeshFunction('size_t', mesh, basedim)
				mf.set_values(grp[key][:])
				h5_cell_functions.append(mf)
				ii += 1
				key = '{:d}'.format(ii)
			ret['cell_functions'] = h5_cell_functions
			print('	read {:d} cell functions'.format(ii))

		h5_facet_functions = None
		if '/facet_functions' in h5_file:
			h5_facet_functions = []
			grp = h5_file['/facet_functions']
			ii = 0
			key = '{:d}'.format(ii)
			while key in grp:
				mf = MeshFunction('size_t', mesh, basedim-1)
				mf.set_values(grp[key][:])
				h5_facet_functions.append(mf)
				ii += 1
				key = '{:d}'.format(ii)
			ret['facet_functions'] = h5_facet_functions
			print('	read {:d} facet functions'.format(ii))

	h5_file.close()
	print('read_dolfin_mesh_functions({:s}) end'.format(basename))
	return ret

def test_new_h5():
	print('dolfin stuff')
	mesh2 = RectangleMesh(Point(-1, -1), Point(1, 1), 20, 20)
	mc2 = MeshFunction('size_t', mesh2, 2, 0)
	mc2.array()[0] = 1
	mc2.array()[-1] = 1
	mf2 = MeshFunction('size_t', mesh2, 1, 0)
	mf2.array()[0] = 1
	mf2.array()[-1] = 1
	cf2 = MeshFunction('double', mesh2, 2, 0)
	cf2.set_values(rnd.random(mesh2.num_cells()))
	print('Dolfin stuff done')
	print('write mesh')
	write_dolfin_mesh(mesh2, 'test_2d', cell_function = mc2, facet_function = mf2, cell_coeff = cf2)
	print('read mesh')

	ret = read_dolfin_mesh('test_2d', dolf = False)
	print('write mesh again')
	ret = read_dolfin_mesh('test_2d', dolf = True)
	write_dolfin_mesh(ret['mesh'], 'test_2d_2', cell_function = ret['cell_function'], facet_function = ret['facet_function'], cell_coeff = ret['cell_coeff'])
	
	write_dolfin_mesh_functions(mesh2, 'test_2d_3', cell_functions = [mc2, mc2], facet_functions = [mf2, mf2])
	ret = read_dolfin_mesh_functions('test_2d_3')

	print('functions')
	UU2 = FunctionSpace(mesh2, 'CG', 1)
	u0 = interpolate(Expression('x[0]*x[1]', degree = 2), UU2)
	VV2 = VectorFunctionSpace(mesh2, 'CG', 1, 2)
	v0 = interpolate(Expression(('0', 'x[0]*x[1]'), degree = 2), VV2)
	uus = []
	vvs = []
	for ii in range(1, 6):
		uus.append(Function(UU2))
		uus[-1].vector().set_local(u0.vector().get_local()**ii)
		vvs.append(Function(VV2))
		vvs[-1].vector().set_local(v0.vector().get_local()**ii)
	
	print('writing scalar functions')
	write_dolfin_scalar_cg1('test_2d_scalar', uus)
	print('reading scalar functions')
	ret = read_dolfin_scalar_cg1('test_2d_scalar')
	print('writing scalar functions again')
	write_dolfin_scalar_cg1('test_2d_scalar_2', ret)
	print('writing scalar functions')
	write_dolfin_vector_cg1('test_2d_vector', vvs)
	print('reading scalar functions')
	ret = read_dolfin_vector_cg1('test_2d_vector')
	print('writing scalar functions again')
	write_dolfin_vector_cg1('test_2d_vector_2', ret)
	print('done')

if __name__ == '__main__':
	test_new_h5()
"""