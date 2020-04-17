"""
3D meshing using mostly Netgen
"""

import os
from datetime import datetime

import dolfin
from dolfin_utils.meshconvert import meshconvert
import netgen.csg as netgen_csg
import numpy

from ..helpers import io
from .base import CSG, CSGCollection, csg_eps

class CSG3D(CSG):
	@classmethod
	def getCollection(cls):
		return CSG3DCollection()

class HalfPlane(CSG3D):
	def __init__(self, zero, normal, *, eps = csg_eps):
		value = numpy.array(zero).dot(numpy.array(normal))
		super().__init__(netgen_csg.Plane(netgen_csg.Pnt(*zero), netgen_csg.Vec(*normal)), boundaries = [
			dolfin.CompiledSubDomain(
				'on_boundary && near(x[0]*n0 + x[1]*n1 + x[2]*n2, value, eps)',
				n0 = normal[0], n1 = normal[1], n2 = normal[2], value = value, eps = eps
			)
		])

class Sphere(CSG3D):
	def __init__(self, center, radius, *, eps = csg_eps):
		eps *= radius
		super().__init__(netgen_csg.Sphere(netgen_csg.Pnt(*center), radius), boundaries = [
			dolfin.CompiledSubDomain(
				'on_boundary && near((x[0]-c0)*(x[0]-c0) + (x[1]-c1)*(x[1]-c1) + (x[2]-c2)*(x[2]-c2), rr, eps)',
				c0 = center[0], c1 = center[1], c2 = center[2], rr = radius*radius, eps = eps
			)
		])


class Cylinder(CSG3D):
	def __init__(self, pointa, pointb, radius, *, eps = csg_eps):
		unit = 1.*numpy.array(pointb) - numpy.array(pointa)
		unit /= numpy.linalg.norm(unit)
		eps *= radius
		super().__init__(netgen_csg.Cylinder(netgen_csg.Pnt(*pointa), netgen_csg.Pnt(*pointb), radius), boundaries = [
			dolfin.CompiledSubDomain(
				'on_boundary && near((x[0]-p0)*(x[0]-p0) + (x[1]-p1)*(x[1]-p1) + (x[2]-p2)*(x[2]-p2) - ((x[0]-p0)*u0 + (x[1]-p1)*u1 + (x[2]-p2)*u2)*((x[0]-p0)*u0 + (x[1]-p1)*u1 + (x[2]-p2)*u2), rr, eps)',
				u0 = unit[0], u1 = unit[1], u2 = unit[2], p0 = pointa[0], p1 = pointa[1], p2 = pointa[2], rr = radius*radius, eps = eps
			)
		])

class OrthoBrick(CSG3D):
	def __init__(self, pointa, pointb, *, eps = csg_eps):
		eps *= numpy.linalg.norm(numpy.array(pointb)-numpy.array(pointa))
		super().__init__(netgen_csg.OrthoBrick(netgen_csg.Pnt(*pointa), netgen_csg.Pnt(*pointb)), boundaries = [
			dolfin.CompiledSubDomain('on_boundary && near(x[0], xx, eps)', xx = pointa[0], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[0], xx, eps)', xx = pointb[0], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[1], xx, eps)', xx = pointa[1], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[1], xx, eps)', xx = pointb[1], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[2], xx, eps)', xx = pointa[2], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[2], xx, eps)', xx = pointb[2], eps = eps)
		])

class CSG3DCollection(CSGCollection):
	def __init__(self):
		self.collection = netgen_csg.CSGeometry()
		self.boundaries = []

	def add(self, csg3d, *, add_boundaries = True):
		self.collection.Add(csg3d.geometry)
		if add_boundaries:
			self.boundaries.extend(csg3d.boundaries)
		
	def generateMesh(self, maxh):
		mesh = self.collection.GenerateMesh(maxh = maxh)
		mesh.GenerateVolumeMesh()
		tmp_name = datetime.now().isoformat()
		mesh.Export(f'{tmp_name}.msh', 'Gmsh2 Format')
		meshconvert.convert2xml(f'{tmp_name}.msh', f'{tmp_name}.xml')
		os.remove(f'{tmp_name}.msh')
		os.remove(f'{tmp_name}_facet_region.xml')
		self.mesh = dolfin.Mesh(f'{tmp_name}.xml')
		os.remove(f'{tmp_name}.xml')
		self.domains = dolfin.MeshFunction('size_t', self.mesh, f'{tmp_name}_physical_region.xml')
		os.remove(f'{tmp_name}_physical_region.xml')
		self.domains.array()[:] -= self.domains.array().min()
		self.facets = dolfin.MeshFunction('size_t', self.mesh, 2, 0)
		for ii, subdomain in enumerate(self.boundaries):
			subdomain.mark(self.facets, ii+1)

class Layers(CSG3DCollection):
	def __init__(self, boundary, low, high, layers = 1, *, elements_per_layer = 2):
		super().__init__()
		self.boundaries = boundary.boundaries
		self.layers = layers
		self.elements_per_layer = elements_per_layer
		hh = (numpy.array(high) - numpy.array(low))/(self.layers*self.elements_per_layer)
		lower = low + hh
		super().add(boundary*HalfPlane(lower, hh), add_boundaries = False)
		for ii in range(1, self.layers*self.elements_per_layer - 1):
			super().add(boundary*(HalfPlane(lower+hh, hh)-HalfPlane(lower, hh)), add_boundaries = False)
			lower += hh
		super().add(boundary*HalfPlane(lower, -hh), add_boundaries = False)

	def generateMesh(self, maxh):
		super().generateMesh(maxh)
		self.domains.array()[:] //= self.elements_per_layer

"""
from dolfin import *
import numpy as np
import scipy
import scipy.linalg as la
import gc
import sys, os
import multiprocessing
import ctypes

import sa_hdf5
import sa_utils
from sa_utils import comm

hx = 20.
hy = 10.
hz = 2.

xdim = 60
ydim = 220
zdim = 85
#xdim = 15
#ydim = 55
#zdim = 85

parameters["refinement_algorithm"] = "plaza_with_parent_facets"

low_array = np.array([0., 0., 0.])
high_array = np.array([xdim*hx, ydim*hy, zdim*hz])

myeps = 1e-14

left = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[0], low_array[0], eps=myeps*high_array[0])) 
right = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[0], high_array[0], eps=myeps*high_array[0])) 
front = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[1], low_array[1], eps=myeps*high_array[1])) 
back = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[1], high_array[1], eps=myeps*high_array[1])) 
bottom = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[2], low_array[2], eps=myeps*high_array[2])) 
top = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[2], high_array[2], eps=myeps*high_array[2])) 

wells = np.array([[600., 1100.], 
				  [0., 0.], 
				  [1200., 0.], 
				  [1200., 2200.], 
				  [0., 2200.]])

debug = False

well_radius = 0.4

def intersects_well(cell, eps):
	pts = cell.get_vertex_coordinates().reshape((-1, 3))[:, :-1]
	for well in wells:
		coords = pts-well
		min_dist = np.min([la.norm(pt) for pt in coords])
		max_dist = np.max([la.norm(pt) for pt in coords])
		if min_dist < well_radius and max_dist > well_radius*(1.+eps):
			return True
	return False

def in_well(cell):
	pts = cell.get_vertex_coordinates().reshape((-1, 3))[:, :-1]
	for well in wells:
		coords = pts-well
		min_dist = np.min([la.norm(pt) for pt in coords])
		max_dist = np.max([la.norm(pt) for pt in coords])
		if max_dist < well_radius:
			return True
	return False

def in_outflow(xx):
	for well in wells[1:]:
		if la.norm(well-xx[:2]) < well_radius:
			return True
	return False

def in_inflow(xx):
	if la.norm(wells[0]-xx[:2]) < well_radius:
		return True
	return False

class TopOutFlow(SubDomain):
	def inside(self, xx, on_boundary):
		if on_boundary:
			return near(xx[2], high_array[2], eps=high_array[2]*myeps) and in_outflow(xx)
		return False

class BottomOutFlow(SubDomain):
	def inside(self, xx, on_boundary):
		if on_boundary:
			return near(xx[2], low_array[2], eps=high_array[2]*myeps) and in_outflow(xx)
		return False

class TopInFlow(SubDomain):
	def inside(self, xx, on_boundary):
		if on_boundary:
			return near(xx[2], low_array[2], eps=high_array[2]*myeps) and in_inflow(xx)
		return False

class BottomInFlow(SubDomain):
	def inside(self, xx, on_boundary):
		if on_boundary:
			return near(xx[2], low_array[2], eps=high_array[2]*myeps) and in_inflow(xx)
		return False

top_out = TopOutFlow()
bottom_out = BottomOutFlow()
top_in = TopInFlow()
bottom_in = BottomInFlow()

def to_idx(xx, yy, zz):
	ix = int(xx/hx)
	iy = int(yy/hy)
	iz = int(zz/hz)
	return iz*xdim*ydim+iy*xdim+ix

def create_mesh(low=Point(*low_array), high=Point(*high_array), xdim=xdim, ydim=ydim, zdim=zdim, wells=False, logger=None):
	if logger is None:
		info = print
	else:
		info = logger.info

	info('Creating meshes')
	mesh = BoxMesh(low, high, xdim, ydim, zdim)
	info('Mesh created')

	well_limit = 0.2
	cpu_count = multiprocessing.cpu_count()
	mf = MeshFunction('size_t', mesh, 3, 0)

	mfs = []
	meshes = []
	if wells:
		count = 0
		which = None
		while True:
			num_cells = mesh.num_cells()
#		   num = mesh.num_cells()
			if which is not None:
				num = len(which)
				def populate(shared, low, high):
					for idx in range(low, high):
						real_idx = which[idx]
						cell = Cell(mesh, real_idx)
						if intersects_well(cell, well_limit):
							shared[idx] = True
						else:
							shared[idx] = False
						del cell
			else:
				num = mesh.num_cells()
				def populate(shared, low, high):
					for idx in range(low, high):
						cell = Cell(mesh, idx)
						if intersects_well(cell, well_limit):
							shared[idx] = True
						else:
							shared[idx] = False
						del cell
			shared = multiprocessing.Array(ctypes.c_size_t, num, lock=False)
			info("Refine loop [{:d}], checking [{:d}/{:d}={:.2f}%] cells.".format(count, num, num_cells, num*100./num_cells))
			processes = []
			block = int(np.ceil(num/cpu_count))
			block_low = 0
			block_high = np.min([block_low+block, num])
			for cpu in range(cpu_count):
				if block_low >= block_high:
					break
				processes.append(multiprocessing.Process(target=populate, args=(shared, block_low, block_high)))
				processes[-1].start()
				block_low = block_high
				block_high = np.min([block_low+block, num])
			for cpu, proc in enumerate(processes):
				proc.join()
				processes[cpu] = None
			del processes
			mf_array = np.array(shared)
			del shared
			positive = len(np.where(mf_array > 0)[0])
			info("	[{:d}/{:d}={:.2f}%] of previous [{:d}/{:d}={:.2f}%] of total need refining".format(positive, num, positive*100./num, positive, num_cells, positive*100./num_cells))
			if not positive:
				break
			mf.array()[:] = 0
			mf.array()[which] = mf_array
			mfs.append(mf)
			info("	mesh function for refinement created")

			marker = MeshFunction('bool', mesh, 3, False)
			marker.array()[which] = mf_array
			del which, mf_array
			if debug:
				mf_file = XDMFFile(comm, 'mf_'+str(count).zfill(2)+'.xdmf')
				mf_file.write(marker)
				del mf_file

			meshes.append(mesh)
			mesh = refine(mesh, marker)
			del marker
			info("	refined")
			mf = adapt(mf, mesh)
			which = np.where(mf.array() > 0)[0]
			gc.collect()
			count += 1

	info('Mesh created')

	facets = FacetFunction('size_t',mesh)
	left.mark(facets, 1); right.mark(facets, 2)
	front.mark(facets, 3); back.mark(facets, 4)
	bottom.mark(facets, 5); top.mark(facets, 6)
	bottom_in.mark(facets, 101); top_in.mark(facets, 102)
	bottom_out.mark(facets, 103); top_out.mark(facets, 104)

	return mesh, facets

def create_phi(mesh, wells=False, logger=None):
	if logger is None:
		info = print
	else:
		info = logger.info

	phi_data = np.genfromtxt('spe_phi.dat').flatten()

	info('Preparing mesh function')
	num = mesh.num_cells()
	shared = multiprocessing.Array(ctypes.c_double, num, lock=False)
	if wells:
		def populate(shared, low, high):
			info('populate ['+str(low)+'-'+str(high)+'] starting')
			for idx in range(low, high):
				cell = Cell(mesh, idx)
				pt = cell.midpoint()
				shared[idx] = phi_data[to_idx(pt.x(), pt.y(), pt.z())]
				if in_well(cell):
					shared[idx] = 1.
				del cell
			info('populate ['+str(low)+'-'+str(high)+'] done')
	else:
		def populate(shared, low, high):
			info('populate ['+str(low)+'-'+str(high)+'] starting')
			for idx in range(low, high):
				cell = Cell(mesh, idx)
				pt = cell.midpoint()
				shared[idx] = phi_data[to_idx(pt.x(), pt.y(), pt.z())]
				del cell
			info('populate ['+str(low)+'-'+str(high)+'] done')
	processes = []
	cpu_count = multiprocessing.cpu_count()
	block = int(np.ceil(num/cpu_count))
	block_low = 0
	block_high = np.min([block_low+block, num])
	for cpu in range(cpu_count):
		if block_low >= block_high:
			break
		processes.append(multiprocessing.Process(target=populate, args=(shared, block_low, block_high)))
		processes[-1].start()
		info('batch ['+str(cpu+1)+'/'+str(cpu_count)+']: ['+str(block_low)+'-'+str(block_high)+'] started')
		block_low = block_high
		block_high = np.min([block_low+block, num])
	for cpu, proc in enumerate(processes):
		proc.join()
		processes[cpu] = None
		info('batch ['+str(cpu+1)+'/'+str(cpu_count)+'] joined')
	mf_array = np.array(shared)
	del shared
	gc.collect()

	mf = MeshFunction('double', mesh, 3, 0)
	mf.set_values(mf_array)
	info('Mesh function created')

	del mf_array, info
	gc.collect()
	return mf

def create_patches(mesh, mf, alpha=1.25, beta=2.0, patch_h=60., prefix='anisotropic', logger=None):
	if logger is None:
		info = print
	else:
		info = logger.info
	global_array = mf.array()

	patch_numbers = np.array(np.ceil(high_array/patch_h), dtype=int)
	patch_total = np.prod(patch_numbers)
	patch_fill = int(np.log(patch_total)/np.log(10.))+1
	patch_hs = high_array/patch_numbers
	patch_extended = patch_hs*alpha
	patch_inner = patch_extended*0.5
	info('Creating patches, numbers '+str(patch_numbers)+', hs '+str(patch_hs)+', total ['+str(patch_total)+']')

	sa_utils.makedirs_norace(prefix+'/'+prefix+'_patch_descriptors')
	ff = open(prefix+'/'+prefix+'_patch_descriptors/0.csv', 'w')
	ff.write('idx, left, right, front, back, bottom, top\n')

	dirname = prefix+'/'+prefix+'_0_patches/0'
	sa_utils.makedirs_norace(dirname)

	global_V0 = FunctionSpace(mesh, 'DG', 0)
	u0 = Function(global_V0)
	u0.vector().set_local(mf.array())

	info('DG function created')

	def mesh_patch(kk, jj, ii, count, qq):
		patch_z = patch_hs[2]*(kk+0.5)
		patch_y = patch_hs[1]*(jj+0.5)
		patch_x = patch_hs[0]*(ii+0.5)
		patch_name = dirname+'/patch_'+str(count).zfill(patch_fill)
		ext_name = patch_name+'_2.0'
		center = np.array([patch_x, patch_y, patch_z])
		extended = AutoSubDomain(lambda xx, on_boundary: (np.abs(xx - center) < patch_extended+myeps).all()) 
		extended_mesh = SubMesh(mesh, extended)
		del extended
		info('['+str(count)+'] cut out')

		inner = AutoSubDomain(lambda xx, on_boundary: (np.abs(xx - center) < patch_inner+myeps).all())
		inner_marker = MeshFunction('bool', extended_mesh, 3, False)
		inner.mark(inner_marker, True)

		extended_mesh = refine(extended_mesh, inner_marker)
		del inner_marker
		info('['+str(count)+'] refined')

		mf = MeshFunction('size_t', extended_mesh, 3, 1)
		mf.rename('cell_function', 'label')
		inner.mark(mf, 0)
		info('['+str(count)+'] cell function refined')

		local_V0 = FunctionSpace(extended_mesh, 'DG', 0)
		v0 = interpolate(u0, local_V0)
		coefficient = MeshFunction('double', extended_mesh, 3, 0)
		coefficient.set_values(v0.vector().array())
		info('['+str(count)+'] extended phi created')

		nodes = extended_mesh.coordinates()
		low = np.min(nodes, axis=0)
		high = np.max(nodes, axis=0)
		extended_left = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[0], low[0], eps=(high[0]-low[0])*myeps))
		extended_right = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[0], high[0], eps=(high[0]-low[0])*myeps))
		extended_front = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[1], low[1], eps=(high[1]-low[1])*myeps))
		extended_back = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[1], high[1], eps=(high[1]-low[1])*myeps))
		extended_bottom = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[2], low[2], eps=(high[2]-low[2])*myeps))
		extended_top = AutoSubDomain(lambda xx, on_boundary: on_boundary and near(xx[2], high[2], eps=(high[2]-low[2])*myeps))
		del nodes
		facets = MeshFunction('size_t', extended_mesh, 2, 0)
		extended_left.mark(facets, 7); extended_right.mark(facets, 8)
		extended_front.mark(facets, 9); extended_back.mark(facets, 10)
		extended_bottom.mark(facets, 11); extended_top.mark(facets, 12)
		del extended_left, extended_right, extended_front, extended_back, extended_bottom, extended_top
		left.mark(facets, 1); right.mark(facets, 2)
		front.mark(facets, 3); back.mark(facets, 4)
		bottom.mark(facets, 5); top.mark(facets, 6)
		bottom_in.mark(facets, 101); top_in.mark(facets, 102)
		bottom_out.mark(facets, 103); top_out.mark(facets, 104)
		info('['+str(count)+'] extended facets created')
	   
		sa_hdf5.write_dolfin_mesh(extended_mesh, ext_name, cell_function=mf, cell_coeff=coefficient, facet_function=facets)
#			   inner_mesh = SubMesh(extended_mesh, inner)
#			   del inner, extended_mesh
#			   File(patch_name+'.xml') << inner_mesh
#			   mesh_file = XDMFFile(comm, patch_name+'.xdmf')
#			   mesh_file.write(inner_mesh)
	   
		del facets, low, high, coefficient, v0, local_V0, mf, inner, extended_mesh, center, ext_name, patch_name, patch_x, patch_y, patch_z
		count += 1

		info('['+str(count)+'] finished')
		gc.collect()
		qq.put(None)

	old = sa_utils.get_time()
	out_q = multiprocessing.Queue()
	processes = []
	count = 0
	for kk in range(patch_numbers[2]):
		for jj in range(patch_numbers[1]):
			for ii in range(patch_numbers[0]):
				processes.append(multiprocessing.Process(target=mesh_patch, args=(kk, jj, ii, count, out_q)))
				count += 1

	pt_length = len(processes)
	block = multiprocessing.cpu_count()
	block_low = 0; block_high = np.min([block, pt_length])
	while(block_low < block_high):
		for kk in range(block_low, block_high):
			processes[kk].start()
		for kk in range(block_low, block_high):
			out_q.get()
		for kk in range(block_low, block_high):
			processes[kk].join()
			processes[kk] = None
			logger.info('['+str(kk)+'] joined')
		block_low = block_high
		block_high = np.min([block_high+block, pt_length])
	new = sa_utils.get_time()
	info('[{:d}] patches meshed with [{:d}] cpus in [{:s}]'.format(count,block,sa_utils.human_time(new-old)))

	info('Finished creating patches')
	del info

if __name__ == '__main__':
	prefix='nowells'
	mesh, facets = create_mesh()
	phi = create_phi(mesh)
	sa_hdf5.write_dolfin_mesh(mesh,prefix,cell_coeff=phi,facet_function=facets)
#   prefix='anisotropic_wells'
#   create_mesh(prefix=prefix)
#   mesh = load_mesh(prefix=prefix)
#   create_phi(mesh, prefix=prefix, wells=True)
#   del mesh
#   mesh, phi = load_mesh_phi(prefix=prefix)
#   create_patches(mesh, phi, patch_h=200., prefix=prefix)

#   prefix='20x4'
#   create_mesh(prefix=prefix, xdim=int(np.ceil(high_array[0]/20)), ydim=int(np.ceil(high_array[1]/20)), zdim=int(np.ceil(high_array[2]/4)))
#   mesh = load_mesh(prefix=prefix)
#   create_phi(mesh, prefix=prefix, wells=True)
#   del mesh
#   mesh, phi = load_mesh_phi(prefix=prefix)
#   create_patches(mesh, phi, patch_h=200., prefix=prefix)

#   prefix='10x2'
#   create_mesh(prefix=prefix, xdim=int(np.ceil(high_array[0]/10)), ydim=int(np.ceil(high_array[1]/10)), zdim=int(np.ceil(high_array[2]/2)))
#   mesh = load_mesh(prefix=prefix)
#   create_phi(mesh, prefix=prefix, wells=True)
#   del mesh
#   mesh, phi = load_mesh_phi(prefix=prefix)
#   create_patches(mesh, phi, patch_h=200., prefix=prefix)

#   prefix='5x2'
#   create_mesh(prefix=prefix, xdim=int(np.ceil(high_array[0]/5)), ydim=int(np.ceil(high_array[1]/5)), zdim=int(np.ceil(high_array[2]/2)))
#   mesh = load_mesh(prefix=prefix)
#   create_phi(mesh, prefix=prefix, wells=True)
#   del mesh
#   mesh, phi = load_mesh_phi(prefix=prefix)
#   create_patches(mesh, phi, patch_h=200., prefix=prefix)

#   basename = 'spe/10x2/10x2_0_patches/0/patch_65'
#   mesh, domains, phi, facets = load_patch(basename)
#   plot(mesh)
#   plot(domains)
#   plot(phi)
#   plot(facets)
#   interactive()

from netgen.csg import *
import dolfin
import numpy as np
import numpy.random as rnd
from dolfin_utils.meshconvert import meshconvert
import os, sys, gc

import logging
import sa_utils
import sa_hdf5

def create_patches(box = np.array([0, 0, 0, 1, 1, 1]), patch_num = 3, patch_nums = None, alpha = 1.25, beta = 2.0, max_resolution = 0.5,
				   num = 6, create_inclusions = False, skip_patches = [], prefix = 'test', logger = None,
				   ldomain = False, corner_refine = 3, hole = False, hole_radius = None, layers = 1, max_refines = 1,
				   elem_per_layer = 3):
	if logger is not None:
		info = logger.info
	else:
		info = print

	basedim = 3
	low = box[:basedim].copy(); high = box[basedim:].copy()
	lengths = high-low
	diameter = np.sqrt(lengths@lengths)
	myeps = sa_utils.myeps*diameter
	center = (high+low)*.5
	info('low {:s}, high {:s}, lengths {:s}, center {:s}, diameter {:.2e}'.format(str(low), str(high), str(lengths), str(center), diameter))

	layer_bricks = []
	layer_hz = lengths[2]/layers
	layer_low = np.array([low[0]-lengths[0], low[1]-lengths[1], low[2]-lengths[2]])
	layer_high = np.array([low[0]+lengths[0], low[1]+lengths[1], low[2]+2*lengths[2]])
	for ii in range(layers-1):
		for jj in range(1,elem_per_layer+1):
			layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(low[0]+lengths[0], low[1]+lengths[1], low[2]+(ii+jj*1./elem_per_layer)*layer_hz)))
		info('layer [{:d}/{:d}], {:s}, {:s}'.format(ii, layers, str(layer_low), str(np.array([low[0]+lengths[0], low[1]+lengths[1], low[2]+ii*layer_hz]))))
	for jj in range(1,elem_per_layer):
		layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(low[0]+lengths[0], low[1]+lengths[1], low[2]+(layers-1+jj*1./elem_per_layer)*layer_hz)))
	layer_bricks.append(OrthoBrick(Pnt(*layer_low), Pnt(*layer_high)))
	sublayers = len(layer_bricks)
	info('layer [{:d}/{:d}], {:s}, {:s}'.format(layers, layers, str(layer_low), str(layer_high)))

	info('{:d} layers, {:d} sublayers, {:d} bricks'.format(layers, sublayers, len(layer_bricks)))

	bc_dict = dict()
	left = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], low[0], eps = myeps))
	bc_dict[1] = left
	right = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[0], high[0], eps = myeps))
	bc_dict[2] = right
	front = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], low[1], eps = myeps))
	bc_dict[3] = front
	back = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[1], high[1], eps = myeps))
	bc_dict[4] = back
	bottom = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], low[2], eps = myeps))
	bc_dict[5] = bottom
	top = dolfin.AutoSubDomain(lambda xx, on: on and dolfin.near(xx[2], high[2], eps = myeps))
	bc_dict[6] = top
	border = dolfin.AutoSubDomain(lambda xx, on: on)
	if ldomain:
		corner_lr = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[0], center[0], eps = myeps))
		bc_dict[7] = corner_lr
		corner_fb = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[1], center[1], eps = myeps))
		bc_dict[8] = corner_fb
		corner_bt = dolfin.AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and dolfin.near(xx[2], center[2], eps = myeps))
		bc_dict[9] = corner_bt
		corner_subdomains = []
		corner_close = 0.1*diameter+myeps
		for ii in range(corner_refine):
			corner_subdomains.append(dolfin.AutoSubDomain((lambda what: lambda xx, on: np.sqrt(xx@xx) < what)(corner_close)))
			corner_close *= 0.5
	if create_inclusions and num:
		info('random inclusions')
		if num:
			number = num*num*num
			info('n = '+str(number))
			inc_radius = 0.5/num
			info('r = '+str(inc_radius))
			nodes = []
			radii = []
			rnd_low = low-0.5*inc_radius
			rnd_high = high+0.5*inc_radius
			width = rnd_high-rnd_low
			while(len(nodes) < number):
				notok = True
				while(notok):
					new = rnd.rand(3)*width+rnd_low
					radius = (0.5+rnd.rand())*inc_radius
					notok = False
					for old, rr in zip(nodes, radii):
						diff = new-old
						if np.sqrt(diff.dot(diff)) < 1.3*(radius+rr):
							notok = True
							break
				nodes.append(new.copy())
				radii.append(radius)
			nodes = np.array(nodes)
			radii = np.array(radii)
			info('found locations for '+str(len(nodes))+' inclusions')
			np.savetxt(prefix+'/'+prefix+'_inclusions.csv', np.hstack((nodes, radii.reshape(len(nodes), 1))), fmt = '%.15e', delimiter = ', ')
			del nodes, radii, number, inc_radius

	nohole_whole = OrthoBrick(Pnt(*low), Pnt(*high))
	if ldomain is True:
		nohole_whole = nohole_whole-OrthoBrick(Pnt(*center), Pnt(*(center+2*(high-center))))
	if num:
		data = np.loadtxt(prefix+'/'+prefix+'_inclusions.csv', delimiter = ', ')
		number = len(data)
	else:
		number = 0
	if number:
		nodes = data[:, :3]
		radii = data[:, 3]

		inclusions = Sphere(Pnt(*nodes[0]), radii[0])
		for kk in range(1, len(nodes)):
			inclusions += Sphere(Pnt(*nodes[kk]), radii[kk])
		nohole_matrix = nohole_whole-inclusions
		nohole_incs = nohole_whole*inclusions
	if hole_radius is not None:
		hole = True
	if hole: 
		if hole_radius is None:
			hole_radius = lengths[1]/9.
		near_hole = dolfin.AutoSubDomain(lambda xx, on: on and np.sqrt((xx[0]-center[0])*(xx[0]-center[0])+(xx[1]-center[1])*(xx[1]-center[1])) < hole_radius+1e4*myeps)
		bc_dict[10] = near_hole

	if patch_nums is None:
		hh = lengths[0]/float(patch_num)
		patch_nums = np.array(np.ceil(lengths/hh), dtype = int)
	hs = lengths/patch_nums
	hs_alpha = hs*alpha*0.5
	hs_beta = hs_alpha*beta

	patches = []
	patches_ext = []
	for kk in range(patch_nums[2]):
		pt_z = low[0]+(0.5+kk)*hs[2]
		for jj in range(patch_nums[1]):
			pt_y = low[1]+(0.5+jj)*hs[1]
			for ii in range(patch_nums[0]):
				pt_x = low[2]+(0.5+ii)*hs[0]
				pt_center = np.array([pt_x, pt_y, pt_z])
				pt_low = pt_center-hs_alpha
				if ldomain and (p_low >= center-myeps).all():
					print('[{:d}, {:d}, {:d}] skipped'.format(ii, jj, kk))
					continue

				patches.append(OrthoBrick(Pnt(*(pt_center-hs_alpha)), Pnt(*(pt_center+hs_alpha))))
				patches_ext.append(OrthoBrick(Pnt(*(pt_center-hs_beta)), Pnt(*(pt_center+hs_beta)))-patches[-1])
	patch_num = len(patches)
	print('[{:d}] patches total'.format(patch_num))
	patch_fill = int(np.log(patch_num)/np.log(10.))+1

	pt_low = dict()
	pt_high = dict()
	pt_inside = dict()

	info('Patch size computations')
	sa_utils.makedirs_norace(prefix+'/'+prefix+'_patch_descriptors')
	ff = open(prefix+'/'+prefix+'_patch_descriptors/0.csv', 'w')
	ff.write('idx, left, right, front, back, bottom, top\n')
	for kk in range(patch_num):
		info(str(kk+1)+'/'+str(patch_num))
		geo = CSGeometry()
		geo.Add(nohole_whole*patches[kk])
		mesh = geo.GenerateMesh(maxh = max_resolution)
		del geo
		mesh.Export('tmp.msh', 'Gmsh2 Format')
		del mesh
		meshconvert.convert2xml('tmp.msh', 'tmp.xml')
		os.remove('tmp.msh')
		os.remove('tmp_facet_region.xml')
		os.remove('tmp_physical_region.xml')

		mesh = dolfin.Mesh('tmp.xml')
		os.remove('tmp.xml')
		nodes = mesh.coordinates()
		del mesh
		pt_low[kk] = np.min(nodes, axis = 0)
		pt_high[kk] = np.max(nodes, axis = 0)
		ff.write('%d, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n'%(kk, pt_low[kk][0], pt_high[kk][0], pt_low[kk][1], pt_high[kk][1], pt_low[kk][2], pt_high[kk][2]))
		del nodes
		pt_inside[kk] = dolfin.AutoSubDomain(lambda xx, on: (pt_low[kk]-myeps <= xx).all() and (xx <= pt_high[kk]+myeps).all())
	ff.close()
	info('Patch size computations finished')

	hole_ratio = hole_radius/lengths[1]
	for ref in range(max_refines):
		info('Start meshing resolution {:d}/{:d}'.format(ref+1, max_refines))
		res = max_resolution*0.5**ref

		if hole:
			hole_maxh = res*hole_ratio
#		   hole_maxh = np.min([res*hole_ratio, lengths[2]/layers])
			if number:
				matrix = nohole_matrix-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
				incs = nohole_matrix-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
			else:
				whole = nohole_whole-Cylinder(Pnt(center[0], center[1], center[2]-diameter), Pnt(center[0], center[1], center[2]+diameter), hole_radius).maxh(hole_maxh)
			
		dirname = '{:s}/{:s}_{:d}_patches/0/'.format(prefix, prefix, ref)
		sa_utils.makedirs_norace(dirname)

		basename = '{:s}/{:s}_{:d}'.format(prefix, prefix, ref)
		info('Global CSG')
		geo = CSGeometry()

		if number:
			geo.Add(matrix*layer_bricks[0])
			for ii in range(1, sublayers):
				geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1]))
			geo.Add(incs*layer_bricks[0])
			for ii in range(1, sublayers):
				geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1]))
		else:
			geo.Add(whole*layer_bricks[0])
			for ii in range(1, sublayers):
				geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1]))
		info('Global CSG constructed')
		mesh = geo.GenerateMesh(maxh = res)
		info('Global surface meshed')
		del geo

		gc.collect()

		mesh.GenerateVolumeMesh()
		mesh.Export(basename+'.msh', 'Gmsh2 Format')
		meshconvert.convert2xml(basename+'.msh', basename+'.xml')
		del mesh
		os.remove(basename+'.msh')
		os.remove(basename+'_facet_region.xml')
		info('Global volume meshed')

		gc.collect()

		global_mesh = dolfin.Mesh(basename+'.xml')
		tmp_nodes = global_mesh.coordinates()
		tmp_low = np.min(tmp_nodes, axis = 0)
		tmp_high = np.max(tmp_nodes, axis = 0)
		info('global mesh: {:s}, {:s}, {:s}'.format(str(tmp_low), str(tmp_high), str(top.inside(tmp_high, True))))

		os.remove(basename+'.xml')
		info('Correcting cell markers')
		global_domains_tmp = dolfin.MeshFunction('size_t', global_mesh, basename+'_physical_region.xml')
		os.remove(basename+'_physical_region.xml')
		global_domains_tmp.array()[:] -= np.min(global_domains_tmp.array())
		global_domains_tmp.array()[:] //= elem_per_layer
		global_domains = dolfin.MeshFunction('size_t', global_mesh, basedim, 0)
		if number:
			where = np.where(global_domains_tmp.array() < layers)
			global_domains.array()[where] = 4*global_domains_tmp.array()[where]
			where = np.where(layers <= global_domains_tmp.array())
			global_domains.array()[where] = 4*(global_domains_tmp.array()[where]-layers)+1
			del where
		else:
			global_domains.array()[:] = 4*global_domains_tmp.array()
		del global_domains_tmp
		if ldomain:
			for ii in range(corner_refine):
				mf = dolfin.CellFunction('bool', global_mesh, False)
				corner_subdomains[ii].mark(mf, True)
				global_mesh = dolfin.refine(global_mesh, mf)
				global_domains = dolfin.adapt(global_domains, global_mesh)
				del mf
		inside_fun = dolfin.MeshFunction('bool', global_mesh, basedim, True)
		global_mesh = dolfin.refine(global_mesh, inside_fun)
		global_domains = dolfin.adapt(global_domains, global_mesh)
		info('Correcting facet markers')
		global_facets = dolfin.MeshFunction('size_t', global_mesh, basedim-1, 0)
		for key in bc_dict:
			bc_dict[key].mark(global_facets, key)
		sa_hdf5.write_dolfin_mesh(global_mesh, basename, cell_function = global_domains, facet_function = global_facets)
		del global_facets, global_mesh, global_domains, basename

		gc.collect()

		for kk in range(patch_num):
			if kk in skip_patches:
				continue
			info(str(kk)+'/'+str(patch_num))
			basename = 'patch_'+str(kk).zfill(patch_fill)
			extname = basename+'_'+str(beta)

			info('	csg')
			geo = CSGeometry()
			if number:
				geo.Add(matrix*layer_bricks[0]*patches[kk])
				for ii in range(1, sublayers):
					geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
				geo.Add(incs*layer_bricks[0]*patches[kk])
				for ii in range(1, sublayers):
					geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
				geo.Add(matrix*layer_bricks[0]*patches_ext[kk])
				for ii in range(1, sublayers):
					geo.Add(matrix*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
				geo.Add(incs*layer_bricks[0]*patches_ext[kk])
				for ii in range(1, sublayers):
					geo.Add(incs*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
			else:
				geo.Add(whole*layer_bricks[0]*patches[kk])
				for ii in range(1, sublayers):
					geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1])*patches[kk])
				geo.Add(whole*layer_bricks[0]*patches_ext[kk])
				for ii in range(1, sublayers):
					geo.Add(whole*(layer_bricks[ii]-layer_bricks[ii-1])*patches_ext[kk])
			info('	csg done')
			mesh = geo.GenerateMesh(maxh = res)
			info('	surface meshed')
			del geo

			gc.collect()

			mesh.GenerateVolumeMesh()
			info('	volume meshed')
			mesh.Export(dirname+'/'+basename+'.msh', 'Gmsh2 Format')
			meshconvert.convert2xml(dirname+'/'+basename+'.msh', dirname+'/'+basename+'.xml')
			del mesh
			os.remove(dirname+'/'+basename+'.msh')
			os.remove(dirname+'/'+basename+'_facet_region.xml')

			gc.collect()

			ext_mesh = dolfin.Mesh(dirname+'/'+basename+'.xml')
			os.remove(dirname+'/'+basename+'.xml')
			info('	cell function')
			ext_domains_tmp = dolfin.MeshFunction('size_t', ext_mesh, dirname+'/'+basename+'_physical_region.xml')
			os.remove(dirname+'/'+basename+'_physical_region.xml')
			ext_domains_tmp.array()[:] -= np.min(ext_domains_tmp.array())
			ext_domains_tmp.array()[:] //= elem_per_layer
			ext_domains = dolfin.MeshFunction('size_t', ext_mesh, basedim, 0)
			if number:
				where = np.where(ext_domains_tmp.array() < layers)
				ext_domains.array()[where] = 4*ext_domains_tmp.array()[where]
				where = np.where(layers <= ext_domains_tmp.array() < 2*layers)
				ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-layers)+1
				where = np.where(2*layers <= ext_domains_tmp.array() < 3*layers)
				ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-2*layers)+2
				where = np.where(3*layers <= ext_domains_tmp.array())
				ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-3*layers)+3
				del where
			else:
				where = np.where(ext_domains_tmp.array() < layers)
				ext_domains.array()[where] = 4*ext_domains_tmp.array()[where]
				where = np.where(layers <= ext_domains_tmp.array())
				ext_domains.array()[where] = 4*(ext_domains_tmp.array()[where]-layers)+2
			del ext_domains_tmp
			if ldomain:
				for ii in range(corner_refine):
					mf = dolfin.CellFunction('bool', ext_mesh, False)
					corner_subdomains[ii].mark(mf, True)
					ext_mesh = dolfin.refine(ext_mesh, mf)
					ext_domains = dolfin.adapt(ext_domains, ext_mesh)
					del mf
			inside_fun = dolfin.MeshFunction('bool', ext_mesh, basedim, False)
			pt_inside[kk].mark(inside_fun, True)
			ext_mesh = dolfin.refine(ext_mesh, inside_fun)
			del inside_fun
			ext_domains = dolfin.adapt(ext_domains, ext_mesh)
			info('	cell function done')
		 
			pt_mesh = dolfin.SubMesh(ext_mesh, pt_inside[kk])

			pt_domains = dolfin.MeshFunction('size_t', pt_mesh, basedim, 0)
			tree = ext_mesh.bounding_box_tree()
			for cell in dolfin.cells(pt_mesh):
				global_index = tree.compute_first_entity_collision(cell.midpoint())
				pt_domains[cell] = ext_domains[dolfin.Cell(ext_mesh, global_index)]
			del tree

			pt_facets = dolfin.MeshFunction('size_t', pt_mesh, basedim-1, 0)
			border.mark(pt_facets, 100)
			for key in bc_dict:
				bc_dict[key].mark(pt_facets, key)
			sa_hdf5.write_dolfin_mesh(pt_mesh, '{:s}/{:s}'.format(dirname, basename), cell_function = pt_domains, facet_function = pt_facets)

			tmp_nodes = ext_mesh.coordinates()
			tmp_low = np.min(tmp_nodes, axis = 0)
			tmp_high = np.max(tmp_nodes, axis = 0)
			is_part = (tmp_low > low+myeps).any() or (tmp_high < high-myeps).any()
			del tmp_low, tmp_high, tmp_nodes
			info('patch [{:d}/{:d}], beta [{:.2e}] is real subdomain [{:}]'.format(kk+1, patch_num, beta, is_part))

			if is_part:
				vals = np.arange(1,11)
			else:
				vals = np.unique(pt_facets.array())
				vals = vals[np.where(vals > 0)]
			patch_dict = dict()
			for key in bc_dict:
				if key in vals:
					patch_dict[key] = bc_dict[key]
				else:
					patch_dict[key] = dolfin.AutoSubDomain((lambda what: (lambda xx, on: bc_dict[what].inside(xx, on) and pt_inside[kk].inside(xx, on)))(key))
			ext_facets = dolfin.MeshFunction('size_t', ext_mesh, basedim-1, 0)
			border.mark(ext_facets, 100)
			for key in patch_dict:
				patch_dict[key].mark(ext_facets, key)
			del patch_dict, vals
			sa_hdf5.write_dolfin_mesh(ext_mesh, dirname+'/'+basename+'_'+str(beta), cell_function = ext_domains, facet_function = ext_facets)
			del ext_mesh, ext_domains, ext_facets, pt_mesh, pt_domains, pt_facets

			gc.collect()

	del pt_low, pt_high, pt_inside
"""