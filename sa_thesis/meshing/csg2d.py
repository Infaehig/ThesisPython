"""
2D meshing using mostly FEniCS
"""

import dolfin
from dolfin_utils.meshconvert import meshconvert
import mshr
import numpy

from .base import CSG, CSGCollection, csg_eps

class CSG2D(CSG):
	@classmethod
	def getCollection(cls):
		return CSG2DCollection()

class Circle(CSG2D):
	def __init__(self, center, radius, *, maxh = 1., eps = csg_eps):
		segments = max(int(4*radius/maxh), 32)
		eps *= radius
		super().__init__(mshr.Circle(dolfin.Point(*center), radius, segments), boundaries = [
			dolfin.CompiledSubDomain(
				'on_boundary && near((x[0]-c0)*(x[0]-c0) + (x[1]-c1)*(x[1]-c1), rr, eps)',
				c0 = center[0], c1 = center[1], rr = radius * radius, eps = eps
			)
		])

class Rectangle(CSG2D):
	def __init__(self, pointa, pointb, *, eps = csg_eps):
		eps *= numpy.linalg.norm(numpy.array(pointb)-numpy.array(pointa))
		super().__init__(mshr.Rectangle(dolfin.Point(*pointa), dolfin.Point(*pointb)), boundaries = [
			dolfin.CompiledSubDomain('on_boundary && near(x[0], xx, eps)', xx = pointa[0], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[0], xx, eps)', xx = pointb[0], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[1], xx, eps)', xx = pointa[1], eps = eps),
			dolfin.CompiledSubDomain('on_boundary && near(x[1], xx, eps)', xx = pointb[1], eps = eps)
		])

class CSG2DCollection(CSGCollection):
	def __init__(self):
		self.all = None
		self.collection = []
		self.boundaries = []

	def add(self, csg2d, *, add_boundaries = True):
		if self.all is None:
			self.all = csg2d.geometry
		else:
			self.all += csg2d.geometry
		self.collection.append(csg2d.geometry)
		if add_boundaries:
			self.boundaries.extend(csg2d.boundaries)
		
	def generateMesh(self, maxh):
		for ii, subdomain in enumerate(self.collection):
			if not ii:
				continue
			self.all.set_subdomain(ii, subdomain)
		self.mesh = mshr.generate_mesh(self.all, int(1/maxh))
		self.domains = dolfin.MeshFunction('size_t', self.mesh, 2, self.mesh.domains())
		self.facets = dolfin.MeshFunction('size_t', self.mesh, 1, 0)
		for ii, subdomain in enumerate(self.boundaries):
			subdomain.mark(self.facets, ii+1)

"""
class Layers(CSG3DCollection):
	def __init__(self, boundary, low, high, layers = 1, *, elements_per_layer = 2):
		super().__init__()
		self.boundaries = boundary.boundaries
		self.layers = layers
		self.elements_per_layer = elements_per_layer
		hh = (numpy.array(high)-numpy.array(low))/(self.layers*self.elements_per_layer)
		lower = low + hh
		super().add(boundary*HalfPlane(lower, hh), add_boundaries = False)
		for ii in range(1, self.layers*self.elements_per_layer-1):
			super().add(boundary*(HalfPlane(lower+hh, hh)-HalfPlane(lower, hh)), add_boundaries = False)
			lower += hh
		super().add(boundary*HalfPlane(lower, -hh), add_boundaries = False)

	def generateMesh(self, maxh):
		super().generateMesh(maxh)
		self.domains.array()[:] //= self.elements_per_layer

from ..helpers import io

dimension = 2

def do_stuff():
	print(f'meshing {dimension}d')
	io.do_stuff(dimension)

from dolfin import *
from mshr import *

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import logging
import sa_utils
from sa_utils import myeps, comm, rank, size, adapt

import sa_hdf5

import os
import time
import gc
import multiprocessing

corner_refine = 4

def create_patches(box = np.array([0, 0, 1, 1]), prefix = 'test', 
				   myrange = np.arange(2, 6, dtype = int), betas = [2.], 
				   num = 100, bands = None, hole = False, inc_box = None, balls = True, 
				   true_inside = False, oht_like = False, 
				   patches_only = False, 
				   partition_box = None, partition_level = 0, partition_x = 2, 
				   partition_alpha = 1.25, partition_res_mod = -1, 
				   logger = None, sub_inner = True, refine_inside = True, shape_radius = None, 
				   inclusions_file = None, random_radius = None, global_feature = None, 
				   inc_res = 64, corner_refine = 4, strict_oversampling = True,
				   transform = None, channels_params = None, free_oversampling = False):
	if rank:
		return
	
	assert(partition_alpha > 1)

	if partition_box is None:
		partition_box = box 

	rangelen = len(myrange)
	logger.info('Creating ['+str(rangelen)+'] meshes with resolution 2^'+str(myrange))
	basedim = len(box)//2

	low = box[:basedim].copy(); high = box[basedim:].copy()
	lengths = box[basedim:]-box[:basedim]
	diameter = np.sqrt(lengths@lengths)
	center = (box[:basedim]+box[basedim:])/2.

	betas = np.sort(np.array(betas))

	num_betas = len(betas)
	hs = np.array([lengths*(betas[kk]-1.0)/2. for kk in range(num_betas)])

	left = AutoSubDomain(lambda xx, on: on and near(xx[0], low[0], eps = myeps))
	right = AutoSubDomain(lambda xx, on: on and near(xx[0], high[0], eps = myeps))
	front = AutoSubDomain(lambda xx, on: on and near(xx[1], low[1], eps = myeps))
	back = AutoSubDomain(lambda xx, on: on and near(xx[1], high[1], eps = myeps))
	boundary = AutoSubDomain(lambda xx, on: on)
	bc_dict = dict()
	bc_dict[1] = left
	bc_dict[2] = right
	bc_dict[3] = front
	bc_dict[4] = back

	sa_utils.makedirs_norace(prefix)

	logger.info('Setting up csg domain')
	hole_radius = lengths[1]/7.
	if basedim == 2:
		domain = Rectangle(Point(*low), Point(*high)) 
		if hole:
			domain -= Circle(Point(*center), hole_radius, 128)
			holed = AutoSubDomain(lambda xx, on: on and np.sqrt((xx-center)@(xx-center)) < hole_radius+myeps)
			bc_dict[10] = holed
			hole_subdomains = []
			hole_close = hole_radius*0.5
			for ii in range(corner_refine):
				hole_subdomains.append(AutoSubDomain((lambda what: lambda xx, on: np.sqrt((xx-center)@(xx-center)) < what)(hole_radius+hole_close)))
				hole_close *= 0.5
	else:
		raise NameError('Unimplemented')

	def corners_ok(corners):
		ret = False
		for cc in corners:
			if (low-myeps <= cc).all() and (cc <= high+myeps).all():
				ret = True
				break;
		if not ret or not hole:
			return ret
		ret = False
		for cc in corners:
			diff = cc-center
			if np.sqrt(diff @ diff) > hole_radius-myeps: 
				return True
		return False

	logger.info('Basic domain set up')
	ldomain = False
	if global_feature is not None:
		min_idx = np.argmin(lengths)
		min_length = lengths[min_idx]
		if global_feature == 'channel':
			inclusions = CSGTranslation(CSGRotation(Rectangle(Point(-0.05*min_length, -diameter), Point(0.05*min_length, diameter)), -0.1*pi), Point(*center))
		elif global_feature == 'channels':
			assert(channels_params is not None)
			cwidth = channels_params[0]
			csep = channels_params[1]
			cangle = channels_params[2]
			coffset = channels_params[3]
			coffset = coffset % (cwidth+csep)
			tmp = cwidth+csep
			cmatrix = Rectangle(Point(-diameter,-0.5*cwidth+coffset),Point(diameter,0.5*cwidth+coffset))
			while tmp < 0.6*diameter:
				cmatrix += Rectangle(Point(-diameter,-0.5*cwidth+coffset+tmp),Point(diameter,0.5*cwidth+coffset+tmp))
				cmatrix += Rectangle(Point(-diameter,-0.5*cwidth+coffset-tmp),Point(diameter,0.5*cwidth+coffset-tmp))
				tmp += cwidth+csep
			inclusions = CSGTranslation(CSGRotation(cmatrix, cangle*pi), Point(*center))
		elif global_feature == 'ring':
			inclusions = Circle(Point(*center), 0.3*diameter)-Circle(Point(*center), 0.2*diameter)
		elif global_feature == 'x':
			inclusions = CSGTranslation(CSGRotation(Rectangle(Point(-0.05*min_length, -diameter), Point(0.05*min_length, diameter)), -0.1*pi)+
										CSGRotation(Rectangle(Point(-0.05*min_length, -diameter), Point(0.05*min_length, diameter)), 0.1*pi), Point(*center))
		elif global_feature == 'ldomain':
			ldomain = True
			domain -= Rectangle(Point(*center), Point(*high))

	if ldomain:
		corner_subdomains = []
		corner_close = 0.1*diameter+myeps
		for ii in range(corner_refine):
			corner_subdomains.append(AutoSubDomain((lambda what: lambda xx, on: np.sqrt((xx-center)@(xx-center)) < what)(corner_close)))
			corner_close *= 0.5
		corner_lr = AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and near(xx[0], center[0], eps = myeps))
		corner_fb = AutoSubDomain(lambda xx, on: on and (xx >= center-myeps).all() and near(xx[1], center[1], eps = myeps))
		bc_dict[7] = corner_lr
		bc_dict[8] = corner_fb

	logger.info('Global features done')
	if num:
		if hole or oht_like:
			inc_radius = hole_radius/4.
		else:
#		   inc_radius = 0.25/num
			inc_radius = 0.5/num
		if shape_radius is None:
			shape_radius = inc_radius

		if balls:
			Shape = lambda xx, rr: Circle(Point(*xx), rr, inc_res)
		else:
			Shape = lambda xx, rr: Rectangle(Point(*(xx-np.array([rr, rr]))), Point(*(xx+np.array([rr, rr]))))

		logger.info('adding csg inclusions')
		if inc_box is None:
			inc_box = box.copy()
		inc_low = inc_box[:basedim]; inc_high = inc_box[basedim:]
		inc_lengths = inc_high - inc_low

		muls = np.array(inc_lengths*num, dtype = int)
		muls[np.where(muls < 1)] = 1
		num_inclusions = np.prod(muls)
		
		logger.info('Preparing creation of ['+str(lengths)+'] box with approx. ['+str(num_inclusions)+'] inclusions of radius ['+str(shape_radius)+']')
		if true_inside:
			inc_low += 2*inc_radius
		if true_inside:
			inc_high -= 2*inc_radius
		width = inc_high-inc_low

		def get_banded_centers(bands):
			steps = 1.*width/(muls*bands)
			steps[np.where(steps <= 0)] = inc_lengths[np.where(steps <= 0)]
			logger.info('steps: '+str(steps))
			XX, YY = np.mgrid[inc_low[0]:inc_high[0]+steps[0]:steps[0], inc_low[1]:inc_high[1]+steps[1]:steps[1]]
			nodes = np.vstack((XX.flat, YY.flat)).T
			centersn = nodes[::np.prod(bands)]
			return centersn

		def get_random_centers():
			ret = []
			radii = []
			logger.info('Getting random centers (and radii)')
			percentages = np.arange(.1, 1.05, 0.1)*num_inclusions
			idx = 1
			while(len(ret) < num_inclusions):
				notok = True
				while(notok):
					new = rnd.rand(2)*width+inc_low
					if random_radius is not None:
						new_radius = (random_radius+rnd.rand()*(1.-random_radius))*shape_radius
					else:
						new_radius = shape_radius
					notok = False
					for oldx, oldr in zip(ret, radii):
						diff = new-oldx
						if sqrt(diff @ diff) < 1.1*(oldr+new_radius):
							notok = True
							break
				ret.append(new)
				radii.append(new_radius)
				if len(ret) > percentages[idx]:
					logger.info('	[{:d}/{:d} = {:.2f}%] done'.format(len(ret), num_inclusions, 100.*len(ret)/num_inclusions))
					idx += 1
			return np.array(ret), np.array(radii)

		if inclusions_file is not None:
			tmp = np.loadtxt(inclusions_file, delimiter = ', ')
			num_incs = len(tmp)
			centers = tmp[:, :2]
			radii = tmp[:, 2]
		else:
			if bands is None:
				centers, radii = get_random_centers()
			else:
				centers = get_banded_centers(bands)
				radii = np.ones(len(centers))*shape_radius
			num_incs = len(centers)

		if ldomain:
			new_centers = []
			new_radii = []
			for ii in range(len(centers)):
				if (centers[ii] <= center+myeps).any():
					new_centers.append(centers[ii])
					new_radii.append(radii[ii])
			centers = np.array(centers)
			radii = np.array(radii)
			del new_centers, new_radii

		if global_feature is not None:
			inclusions += Shape(centers[0], radii[0])
		else:
			inclusions = Shape(centers[0], radii[0])
		for ii in range(1, num_incs):
			inclusions += Shape(centers[ii], radii[ii])

		np.savetxt(prefix+'/'+prefix+'_inclusions.csv', np.hstack((centers, radii.reshape(num_incs, 1))), fmt = '%.15e', delimiter = ', ')
		logger.info('small inclusions computed')

	if global_feature is not None and not ldomain:
		num += 1

	if num:
		domain.set_subdomain(1, inclusions)
		logger.info('csg inclusions added')

	logger.info('Partitions')
	partition_domains = []
	partition_subdomains = []
	partition_extended_domains = []
	partition_extended_subdomains = []
	partition_lengths = []
	if partition_level > 0:
		assert(prefix is not None)
		partition_low = partition_box[:basedim]
		partition_high = partition_box[basedim:]
		partition_length = partition_high-partition_low
		for lvl in range(partition_level):
			partition_num = partition_x*2.**lvl
			partition_hh = partition_length[0]/partition_num
			partition_nums = np.array(np.ceil(partition_length/partition_hh), dtype = int)
			partition_hs = partition_length/partition_nums
			partition_hh_alpha = (partition_hh*partition_alpha)/2.
			p_inside = []
			p_subdomains = []
			p_extended = [[] for kk in range(num_betas)]
			p_lengths = []
			p_extended_subdomains = [[] for kk in range(num_betas)]
			patch_dir = prefix+'/'+prefix+'_patch_descriptors'
			sa_utils.makedirs_norace(patch_dir)
			patches_file = open(patch_dir+'/'+str(lvl)+'.csv', 'w')
			count = 0
			patches_file.write('idx, left, right, front, back\n')
			for ll in range(partition_nums[1]):
				for kk in range(partition_nums[0]):
					p_center = partition_low+np.array([0.5+kk, 0.5+ll])*partition_hs
					p_low_mshr = p_center-partition_hh_alpha
					p_high_mshr = p_center+partition_hh_alpha
					if ldomain and (p_low_mshr >= center-myeps).all():
						continue
					p_low = np.max([p_low_mshr, low], axis = 0)
					p_high = np.min([p_high_mshr, high], axis = 0)
					if (p_low >= p_high-myeps).any():
						continue;
					p_r_d = np.array([p_high[0], p_low[1]])
					p_l_u = np.array([p_low[0], p_high[1]])
					corners = [p_low, p_high, p_r_d, p_l_u]
					if not corners_ok(corners):
						continue;
					patches_file.write("{:d}, {:.15e}, {:.15e}, {:.15e}, {:.15e}\n".format(count, p_low[0], p_high[0], p_low[1], p_high[1]))
					count += 1
					p_inside.append(domain*Rectangle(Point(*p_low_mshr), Point(*p_high_mshr)))
					if num:
						p_inside[-1].set_subdomain(1, inclusions)
					p_subdomains.append(AutoSubDomain((lambda what_low, what_high, eps: (lambda xx, on: (what_low-eps <= xx).all() and (xx <= what_high+eps).all()))(p_low, p_high, myeps)))
					bla = 4
					for mm in reversed(range(num_betas)):
						p_low_extended_mshr = p_center-(partition_hh_alpha)*betas[mm]
						p_high_extended_mshr = p_center+(partition_hh_alpha)*betas[mm]
						p_low_extended = np.max([p_low_extended_mshr, low], axis = 0)
						p_high_extended = np.min([p_high_extended_mshr, high], axis = 0)
						p_extended_subdomains[mm].append(AutoSubDomain((lambda what_low, what_high, eps: (lambda xx, on: (what_low-eps <= xx).all() and (xx <= what_high+eps).all()))(p_low_extended, p_high_extended, myeps)))
						p_extended[mm].append(domain*Rectangle(Point(*p_low_extended_mshr), Point(*p_high_extended_mshr)))
						p_extended[mm][-1].set_subdomain(2, p_extended[mm][-1]-p_inside[-1])
						if mm < num_betas-1:
							p_extended[num_betas-1][-1].set_subdomain(bla, p_extended[mm][-1]-p_inside[-1])
						if num:
							p_extended[mm][-1].set_subdomain(1, inclusions*p_inside[-1])
							p_extended[mm][-1].set_subdomain(3, inclusions*(p_extended[mm][-1]-p_inside[-1]))
							p_extended[num_betas-1][-1].set_subdomain(bla+1, inclusions*(p_extended[mm][-1]-p_inside[-1]))
						bla += 2
			patches_file.close()
			partition_domains.append(p_inside)
			partition_subdomains.append(p_subdomains)
			partition_extended_domains.append(p_extended)
			partition_extended_subdomains.append(p_extended_subdomains)
			partition_lengths.append(len(p_inside))

	logger.info('partitions set up')

	if partition_level is not None:
		logger.info('Creating partition meshes')
		for lvl in range(partition_level):
			logger.info('level ['+str(lvl)+'], ['+str(partition_lengths[lvl])+']') 
			count = 0
			for ii in myrange[:partition_res_mod]:
				resolution = int(2**ii)
				logger.info('['+str(count+1)+'/'+str(rangelen+partition_res_mod)+'], resolution ['+str(resolution)+']')
				old = sa_utils.get_time()
				pt_inside = partition_domains[lvl]
				pt_subdomains = partition_subdomains[lvl]
				pt_extended = partition_extended_domains[lvl]
				pt_extended_subdomains = partition_extended_subdomains[lvl]
				pt_length = partition_lengths[lvl]
				pt_fill = int(np.log(pt_length)/np.log(10.))+1
				patch_dir = prefix+'/'+prefix+'_'+str(ii)+'_patches/'+str(lvl)
				def mesh_patch(kk, out_q):
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] largest of ['+str(num_betas)+'], beta ['+str(betas[-1])+']')
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] generate mesh')
					pt_mesh_extended = generate_mesh(pt_extended[-1][kk], resolution)
					pt_domains_extended = MeshFunction('size_t', pt_mesh_extended, basedim, pt_mesh_extended.domains())
					arr = pt_domains_extended.array()
					arr[np.where((arr >= 4)*(arr % 2 == 0))] = 2
					arr[np.where((arr >= 4)*(arr % 2 == 1))] = 3
					del arr
					if hole:
						for jj in range(corner_refine):
							mf = MeshFunction('bool', pt_mesh_extended, basedim, False)
							hole_subdomains[jj].mark(mf, True)
							pt_mesh_extended = refine(pt_mesh_extended, mf)
							pt_domains_extended = adapt(pt_domains_extended, pt_mesh_extended)
							del mf
					if ldomain:
						for jj in range(corner_refine):
							mf = MeshFunction('bool', pt_mesh_extended, basedim, False)
							corner_subdomains[jj].mark(mf, True)
							pt_mesh_extended = refine(pt_mesh_extended, mf)
							pt_domains_extended = adapt(pt_domains_extended, pt_mesh_extended)
							del mf
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] mesh generated, refining inner part')
					if refine_inside:
						inside_fun = MeshFunction('bool', pt_mesh_extended, basedim, False)
						pt_subdomains[kk].mark(inside_fun, True)
						pt_mesh_extended = refine(pt_mesh_extended, inside_fun)
						pt_domains_extended = adapt(pt_domains_extended, pt_mesh_extended)
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] inner extraction')
					pt_mesh = SubMesh(pt_mesh_extended, pt_subdomains[kk])
					pt_domains = MeshFunction('size_t', pt_mesh, basedim, 0)
					tree = pt_mesh_extended.bounding_box_tree()
					if num:
						for cell in cells(pt_mesh):
							global_index = tree.compute_first_entity_collision(cell.midpoint())
							global_cell = Cell(pt_mesh_extended, global_index)
							pt_domains[cell] = pt_domains_extended[Cell(pt_mesh_extended, global_index)]
					pt_facets = MeshFunction('size_t', pt_mesh, basedim-1, 0)
					boundary.mark(pt_facets, 100)
					left.mark(pt_facets, 1); right.mark(pt_facets, 2)
					front.mark(pt_facets, 3); back.mark(pt_facets, 4)
					if hole:
						holed.mark(pt_facets, 10)
					if ldomain:
						corner_lr.mark(pt_facets, 7)
						corner_fb.mark(pt_facets, 8)
					sa_hdf5.write_dolfin_mesh(pt_mesh, patch_dir+'/patch_'+str(kk).zfill(pt_fill), cell_function = pt_domains, facet_function = pt_facets)
					if transform:
						sa_hdf5.write_dolfin_mesh(pt_mesh, prefix+'_transform/'+prefix+'_'+str(ii)+'_patches/'+str(lvl)+'/patch_'+str(kk).zfill(pt_fill), cell_function = pt_domains, facet_function = pt_facets, transform = transform)
 
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] inside finished')
					tmp_nodes = pt_mesh_extended.coordinates()
					tmp_low = np.min(tmp_nodes, axis = 0)
					tmp_high = np.max(tmp_nodes, axis = 0)
					is_part = (tmp_low > low+myeps).any() or (tmp_high < high-myeps).any()
					del tmp_low, tmp_high, tmp_nodes
					logger.info('patch [{:d}/{:d}], beta [{:.2e}] is real subdomain [{:}]'.format(kk+1, pt_length, betas[-1], is_part))
					if free_oversampling:
						vals = []
					elif strict_oversampling and is_part:
						vals = np.arange(1,11)
					else:
						vals = np.unique(pt_facets.array())
						vals = vals[np.where(vals > 0)]
					patch_dict = dict()
					for key in bc_dict:
						if key in vals:
							patch_dict[key] = bc_dict[key]
						else:
							patch_dict[key] = AutoSubDomain((lambda what: (lambda xx, on: bc_dict[what].inside(xx, on) and pt_subdomains[kk].inside(xx, on)))(key))
					pt_facets_extended = MeshFunction('size_t', pt_mesh_extended, basedim-1, 0)
					boundary.mark(pt_facets_extended, 100)
					for key in patch_dict:
						patch_dict[key].mark(pt_facets_extended, key)
					del patch_dict, vals

					sa_hdf5.write_dolfin_mesh(pt_mesh_extended, patch_dir+'/patch_'+str(kk).zfill(pt_fill)+'_'+str(betas[-1]), cell_function = pt_domains_extended, facet_function = pt_facets_extended)
					if not transform is None:
						sa_hdf5.write_dolfin_mesh(pt_mesh_extended, prefix+'_transform/'+prefix+'_'+str(ii)+'_patches/'+str(lvl)+'/patch_'+str(kk).zfill(pt_fill)+'_'+str(betas[-1]), cell_function = pt_domains_extended, facet_function = pt_facets_extended, transform = transform)
					logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] finished')
					del pt_facets_extended, pt_mesh, pt_domains, pt_facets 
					gc.collect()
					all_array = pt_domains_extended.array()
					for mm in range(num_betas-1):
						logger.info('patch ['+str(kk+1)+'/'+str(pt_length)+'] other betas [{:d}/{:d}]'.format(mm+1, num_betas-1))
						subdomain = pt_extended_subdomains[mm][kk]
						submesh = SubMesh(pt_mesh_extended, subdomain)
						num_submesh = submesh.num_cells()
						cmap = submesh.data().array('parent_cell_indices', 2)
						subcells = MeshFunction('size_t', submesh, basedim, 0)
						subcells_array = subcells.array()
						for idx in range(num_submesh):
							subcells_array[idx] = all_array[cmap[idx]]

						tmp_nodes = submesh.coordinates()
						tmp_low = np.min(tmp_nodes, axis = 0)
						tmp_high = np.max(tmp_nodes, axis = 0)
						is_part = (tmp_low > low+myeps).any() or (tmp_high < high-myeps).any()
						del tmp_low, tmp_high, tmp_nodes
						logger.info('patch [{:d}/{:d}], beta [{:.2e}] is real subdomain [{:}]'.format(kk+1, pt_length, betas[mm], is_part))
						if free_oversampling:
							vals = []
						elif strict_oversampling and is_part:
							vals = np.arange(1,11)
						else:
							vals = np.unique(pt_facets.array())
							vals = vals[np.where(vals > 0)]
						patch_dict = dict()
						for key in bc_dict:
							if key in vals:
								patch_dict[key] = bc_dict[key]
							else:
								patch_dict[key] = AutoSubDomain((lambda what: (lambda xx, on: bc_dict[what].inside(xx, on) and pt_subdomains[kk].inside(xx, on)))(key))
						pt_facets_extended = MeshFunction('size_t', submesh, basedim-1, 0)
						boundary.mark(pt_facets_extended, 100)
						for key in patch_dict:
							patch_dict[key].mark(pt_facets_extended, key)
						del patch_dict, vals
 
						sa_hdf5.write_dolfin_mesh(submesh, patch_dir+'/patch_'+str(kk).zfill(pt_fill)+'_'+str(betas[mm]), cell_function = subcells, facet_function = pt_facets_extended)
						if not transform is None:
							sa_hdf5.write_dolfin_mesh(submesh, prefix+'_transform/'+prefix+'_'+str(ii)+'_patches/'+str(lvl)+'/patch_'+str(kk).zfill(pt_fill)+'_'+str(betas[mm]), cell_function = subcells, facet_function = pt_facets_extended, transform = transform)

						del subdomain, submesh, num_submesh, cmap, subcells, idx, pt_facets_extended
						gc.collect()
					del pt_mesh_extended, pt_domains_extended, all_array
					gc.collect()
					out_q.put(True)
				mesh_processes = []
				out_q = multiprocessing.Queue()
				for kk in range(pt_length):
					mesh_processes.append(multiprocessing.Process(target = mesh_patch, args = (kk, out_q)))
				block = multiprocessing.cpu_count()
				block_low = 0; block_high = np.min([block, pt_length])
				while(block_low < block_high):
					for kk in range(block_low, block_high):
						mesh_processes[kk].start()
					for kk in range(block_low, block_high):
						out_q.get()
					for kk in range(block_low, block_high):
						mesh_processes[kk].join()
						mesh_processes[kk] = None
						logger.info('['+str(kk)+'] joined')
					block_low = block_high
					block_high = np.min([block_high+block, pt_length])
				new = sa_utils.get_time()
				logger.info('['+str(count+1)+'/'+str(rangelen)+'], resolution ['+str(resolution)+'] finished in ['+sa_utils.human_time(new-old)+']s')
				count += 1
		logger.info('done')

	if not patches_only:
		logger.info('Creating full meshes')
		count = 0
		for ii in myrange:
			resolution = int(2**ii)
			logger.info('['+str(count+1)+'/'+str(rangelen)+'], resolution ['+str(resolution)+']')
			mesh = generate_mesh(domain, resolution)
			logger.info('meshed ['+str(mesh.num_vertices())+']')
			if num:
				domains = MeshFunction('size_t', mesh, basedim, mesh.domains())
			else:
				domains = MeshFunction('size_t', mesh, basedim, 0)
			if hole:
				for jj in range(corner_refine):
					mf = MeshFunction('bool', mesh, basedim, False)
					hole_subdomains[jj].mark(mf, True)
					mesh = refine(mesh, mf)
					domains = adapt(domains, mesh)
					del mf
			if ldomain:
				for jj in range(corner_refine):
					mf = MeshFunction('bool', mesh, basedim, False)
					corner_subdomains[jj].mark(mf, True)
					mesh = refine(mesh, mf)
					domains = adapt(domains, mesh)
					del mf
			facets = MeshFunction('size_t', mesh, basedim-1, 0)
			left.mark(facets, 1); right.mark(facets, 2)
			front.mark(facets, 3); back.mark(facets, 4)
			if ldomain:
				corner_lr.mark(facets, 7)
				corner_fb.mark(facets, 8)
			logger.info('writing')
			sa_hdf5.write_dolfin_mesh(mesh, prefix+'/'+prefix+'_'+str(ii), cell_function = domains, facet_function = facets)
			if not transform is None:
				sa_hdf5.write_dolfin_mesh(mesh, prefix+'_transform/'+prefix+'_'+str(ii), cell_function = domains, facet_function = facets, transform = transform)
			count += 1
			del mesh, domains, facets
			gc.collect()
		logger.info('full meshes created')

	logger.info('DONE')

def get_num_patches(meshdir, prefix, patch_level):
	patch_boxes = np.genfromtxt(meshdir+'/'+prefix+'_patch_descriptors/'+str(patch_level)+'.csv', delimiter = ', ', names = True)
	patch_num = len(patch_boxes['idx'])
	patch_fill = int(np.log(patch_num)/np.log(10.))+1
	return patch_num, patch_fill

def load_mesh_h5py(basename):
	ret = sa_hdf5.read_dolfin_mesh(basename)
	return ret['mesh'], ret['cell_function'], ret['facet_function']

def load_meshes_h5py(meshdir, prefix, myrange):
	meshes = []
	domains = []
	facets = []
	head = meshdir+'/'+prefix+'_'
	ct = 0
	num = len(myrange)
	for ii in myrange:
		mesh, domain, facet = load_mesh_h5py(head+str(ii))
		print('Loading [{:d}/{:d}]'.format(ct+1, num))
		ret = sa_hdf5.read_dolfin_mesh(head+str(ii))
		meshes.append(mesh)
		domains.append(domain)
		facets.append(facet)
		del mesh, domain, facet
		
		ct += 1
	return meshes, domains, facets

def load_patch_h5py(basename, contrast = 1e4, values = None, beta = 2.0, debug = False, orth_tensor = None, orth_angles = None, orth_axes = None):
	mesh, domains, facets = load_mesh_h5py(basename+'_'+str(beta))
	basedim = mesh.geometry().dim()

	arr = np.asarray(domains.array(), dtype = np.int32)
	arr_values = np.unique(arr)
	del domains

	inside = MeshFunction('size_t', mesh, basedim, 0)
	inside.array()[np.where((arr % 4) < 2)] = 1
	if debug:
		File('debug/{:s}_inside.pvd'.format(basename.replace('/','_').replace('.',''))) << inside

	coeff = None
	if orth_tensor is None or orth_angles is None or orth_axes is None:
		V0 = FunctionSpace(mesh, 'DG', 0)
		coeff = Function(V0)
		del V0
		if values is None:
			cc = [1., 1.*contrast, 1., 1.*contrast]
		else:
			cc = []
			tmp = []
			for ii in range(len(values)):
				cc.append(values[ii])
				tmp.append(values[ii])
				if ii % 2:
					cc += tmp
					tmp = []
		tmp = np.zeros_like(arr, dtype=float)
		for val in arr_values:
			tmp[np.where(arr == val)] = cc[val]
		coeff.vector().set_local(tmp)
		del tmp, cc
		if debug:
			mf = MeshFunction('double', mesh, basedim, 0)
#		   mf.array()[:] = coeff.vector().array()
			mf.array()[:] = coeff.vector().get_local() 
			File('debug/{:s}_kappa.pvd'.format(basename.replace('/','_').replace('.',''))) << mf
	else:
		VT = TensorFunctionSpace(mesh, 'DG', 0, (basedim, basedim, basedim, basedim))
	
	del arr
	return mesh, inside, coeff, facets 
"""