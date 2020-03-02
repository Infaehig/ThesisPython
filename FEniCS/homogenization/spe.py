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
#           num = mesh.num_cells()
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
            info("    [{:d}/{:d}={:.2f}%] of previous [{:d}/{:d}={:.2f}%] of total need refining".format(positive, num, positive*100./num, positive, num_cells, positive*100./num_cells))
            if not positive:
                break
            mf.array()[:] = 0
            mf.array()[which] = mf_array
            mfs.append(mf)
            info("    mesh function for refinement created")

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
            info("    refined")
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
#               inner_mesh = SubMesh(extended_mesh, inner)
#               del inner, extended_mesh
#               File(patch_name+'.xml') << inner_mesh
#               mesh_file = XDMFFile(comm, patch_name+'.xdmf')
#               mesh_file.write(inner_mesh)
       
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
