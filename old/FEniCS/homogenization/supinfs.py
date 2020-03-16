from dolfin import *

import logging
import sa_utils

comm = mpi_comm_world()
rank = MPI.rank(comm)
size = MPI.size(comm)

import sa_hdf5
import create_patches
import spe

import sympy as smp
import numpy as np
import scipy as sp
import scipy.linalg as la

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import time
import operator
import sys, os
import io
import gc
import multiprocessing

sys.setrecursionlimit(5000)

parameters["allow_extrapolation"] = True

from sa_utils import myeps
myeps2 = myeps*myeps

ortho_eps = myeps 
ortho_eps2 = ortho_eps*ortho_eps

efendiev_eps = myeps

debug = False
debug_lipton = False

direct_params = ('pastix',)
krylov_params = ('cg', 'hypre_amg')

log_solves = False

legend_rows = 2

def compute_supinfs(extended_mesh, extended_domains, extended_kappa, extended_facets, 
                    contrast = 1e4, elasticity = False, matplotlib = True,
                    keys = dict(), labels = dict() plotargs = dict(),
                    betas = [2.0], res
                    prefix = 'test', outdir = 'heat', ortho = True, 
                    logger = None, max_procs = None, batch_procs = None, max_keys = None, 
                    krylov = False, bc_case = 0):
    basedim = extended_mesh.geometry().dim()
    logger.info('['+str(basedim)+'] dimension mesh')

    for ii, key in enumerate(keys):
        if not key in labels:
            labels[key] = key
        if not key in plotargs:
            plotargs[key] = sa_utils.styles[ii%sa_utils.num_styles]+sa_utils.color_styles[ii%sa_utils.num_color_styles]

    cpu_count=multiprocessing.cpu_count() if max_procs is None else np.min([multiprocessing.cpu_count(), max_procs])

    sa_utils.makedirs_norace(outdir)

    extended_bmesh = BoundaryMesh(extended_mesh, 'exterior')
    extended_bmap = extended_bmesh.entity_map(basedim-1)

    fine_mesh = SubMesh(extended_mesh, extended_domains, 1)
    fine_bmesh = BoundaryMesh(fine_mesh, 'exterior')
    fine_bmap = fine_bmesh.entity_map(basedim-1)
   
    facets_array = extended_facets.array()
    fine_facets = MeshFunction('size_t', fine_mesh, basedim-1, 0)
    fine_facets_array = fine_facets.array()
    vmap = fine_mesh.data().array('parent_vertex_indices', 0)
    for fine_cell in cells(fine_bmesh):
        fine_facet = Facet(fine_mesh, fine_bmap[fine_cell.index()])
        fine_facet_vertices_set = set(vmap[fine_facet.entities(0)])
        for extended_cell in cells(extended_bmesh):
            extended_facet = Facet(extended_mesh, extended_bmap[extended_cell.index()])
            if set(extended_facet.entities(0)) == fine_facet_vertices_set:
                fine_facets_array[fine_facet.index()] = facets_array[extended_facet.index()]
                break
        del fine_facet_vertices_set
    logger.info('fine_facets_array:'+str(np.unique(fine_facets_array)))
    del extended_bmesh, extended_bmap, fine_bmesh, fine_bmap
    logger.info('submesh with [{:d}/{:d}={:.2f}%] cells'.format(fine_mesh.num_cells(), extended_mesh.num_cells(), 100*fine_mesh.num_cells()/extended_mesh.num_cells()))

    facets_array = extended_facets.array()

    nodes = fine_mesh.coordinates()
    low = np.min(nodes, axis=0)
    high = np.max(nodes, axis=0)
    del nodes
    lengths = high-low
    center = (low+high)*.5
    radius = np.sqrt(lengths.dot(lengths))*.5

    extended_nodes = extended_mesh.coordinates()
    extended_low = np.min(extended_nodes, axis=0)
    extended_high = np.max(extended_nodes, axis=0)
    del extended_nodes
    extended_lengths = extended_high-extended_low
    extended_center = (extended_high-extended_low)*.5
    extended_radius = np.sqrt(extended_lengths.dot(extended_lengths))*.5

    logger.info('Setup')
    # Global Test problem

    if elasticity:
        fine_VV = VectorFunctionSpace(fine_mesh, 'CG', 1, basedim)
    else:
        fine_VV = FunctionSpace(fine_mesh, 'CG', 1)
    fine_dim = fine_VV.dim()
    fine_hh = fine_mesh.hmax()
    fine_normal = FacetNormal(fine_mesh)
    fine_dx = Measure('dx', domain=fine_mesh)

    # On extended mesh
    extended_V0 = FunctionSpace(extended_mesh, 'DG', 0)
    if elasticity:
        extended_VV = VectorFunctionSpace(extended_mesh, 'CG', 1, basedim)
    else:
        extended_VV = FunctionSpace(extended_mesh, 'CG', 1)
    extended_dim = extended_VV.dim()
    extended_normal = FacetNormal(extended_mesh)
    
    extended_ds = Measure('ds', domain=extended_mesh, subdomain_data=extended_facets)
    extended_dx = Measure('dx', domain=extended_mesh, subdomain_data=extended_domains)

    logger.info('Forms')

    if elasticity:
        poisson = 0.3
        ll = (poisson*extended_kappa)/((1.+poisson)*(1.-2.*poisson))
        mu = extended_kappa/(2.*(1.+poisson))
        epsilon = lambda uu: sym(grad(uu))
        sigma = lambda uu: 2.*mu*epsilon(uu)+ll*tr(epsilon(uu))*Identity(basedim)
        zero = Constant([0]*basedim)
        ones = Constant([1]*basedim)

        VT = TensorFunctionSpace(fine_mesh, 'DG', 0)

        write_functions = sa_hdf5.write_dolfin_vector_cg1
    else:
        epsilon = lambda uu: grad(uu)
        sigma = lambda uu: extended_kappa*epsilon(uu)
        zero = Constant(0)
        ones = Constant(1)
        
        VT = VectorFunctionSpace(fine_mesh, 'DG', 0, basedim)

        write_functions = sa_hdf5.write_dolfin_scalar_cg1

    extended_trial = TrialFunction(extended_VV)
    extended_test = TestFunction(extended_VV)
   
    extended_kk = inner(sigma(extended_trial), epsilon(extended_test))*extended_dx
    extended_l2 = inner(extended_trial, extended_test)*dx

    interior_kk = inner(sigma(extended_trial), epsilon(extended_test))*extended_dx(1)
    interior_l2 = inner(extended_trial, extended_test)*extended_dx(1)
    interior_h1 = inner(grad(extended_trial), grad(extended_test))*extended_dx(1)

    extended_zero = assemble(inner(zero, extended_test)*extended_dx)
    extended_ones = interpolate(ones, extended_VV)
  
    extended_KK_nonzero = PETScMatrix()
    assemble(extended_kk, tensor=extended_KK_nonzero)
   
    fine_trial = TrialFunction(fine_VV)
    fine_test = TestFunction(fine_VV)

    fine_kk = inner(sigma(fine_trial), epsilon(fine_test))*fine_dx
    fine_l2 = inner(fine_trial, fine_test)*fine_dx
    fine_h1 = inner(grad(fine_trial), grad(fine_test))*fine_dx
 
    fine_zero = assemble(inner(zero, fine_test)*dx)
    fine_ones = interpolate(ones, fine_VV)

    extended_facet_plain = np.unique(facets_array)
    logger.info('Extended, plain: '+str(extended_facet_plain))
    del facets_array

    mask_domain_boundary = np.zeros(len(extended_facet_plain), dtype=bool)
    mask_patch_boundary = np.zeros(len(extended_facet_plain), dtype=bool)
    mask_domain_boundary[np.where((0 < extended_facet_plain)*(extended_facet_plain < 100))] = True
    mask_patch_boundary[np.where(0 < extended_facet_plain)] = True
    mask_patch_boundary[mask_domain_boundary] = False
    extended_facet_domain_boundary = extended_facet_plain[mask_domain_boundary]
    extended_facet_patch_boundary = extended_facet_plain[mask_patch_boundary]
    del mask_domain_boundary, mask_patch_boundary

    logger.info('Null space')
    if elasticity:
        if basedim == 2:
            null_expressions = [Constant((1, 0)), 
                                Constant((0, 1)), 
                                Expression(('-x[1]', 'x[0]'), degree=1)]
        elif basedim == 3:
            null_expressions = [Constant((1, 0, 0)), 
                                Constant((0, 1, 0)), 
                                Constant((0, 0, 1)), 
                                Expression(('0', '-x[2]', 'x[1]'), degree=1), 
                                Expression(('x[2]', '0', '-x[0]'), degree=1), 
                                Expression(('-x[1]', 'x[0]', '0'), degree=1)]
        else:
            raise NameError('Dimension ['+str(basedim)+'] null space not available')
    else:
        null_expressions = [Constant(1)]
    const_expressions = null_expressions
    null_dim = len(null_expressions)

    logger.info('Pure Neumann preparatory assembly')
    solver_params = None
    if krylov:
        solver_params = krylov_params
        extended_nullspace = sa_utils.build_nullspace(extended_VV, elasticity=elasticity)
        extended_KK_nonzero.set_nullspace(extended_nullspace)

        def neumann_extended_harmonic(uu):
            solve_log('            neumann dirichlet setup')
            AA = extended_KK_nonzero.copy()
            bb = extended_zero.copy()
            for kk in extended_facet_patch_boundary:
                bc = DirichletBC(extended_VV, uu, extended_facets, kk)
                bc.apply(AA, bb)
            AA.set_nullspace(extended_nullspace)
            extended_nullspace.orthogonalize(bb)
            vv = Function(extended_VV, name='u')
            solve_log('            neumann dirichlet solve')
            solve(AA, vv.vector(), bb, *krylov_params)
            del AA, bb
            solve_log('            neumann dirichlet return')
            return vv
    else:
        solver_params = direct_params
        if elasticity:
            extended_WV = VectorElement('CG', extended_mesh.ufl_cell(), 1, basedim)
        else:
            extended_WV = FiniteElement('CG', extended_mesh.ufl_cell(), 1)
        extended_WR = VectorElement('R', extended_mesh.ufl_cell(), 0, null_dim)
        extended_WW = FunctionSpace(extended_mesh, extended_WV*extended_WR)

        extended_uw, extended_cw = TrialFunctions(extended_WW)
        extended_vw, extended_dw = TestFunctions(extended_WW)
        extended_KW = PETScMatrix()
        assemble(inner(sigma(extended_uw), epsilon(extended_vw))*extended_dx-
                 (sum(extended_cw[ii]*inner(extended_vw, null_expressions[ii]) for ii in range(null_dim))-
                  sum(extended_dw[ii]*inner(extended_uw, null_expressions[ii]) for ii in range(null_dim)))*extended_dx(1), 
                  tensor = extended_KW)

        extended_bw = assemble(inner(extended_vw, zero)*dx)

        def neumann_extended_harmonic(uu):
            AA = extended_KW.copy()
            bb = extended_bw.copy()
            for kk in extended_facet_patch_boundary:
                bc = DirichletBC(extended_WW.sub(0), uu, extended_facets, kk)
                bc.apply(AA, bb)
            ww = Function(extended_WW)
            solve(AA, ww.vector(), bb, *solver_params)
            del AA, bb
            vv = Function(extended_VV, name='u')
            assign(vv, ww.sub(0))
            del ww
            return vv

    def fine_orthogonalize_null(uu):
        for vv in fine_const_vectors:
            uu.vector().axpy(-uu.vector().inner(fine_MM*vv),vv)

    def extended_orthogonalize_null(uu):
        for vv in extended_const_vectors:
            uu.vector().axpy(-uu.vector().inner(interior_MM*vv),vv)

    def fine_interpolate(uu):
        vv = interpolate(uu, fine_VV)
        vv.rename('u', 'label')
        return vv

    def extended_interpolate(uu):
        vv = interpolate(uu, extended_VV)
        vv.rename('u', 'label')
        return vv

    logger.info('Extracting free boundary dofs')
    boundary_dict = dict()
    num_vertices = extended_mesh.num_vertices()
    dofmap = vertex_to_dof_map(extended_VV)
    if elasticity:
        dofmap = [[dofmap[vertex*basedim+sub] for vertex in range(num_vertices)] for sub in range(basedim)]
        for kk in extended_facet_patch_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    for sub in range(basedim):
                        boundary_dict[dofmap[sub][vertex.index()]] = None
    else:
        for kk in extended_facet_patch_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    boundary_dict[dofmap[vertex.index()]] = None

    domain_boundary_num = len(extended_facet_domain_boundary)
    num_cases = 2**domain_boundary_num
    logger.info('bc case [{:d}/{:d}]'.format(bc_case+1,num_cases))
    gc.collect()
    mask_dirichlet = np.zeros(domain_boundary_num, dtype=bool)
    mask_neumann = np.zeros(domain_boundary_num, dtype=bool)
    binary = format(bc_ii,'0{:d}b'.format(domain_boundary_num))
    for jj in range(domain_boundary_num):
        if int(binary[jj]):
            mask_neumann[jj] = True
        else:
            mask_dirichlet[jj] = True
    extended_facet_dirichlet = extended_facet_domain_boundary[mask_dirichlet]
    extended_facet_neumann = extended_facet_domain_boundary[mask_neumann]
    del mask_dirichlet, mask_neumann
    if not len(extended_facet_dirichlet):
        bc_suffix = '/neumann'
    elif not len(extended_facet_neumann):
        bc_suffix = '/dirichlet'
    else:
        bc_suffix = '/'+binary

    outdir_bc = outdir+bc_suffix
    sa_utils.makedirs_norace(outdir_bc)

    logger.info('Extracting dirichlet boundary dofs')
    if len(extended_facet_dirichlet):
        free_dict = boundary_dict.copy()
        if elasticity:
            for kk in extended_facet_domain_boundary:
                for ff in SubsetIterator(extended_facets, kk):
                    for vertex in vertices(ff):
                        for sub in range(basedim):
                            tmp = dofmap[sub][vertex.index()]
                            if tmp in free_dict:
                                del free_dict[tmp]
        else:
            for kk in extended_facet_domain_boundary:
                for ff in SubsetIterator(extended_facets, kk):
                    for vertex in vertices(ff):
                        tmp = dofmap[vertex.index()]
                        if tmp in free_dict:
                            del free_dict[tmp]
    else:
        free_dict = boundary_dict
    boundary_dofs = sorted(free_dict.keys())
    del free_dict
    num_boundary = len(boundary_dofs)

    logger.info('extended mesh matrix assembly')
   
    extended_zero_bcs = [DirichletBC(extended_VV, zero, extended_facets, ii) for ii in extended_facet_dirichlet]
    
    _dummy = inner(ones, extended_test)*extended_dx
    _bb = PETScVector()
 
    extended_MM = PETScMatrix()
    assemble_system(extended_l2, _dummy, extended_zero_bcs, A_tensor=extended_MM, b_tensor=_bb)
    extended_KK = PETScMatrix()
    assemble_system(extended_kk, _dummy, extended_zero_bcs, A_tensor=extended_KK, b_tensor=_bb)

    interior_KK = PETScMatrix()
    assemble_system(interior_kk, _dummy, extended_zero_bcs, A_tensor=interior_KK, b_tensor=_bb)
    interior_MM = PETScMatrix()
    assemble_system(interior_l2, _dummy, extended_zero_bcs, A_tensor=interior_MM, b_tensor=_bb)
    interior_H1 = PETScMatrix()
    assemble_system(interior_h1, _dummy, extended_zero_bcs, A_tensor=interior_H1, b_tensor=_bb)

    del _bb, _dummy

    logger.info('fine mesh matrix assembly')
    fine_zero_bcs = [DirichletBC(fine_VV, zero, fine_facets, ii) for ii in extended_facet_dirichlet]

    _bb = PETScVector()
    _fine_dummy = inner(ones, fine_test)*dx
    fine_MM = PETScMatrix()
    assemble_system(fine_l2, _fine_dummy, fine_zero_bcs, A_tensor=fine_MM, b_tensor=_bb)
    fine_KK = PETScMatrix()
    assemble_system(fine_kk, _fine_dummy, fine_zero_bcs, A_tensor=fine_KK, b_tensor=_bb)
    fine_H1 = PETScMatrix()
    assemble_system(fine_h1, _fine_dummy, fine_zero_bcs, A_tensor=fine_H1, b_tensor=_bb)

    del _bb, _fine_dummy

    def orthogonalize(uu, uus, norm_KK):
        ret = uu.vector().inner(norm_KK*uu.vector())
        if ret < ortho_eps2:
            return 0
        ret = sqrt(ret)
        uu.vector()[:] /= ret
        if len(uus):
            for vv in uus:
                uu.vector().axpy(-uu.vector().inner(norm_KK*vv.vector()), vv.vector())
            ret = uu.vector().inner(norm_KK*uu.vector())
            if ret < ortho_eps2:
                return 0
            ret = sqrt(ret)
            uu.vector()[:] /= ret
        if -min(uu.vector()) > max(uu.vector()):
            uu.vector()[:] *= -1
        return ret

    fine_orthogonalize = lambda uu, uus: orthogonalize(uu, uus, fine_KK)
    extended_orthogonalize = lambda uu, uus: orthogonalize(uu, uus, interior_KK)

    logger.info('Computing coarse bases')
   
    logger.info('Null vectors')
    fine_const_vectors = [fine_interpolate(nu).vector() for nu in const_expressions]
    for vv in fine_const_vectors:
        vv[:] /= np.sqrt(vv.inner(fine_MM*vv))
    extended_const_vectors = [extended_interpolate(nu).vector() for nu in const_expressions]
    for vv in extended_const_vectors:
        vv[:] /= np.sqrt(vv.inner(extended_MM*vv))

    if len(extended_facet_dirichlet):
        def extended_harmonic(uu):
            solve_log('            dirichlet dirichlet setup')
            AA = extended_KK_nonzero.copy()
            bb = extended_zero.copy()
            for kk in extended_facet_patch_boundary:
                bc = DirichletBC(extended_VV, uu, extended_facets, kk)
                bc.apply(AA, bb)
            for bc in extended_zero_bcs:
                bc.apply(AA, bb)
            vv = Function(extended_VV, name='u')
            solve_log('            dirichlet dirichlet solve')
            solve(AA, vv.vector(), bb, *solver_params)
            del AA, bb
            solve_log('            dirichlet dirichlet return')
            return vv
    else:
        extended_harmonic = neumann_extended_harmonic

    logger.info('Building basis orthogonal to null space')
    old = sa_utils.get_time()
    def get_harmonic(kk):
        uu=Function(extended_VV)
        uu.vector()[kk] = 1.
        return extended_harmonic(uu)
    harmonic_hats = sa_utils.simple_batch_fun(boundary_dofs, extended_VV, low=0, high=num_boundary, fun=get_harmonic, logger=batch_logger, max_procs=batch_procs)
    new = sa_utils.get_time()
    time_harmonic_hats = new-old
    logger.info('Boundary basis constructed [{:d}] functions in [{:.2f}s={:s}]'.format(len(harmonic_hats),time_harmonic_hats,sa_utils.human_time(time_harmonic_hats)))

    old = sa_utils.get_time()
    for uu in harmonic_hats:
        extended_orthogonalize(uu, [])
    extended_orthogonalized = harmonic_hats
    interior_orthogonalized = sa_utils.simple_batch_fun(harmonic_hats, fine_VV, low=0, high=num_boundary, fun=fine_interpolate, logger=batch_logger, max_procs=batch_procs)
    del harmonic_hats
    extended_harmonic_dim = len(extended_orthogonalized)
    new = sa_utils.get_time()
    time_boundary_basis = new-old+time_dofmap+time_harmonic_hats
    logger.info('{:d} Basis for harmonic functions constructed in [{:.2f}s={:s}]'.format(extended_harmonic_dim,time_boundary_basis,sa_utils.human_time(time_boundary_basis)))

    old = sa_utils.get_time()
    extended_KK_harmonic = sa_utils.get_shared_array((extended_harmonic_dim, extended_harmonic_dim))
    interior_KK_harmonic = sa_utils.get_shared_array((extended_harmonic_dim, extended_harmonic_dim))
    interior_MM_harmonic = sa_utils.get_shared_array((extended_harmonic_dim, extended_harmonic_dim))
    interior_H1_harmonic = sa_utils.get_shared_array((extended_harmonic_dim, extended_harmonic_dim))

    times_dict = dict()
    times_queue = multiprocessing.Queue()
    def product_fun(low, high, times_q):
        old = sa_utils.get_time()
        for ii in range(low, high):
            extended_KK_harmonic[ii, ii] = extended_orthogonalized[ii].vector().inner(extended_KK*extended_orthogonalized[ii].vector())
            interior_KK_harmonic[ii, ii] = extended_orthogonalized[ii].vector().inner(interior_KK*extended_orthogonalized[ii].vector())
            interior_MM_harmonic[ii, ii] = extended_orthogonalized[ii].vector().inner(interior_MM*extended_orthogonalized[ii].vector())
            interior_H1_harmonic[ii, ii] = extended_orthogonalized[ii].vector().inner(interior_H1*extended_orthogonalized[ii].vector())
            for jj in range(ii):
                extended_KK_harmonic[ii, jj] = extended_orthogonalized[jj].vector().inner(extended_KK*extended_orthogonalized[ii].vector())
                interior_KK_harmonic[ii, jj] = extended_orthogonalized[jj].vector().inner(interior_KK*extended_orthogonalized[ii].vector())
                interior_MM_harmonic[ii, jj] = extended_orthogonalized[jj].vector().inner(interior_MM*extended_orthogonalized[ii].vector())
                interior_H1_harmonic[ii, jj] = extended_orthogonalized[jj].vector().inner(interior_H1*extended_orthogonalized[ii].vector())
                extended_KK_harmonic[jj, ii] = extended_KK_harmonic[ii, jj]
                interior_KK_harmonic[jj, ii] = interior_KK_harmonic[ii, jj]
                interior_MM_harmonic[jj, ii] = interior_MM_harmonic[ii, jj]
                interior_H1_harmonic[jj, ii] = interior_H1_harmonic[ii, jj]
        new = sa_utils.get_time()
        times_dict[low] = new-old
        times_q.put(times_dict)
    if cpu_count > 1:
        processes = []
        block = 20
        low = 0
        high = np.min([low+block, extended_harmonic_dim])
        logger.info('Creating processes for matrix assembly')
        while(low < high):
            processes.append(multiprocessing.Process(target=product_fun, args=(low, high, times_queue)))
            low = high
            high = np.min([low+block, extended_harmonic_dim])
        del block, low, high
        process_count = len(processes)
        logger.info('Created ['+str(process_count)+'] processes for matrix assembly')
        started = 0
        fetched = 0
        while(started < np.min([cpu_count, process_count])):
            processes[started].start()
            started += 1
        while(fetched < process_count):
            times_dict.update(times_queue.get())
            fetched += 1
            if started < process_count:
                processes[started].start()
                started += 1
        logger.info('Processes fetched')
        for kk in range(process_count):
            processes[kk].join()
            processes[kk] = None
        del processes, process_count, started, fetched
    else:
        product_fun(0, extended_harmonic_dim, times_queue)
        times_dict.update(times_queue.get())
    for tt in times_dict.values():
        old -= tt
    del product_fun, times_dict, times_queue
    new = sa_utils.get_time()
    time_boundary_matrices = new-old
    gc.collect()
    logger.info('Harmonic Polynomial matrices set up in [{:.2f}s={:s}]'.format(time_boundary_matrices,sa_utils.human_time(time_boundary_matrices)))

    min_dof = np.min([dofs[key][0] for key in keys])
    max_dof = np.max([dofs[key][-1] for key in keys])

    logger.info('Energy optimal')
    eigvals, eigvecs = la.eigh(interior_KK_harmonic, extended_KK_harmonic)
    idx = np.argsort(eigvals)[::-1]
    eigvals_max = eigvals[idx[0]]
    max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
    logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
    nwidthsE = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
    nwidthE_limit = len(nwidthsE)

    logger.info('L2 optimal')
    eigvals, eigvecs = la.eigh(interior_MM_harmonic, extended_KK_harmonic)
    idx = np.argsort(eigvals)[::-1]
    eigvals_max = eigvals[idx[0]]
    max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
    logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
    nwidths0 = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
    nwidth0_limit = len(nwidths0)

    logger.info('H1 optimal')
    eigvals, eigvecs = la.eigh(interior_H1_harmonic, extended_KK_harmonic)
    idx = np.argsort(eigvals)[::-1]
    eigvals_max = eigvals[idx[0]]
    max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
    logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
    nwidths1 = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
    nwidth1_limit = len(nwidths1)

    logger.info('Removing linearly dependent shape functions')
    logger.info('Ortho: '+str(ortho))
    for key in keys:
        count = 0
        orthogonalized = []
        new_uus = []
        new_dofs = []
        tmp_dofs = dofs[key]
        tmp_uus = uus[key]
        logger.info('['+key+'] starting, ['+str(len(tmp_uus))+'] functions preset')
        last = 0
        count_null = 0
        for deg in range(len(tmp_dofs)):
            old = sa_utils.get_time()
            current = tmp_dofs[deg]
            last_count = count
            for kk in range(last, current):
                uu = Function(fine_VV, tmp_uus[kk], name='u')
                ret = fine_orthogonalize(uu, orthogonalized)
                if not len(extended_facet_dirichlet):
                    fine_orthogonalize_null(uu)
                if ret < ortho_eps:
                    count_null += 1
                else:
                    orthogonalized.append(uu)
                    if ortho:
                        new_uus.append(uu)
                    else:
                        new_uus.append(tmp_uus[kk])
                    count += 1
            new = sa_utils.get_time()
            if count > last_count:
                new_dofs.append(count)
            last = current
        logger.info('['+key+'] finished, ['+str(count_null)+'] functions removed ['+str(count)+'] remaining')
        dofs[key] = np.array(new_dofs)
        uus[key] = new_uus
        del orthogonalized, new_uus, new_dofs, tmp_uus
        logger.info('['+key+'] old dofs:'+str(tmp_dofs))
        del tmp_dofs
        logger.info('['+key+'] dofs:    '+str(dofs[key]))

    logger.info('Filtering with respect to nwidths and stiffness condition')
    logger.info('Limits: ['+str(nwidthE_limit)+', '+str(nwidth0_limit)+', '+str(nwidth1_limit)+']')
    remove_keys = []
    for key in keys:
        logger.info('['+key+'] starting')
        logger.info('['+key+'] dofs: '+str(dofs[key]))
        count = 0
        broken = False
        HH = None
        if len(dofs[key]):
            dof0 = dofs[key][-1]
        else:
            dof0 = 0
        dof1 = 0
        while True:
            if len(dofs[key]):
                dof1 = dofs[key][-1]
                if dofs[key][-1] >= nwidthE_limit or dofs[key][-1] >= nwidth0_limit or dofs[key][-1] >= nwidth1_limit or (len(dofs[key]) > 1 and dofs[key][-2] >= dofmax):
                    count += 1
                    dofs[key] = dofs[key][:-1]
                    continue
                coarse_deg = len(dofs[key])
                coarse_dim = dofs[key][coarse_deg-1]
                if HH is None:
                    funs = coarse_spaces[key][coarse_deg-1]
                    HH = np.zeros((coarse_dim, coarse_dim))
                    for ii in range(coarse_dim):
                        HH[ii, ii] = funs[ii].vector().inner(fine_KK*funs[ii].vector())
                        for jj in range(ii):
                            HH[ii, jj] = funs[jj].vector().inner(fine_KK*funs[ii].vector())
                            HH[jj, ii] = HH[ii, jj]
                else:
                    funs = None
                    HH = HH[:coarse_dim, :coarse_dim]
                eigvals = la.eigvalsh(HH)
                maxval = np.max(eigvals)
                minval = np.min(eigvals)
                if minval < ortho_eps2*maxval:
                    count += 1
                    dofs[key] = dofs[key][:-1]
                    continue
                break
            else:
                dof1 = 0
                funs = None; eigvals = None; maxval = None; minval = None;
                broken = True
                break
        del funs, HH, eigvals, maxval, minval
        if broken:
            remove_keys.append(key)
            logger.info('['+key+'] will be removed')
        else:
            logger.info('['+key+'] shortened by ['+str(count)+'] degrees from ['+str(dof0)+'] to ['+str(dof1)+']')
            logger.info('['+key+'] lengths: ['+str(dofs[key][-1])+'] < ['+str(nwidthE_limit)+', '+str(nwidth0_limit)+', '+str(nwidth1_limit)+']')
    for key in remove_keys:
        keys.remove(key)
    numkeys = len(keys)

    min_dof = np.min([dofs[key][0] for key in keys])
    max_dof = np.max([dofs[key][-1] for key in keys])
    num_fill = int(np.log(max_dof)/np.log(10.))+1

    logger.info('Writing nwidths')
    nw_Err = (np.log(nwidthsE[-1])-np.log(nwidthsE[min_dof]))/(np.log(len(nwidthsE)-1)-np.log(min_dof))
    nw_1rr = (np.log(nwidths1[-1])-np.log(nwidths1[min_dof]))/(np.log(len(nwidths1)-1)-np.log(min_dof))
    nw_0rr = (np.log(nwidths0[-1])-np.log(nwidths0[min_dof]))/(np.log(len(nwidths0)-1)-np.log(min_dof))
    matfile = open(outdir_bc+'/'+prefix+'_nwidths.csv', 'w')
    matfile.write('dof, nwidthE, nw_Err, nwidth1, nw1rr, nwidth0, nw_0rr\n')
    matfile.write('{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n'.format(extended_dim, 0, nw_Err, 0, nw_1rr, 0, nw_0rr))
    matfile.write('{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n'.format(min_dof, nwidthsE[min_dof], 0, nwidths1[min_dof], 0, nwidths0[min_dof], 0))
    for ii in range(min_dof+1, np.min([nwidthE_limit, nwidth0_limit, nwidth1_limit])):
        denom = 1./np.log(1.*ii/(ii-1.))
        suprrE=ln(nwidthsE[ii]/nwidthsE[ii-1])*denom
        suprr1=ln(nwidths0[ii]/nwidths0[ii-1])*denom
        suprr0=ln(nwidths0[ii]/nwidths0[ii-1])*denom
        matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n".format(ii, nwidthsE[ii], suprrE, nwidths1[ii], suprr1, nwidths0[ii], suprr0))
    matfile.close()

    supinfsE = dict()
    supinfs0 = dict()
    supinfs1 = dict()
    conditions_stiffness = dict()
    supinfsE_rates = dict()
    supinfs0_rates = dict()
    supinfs1_rates = dict()
    conditions_stiffness_rates = dict()

    gc.collect()
    
    logger.info('Computing stuff with coarse bases')
    keys_count=multiprocessing.cpu_count() if max_keys is None else np.min([multiprocessing.cpu_count(), max_keys])

    def compute_stuff_for_key(key, supinfsE_q, supinfs0_q, supinfs1_q, 
                              conditions_stiffness_q, 
                              supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, 
                              conditions_stiffness_rates_q,
                              done_q):
        logger.info('['+key+'] major assembly starting')

        finest_dim = dofs[key][-1]
        finest_space = uus[key][:finest_dim]
        HH_E = sa_utils.get_shared_array((finest_dim, finest_dim)) 
        HH_1 = sa_utils.get_shared_array((finest_dim, finest_dim)) 
        HH_0 = sa_utils.get_shared_array((finest_dim, finest_dim)) 
        TT_E = sa_utils.get_shared_array((finest_dim, extended_harmonic_dim))
        TT_1 = sa_utils.get_shared_array((finest_dim, extended_harmonic_dim))
        TT_0 = sa_utils.get_shared_array((finest_dim, extended_harmonic_dim))

        matrix_queue = multiprocessing.Queue()
        def matrix_populate(low, high, qq):
            for ii in range(low, high):
                HH_E[ii, ii] = finest_space[ii].vector().inner(fine_KK*finest_space[ii].vector())
                HH_1[ii, ii] = finest_space[ii].vector().inner(fine_H1*finest_space[ii].vector())
                HH_0[ii, ii] = finest_space[ii].vector().inner(fine_MM*finest_space[ii].vector())
                for jj in range(ii):
                    HH_E[ii, jj] = finest_space[jj].vector().inner(fine_KK*finest_space[ii].vector())
                    HH_E[jj, ii] = HH_E[ii, jj]
                    HH_1[ii, jj] = finest_space[jj].vector().inner(fine_H1*finest_space[ii].vector())
                    HH_1[jj, ii] = HH_1[ii, jj]
                    HH_0[ii, jj] = finest_space[jj].vector().inner(fine_MM*finest_space[ii].vector())
                    HH_0[jj, ii] = HH_0[ii, jj]
                for jj in range(extended_harmonic_dim):
                    TT_E[ii, jj] = finest_space[ii].vector().inner(fine_KK*interior_orthogonalized[jj].vector())
                    TT_1[ii, jj] = finest_space[ii].vector().inner(fine_H1*interior_orthogonalized[jj].vector())
                    TT_0[ii, jj] = finest_space[ii].vector().inner(fine_MM*interior_orthogonalized[jj].vector())
            qq.put(None)
        if cpu_count > 1:
            matrix_processes = []
            block = 4
            low = 0
            high = np.min([low+block, finest_dim])
            while(low < high):
                matrix_processes.append(multiprocessing.Process(target=matrix_populate, args=(low, high, matrix_queue)))
                low = high
                high = np.min([low+block, finest_dim])
            del block, low, high
            process_count = len(matrix_processes)
            started = 0
            fetched = 0
            while(started < np.min([cpu_count, process_count])):
                matrix_processes[started].start()
                started += 1
            while(fetched < process_count):
                matrix_queue.get()
                fetched += 1
                if started < process_count:
                    matrix_processes[started].start()
                    started += 1
            assert(started == fetched)
            assert(started == process_count)
            for ii in range(process_count):
                matrix_processes[ii].join()
                matrix_processes[ii] = None
            del matrix_processes, process_count, started, fetched
        else:
            matrix_populate(0, finest_dim, matrix_queue)
            matrix_queue.get()
        del matrix_queue, matrix_populate
        gc.collect()

        logger.info('['+key+'] computations starting')
        dof_len = len(dofs[key])
        supinfsE[key] = sa_utils.get_shared_array(dof_len)
        supinfs0[key] = sa_utils.get_shared_array(dof_len)
        supinfs1[key] = sa_utils.get_shared_array(dof_len)
        conditions_stiffness[key] = sa_utils.get_shared_array(dof_len)
        conditions_stiffness_rates[key] = []

        eval_queue = multiprocessing.Queue()

        def embedded_fun(low, high, qq):
            for deg in range(low, high):
                coarse_dim = dofs[key][deg]
                logger.info('['+key+'], degree ['+str(deg)+'], dim ['+str(coarse_dim)+'] started')

                HH = HH_E[:coarse_dim, :coarse_dim]
                eigvals = la.eigvalsh(HH)
                maxval = np.max(eigvals)
                minval = np.min(eigvals)
                conditions_stiffness[key][deg] = maxval/minval

                TT = TT_E[:coarse_dim]
                Mtilde = interior_KK_harmonic - TT.T.dot(la.solve(HH, TT))
                main = np.diag(Mtilde).copy(); diag = np.ones(extended_harmonic_dim)
                where = np.where(main > 0)
                diag[where] = 1./np.sqrt(main[where])
                eigvals = la.eigvalsh(diag[:, np.newaxis]*Mtilde*diag, diag[:, np.newaxis]*extended_KK_harmonic*diag)
                supinfsE[key][deg] = np.sqrt(eigvals[-1])
                del HH, TT, Mtilde, eigvals

                HH = HH_0[:coarse_dim, :coarse_dim]
                TT = TT_0[:coarse_dim]
                Mtilde = interior_MM_harmonic - TT.T.dot(la.solve(HH, TT))
                main = np.diag(Mtilde).copy(); diag = np.ones(extended_harmonic_dim)
                where = np.where(main > 0)
                diag[where] = 1./np.sqrt(main[where])
                eigvals = la.eigvalsh(diag[:, np.newaxis]*Mtilde*diag, diag[:, np.newaxis]*extended_KK_harmonic*diag)
                supinfs0[key][deg] = np.sqrt(eigvals[-1])
                del HH, TT, Mtilde, eigvals

                HH = HH_1[:coarse_dim, :coarse_dim]
                TT = TT_1[:coarse_dim]
                Mtilde = interior_H1_harmonic - TT.T.dot(la.solve(HH, TT))
                main = np.diag(Mtilde).copy(); diag = np.ones(extended_harmonic_dim)
                where = np.where(main > 0)
                diag[where] = 1./np.sqrt(main[where])
                eigvals = la.eigvalsh(diag[:, np.newaxis]*Mtilde*diag, diag[:, np.newaxis]*extended_KK_harmonic*diag)
                supinfs1[key][deg] = np.sqrt(eigvals[-1])
                del HH, TT, Mtilde, eigvals

                logger.info('['+key+'], degree ['+str(deg)+'], dim ['+str(coarse_dim)+'] finished')

            qq.put(None)
        if keys_count > 1:
            eval_processes = []
            block = 4
            low = 0
            high = np.min([low+block, dof_len])
            while(low < high):
                eval_processes.append(multiprocessing.Process(target=embedded_fun, args=(low, high, eval_queue)))
                low = high
                high = np.min([low+block, dof_len])
            del block, low, high
            process_count = len(eval_processes)
            started = 0
            fetched = 0
            while(started < np.min([multiprocessing.cpu_count(), process_count])):
                eval_processes[started].start()
                started += 1
            while(fetched < process_count):
                eval_queue.get()
                fetched += 1
                if started < process_count:
                    eval_processes[started].start()
                    started += 1
            for ii in range(process_count):
                eval_processes[ii].join()
                eval_processes[ii] = None
            del eval_processes, process_count, started, fetched
        else:
            embedded_fun(0, dof_len, eval_queue)
            eval_queue.get()
        del eval_queue, HH_E, HH_1, HH_0, TT_E, TT_1, TT_0, embedded_fun
        gc.collect()

        logger.info('['+key+'] Computing errors, rates etc')

        basename = prefix+'_'+key
        supinfE = supinfsE[key]
        supinf0 = supinfs0[key]
        supinf1 = supinfs1[key]
        dof = dofs[key]

        denom = 1./(np.log(dof[-1])-np.log(dof[0]))
        suprrE = (np.log(supinfE[-1])-np.log(supinfE[0]))*denom
        supinfsE_rates[key] = suprrE
        suprr0 = (np.log(supinf0[-1])-np.log(supinf0[0]))*denom
        supinfs0_rates[key] = suprr0
        suprr1 = (np.log(supinf1[-1])-np.log(supinf1[0]))*denom
        supinfs1_rates[key] = suprr1
        conditions_stiffness_r = (np.log(conditions_stiffness[key][-1])-np.log(conditions_stiffness[key][0]))*denom
        conditions_stiffness_rates[key] = conditions_stiffness_r
        
        matfile = open(outdir_bc+'/'+basename+'_supinfs.csv', 'w')
        matfile.write('dof, supinfE, lambdaE, supinfEr, supinf1, lambda1, supinf1r, supinf0, lambda0, supinf0r, condstiffness\n')
        matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}\n'.format(fine_dim, 0, 0, suprrE, 0, 0, suprr1, 0, 0, suprr0, 0))
        matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}\n'.format(dof[0], supinfE[0], supinfE[0]/nwidthsE[dof[0]], 0, supinf1[1], supinf1[0]/nwidths1[dof[0]], 0, supinf0[0], supinf0[0]/nwidths0[dof[0]], 0, conditions_stiffness[key][0]))
        for ii in range(1, len(dof)):
            denom = 1./np.log(1.*dof[ii]/dof[ii-1])
            suprrE=ln(supinfE[ii]/supinfE[ii-1])*denom
            suprr1=ln(supinf1[ii]/supinf1[ii-1])*denom
            suprr0=ln(supinf0[ii]/supinf0[ii-1])*denom
            matfile.write("{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}\n".format(dof[ii], supinfE[ii], supinfE[ii]/nwidthsE[dof[ii]], suprrE, supinf1[ii], supinf1[ii]/nwidths1[dof[ii]], suprr1, supinf0[ii], supinf0[ii]/nwidths0[dof[ii]], suprr0, conditions_stiffness[key][ii]))
        matfile.close()

        logger.info('['+key+'] global rates computed')

        supinfsE_q.put(supinfsE); supinfs0_q.put(supinfs0); supinfs1_q.put(supinfs1)
        conditions_stiffness_q.put(conditions_stiffness)
        supinfsE_rates_q.put(supinfsE_rates); supinfs0_rates_q.put(supinfs0_rates); supinfs1_rates_q.put(supinfs1_rates)
        conditions_stiffness_rates_q.put(conditions_stiffness_rates)
        logger.info('['+key+'] everything put in queues')

        done_q.put(None)

    done_q = multiprocessing.Queue()

    compute_processes = []
    supinfsE_q = multiprocessing.Queue(); supinfs0_q = multiprocessing.Queue(); supinfs1_q = multiprocessing.Queue(); 
    conditions_stiffness_q = multiprocessing.Queue()
    supinfsE_rates_q = multiprocessing.Queue(); supinfs0_rates_q = multiprocessing.Queue(); supinfs1_rates_q = multiprocessing.Queue()
    conditions_stiffness_rates_q = multiprocessing.Queue()
    if keys_count > 1:
        for key in keys:
            proc = multiprocessing.Process(target=compute_stuff_for_key, args=(key, supinfsE_q, supinfs0_q, supinfs1_q, 
                                                                               conditions_stiffness_q, 
                                                                               supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, 
                                                                               conditions_stiffness_rates_q,
                                                                               done_q))
            compute_processes.append(proc)
            proc.start()
        logger.info('processes started')
        if compute_supinfs:
            for (kk, proc) in enumerate(compute_processes):
                done_q.get()
                logger.info('process [{:d}] done'.format(kk+1))
                supinfsE.update(supinfsE_q.get()); supinfs0.update(supinfs0_q.get()); supinfs1.update(supinfs1_q.get()); 
                conditions_stiffness.update(conditions_stiffness_q.get())
                supinfsE_rates.update(supinfsE_rates_q.get()); supinfs0_rates.update(supinfs0_rates_q.get()); supinfs1_rates.update(supinfs1_rates_q.get())
                conditions_stiffness_rates.update(conditions_stiffness_rates_q.get())
                logger.info('process [{:d}] data fetches'.format(kk+1))
        else:
            for (kk, proc) in enumerate(compute_processes):
                done_q.get()
                logger.info('process [{:d}] done'.format(kk+1))
        logger.info('data fetched')
        for kk, proc in enumerate(compute_processes):
            proc.join()
            proc = None
            compute_processes[kk] = None
        logger.info('All processes joined')
    else:
        for key in keys:
            compute_stuff_for_key(key, supinfsE_q, supinfs0_q, supinfs1_q, 
                                  conditions_stiffness_q, 
                                  supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, 
                                  conditions_stiffness_rates_q,
                                  done_q)
            done_q.get()
            logger.info('process [{:d}] done'.format(kk+1))
            if compute_supinfs:
                supinfsE.update(supinfsE_q.get()); supinfs0.update(supinfs0_q.get()); supinfs1.update(supinfs1_q.get()); 
                conditions_stiffness.update(conditions_stiffness_q.get())
                supinfsE_rates.update(supinfsE_rates_q.get()); supinfs0_rates.update(supinfs0_rates_q.get()); supinfs1_rates.update(supinfs1_rates_q.get())
                conditions_stiffness_rates.update(conditions_stiffness_rates_q.get())
                logger.info('process [{:d}] data fetches'.format(kk+1))
        logger.info('all keys done')

    del supinfsE_q, supinfs0_q, supinfs1_q, conditions_stiffness_q, supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, conditions_stiffness_rates_q, done_q
    del compute_processes, fine_MM, fine_KK, fine_H1, interior_H1_harmonic, interior_MM_harmonic, interior_KK_harmonic, extended_KK_harmonic
    gc.collect()

    if matplotlib:
        logger.info('matplotlib plots')
        minx = min_dof; maxx = max_dof
        doflimits = np.array([min_dof, np.sqrt(min_dof*max_dof), max_dof], dtype=float)
        dofrange=np.arange(min_dof, max_dof+1, dtype=int)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = 0.9; maxy = 2.;
        for key in keys:
            supinf = supinfsE[key]
            nE = nwidthsE[dofs[key]]
            handles.append(*ax.loglog(dofs[key], supinf/nE, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/nE); ymax = np.max(supinf/nE)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^E(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambdaE.pdf')

        figlegend = plt.figure(figsize=(1.05*np.ceil(numkeys/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil(numkeys/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_legend.pdf', bbox_extra_artists=(lgd, ))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = .9; maxy = 2.;
        for key in keys:
            supinf = supinfs0[key]
            n0 = nwidths0[dofs[key]]
            handles.append(*ax.loglog(dofs[key], supinf/n0, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n0); ymax = np.max(supinf/n0)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^0(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambda0.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = .9; maxy = 2.;
        for key in keys:
            supinf = supinfs1[key]
            n1 = nwidths1[dofs[key]]
            handles.append(*ax.loglog(dofs[key], supinf/n1, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n1); ymax = np.max(supinf/n1)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^1(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambda1.pdf')

        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        ymin = np.min([np.min(conditions_stiffness[key]) for key in keys])
        ymax = np.max([np.max(conditions_stiffness[key]) for key in keys])
        miny0 = np.min([conditions_stiffness[key][0] for key in keys])
        maxy0 = np.max([conditions_stiffness[key][0] for key in keys])
        minr = np.min([conditions_stiffness_rates[key] for key in keys])
        maxr = np.max([conditions_stiffness_rates[key] for key in keys])
        minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
        for key in keys:
            handles.append(*ax.loglog(dofs[key], conditions_stiffness[key], plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
        handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
        handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label='$c n^{'+str("%.2g"%maxr)+r'}$'))
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, ymin, ymax)
        ax.set_ylabel(r'$\operatorname{cond}(K)$')
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        fig.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness.pdf')

        figlegend = plt.figure(figsize=(1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+2)/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_legend.pdf', bbox_extra_artists=(lgd, ))

        plt.close('all')

        logger.info('semilog plots')
        minx = min_dof; maxx = max_dof
        doflimits = np.array([min_dof, np.sqrt(min_dof*max_dof), max_dof], dtype=float)
        dofrange=np.arange(min_dof, max_dof+1, dtype=int)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidthsE[dofrange]); maxy = np.max(nwidthsE[dofrange]);
        for key in keys:
            supinf = supinfsE[key]
            rr = supinfsE_rates[key]
            handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        handles.append(*ax.semilogy(dofrange, nwidthsE[dofrange], 'k-', mfc='none', label='$d^E_{l, n}$'))
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Psi_l^E(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_supinfsE_semilogy.pdf')

        figlegend = plt.figure(figsize=(np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_supinfsE_legend_semilogy.pdf', bbox_extra_artists=(lgd, ))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = 0.9; maxy = 2.;
        for key in keys:
            supinf = supinfsE[key]
            nE = nwidthsE[dofs[key]]
            handles.append(*ax.semilogy(dofs[key], supinf/nE, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/nE); ymax = np.max(supinf/nE)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^E(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambdaE_semilogy.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidths0[dofrange]); maxy = np.max(nwidths0[dofrange]);
        for key in keys:
            supinf = supinfs0[key]
            handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        handles.append(*ax.semilogy(dofrange, nwidths0[dofrange], 'k-', mfc='none', label='$d^0_{l, n}$'))
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Psi_l^0(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_supinfs0_semilogy.pdf')

        figlegend = plt.figure(figsize=(np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs0_legend_semilogy.pdf', bbox_extra_artists=(lgd, ))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = .9; maxy = 2.;
        for key in keys:
            supinf = supinfs0[key]
            n0 = nwidths0[dofs[key]]
            handles.append(*ax.semilogy(dofs[key], supinf/n0, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n0); ymax = np.max(supinf/n0)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^0(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambda0_semilogy.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths0[dofrange]);
        for key in keys:
            supinf = supinfs1[key]
            handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        handles.append(*ax.semilogy(dofrange, nwidths1[dofrange], 'k-', mfc='none', label='$d^1_{l, n}$'))
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Psi_l^1(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_supinfs1_semilogy.pdf')

        figlegend = plt.figure(figsize=(np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs1_legend_semilogy.pdf', bbox_extra_artists=(lgd, ))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = .9; maxy = 2.;
        for key in keys:
            supinf = supinfs1[key]
            n1 = nwidths1[dofs[key]]
            handles.append(*ax.semilogy(dofs[key], supinf/n1, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n1); ymax = np.max(supinf/n1)
            miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        ax.set_ylabel(r'$\Lambda_l^1(\mathcal{V}_l)$')
        fig.savefig(outdir_bc+'/'+prefix+'_lambda1_semilogy.pdf')

        plt.close('all')

        mint = np.min([np.min(times_compute[key]) for key in keys]); maxt = np.max([np.max(times_compute[key]) for key in keys])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        ymin = np.min([np.min(conditions_stiffness[key]) for key in keys])
        ymax = np.max([np.max(conditions_stiffness[key]) for key in keys])
        for key in keys:
            handles.append(*ax.semilogy(dofs[key], conditions_stiffness[key], plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
        sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
        sa_utils.set_log_ticks(ax, ymin, ymax)
        ax.set_ylabel(r'$\operatorname{cond}(K)$')
        ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
        fig.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_semilogy.pdf')

        figlegend = plt.figure(figsize=(1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+2)/legend_rows)))
        figlegend.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_legend_semilogy.pdf', bbox_extra_artists=(lgd, ))

        plt.close('all')

        for key in keys:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            miny = np.min([np.min(nwidthsE[dofrange]), np.min(nwidths0[dofrange]),
                           np.min(supinfsE[key]), np.min(supinfs0[key])])
            maxy = np.max([np.max(nwidthsE[max_dof:max_dof+1]), np.max(nwidths0[max_dof:max_dof+1]),
                           np.max(supinfsE[key]), np.max(supinfs0[key])])
            handles = []
            handles.append(*ax.semilogy(dofrange, nwidthsE[dofrange], 'r-', mfc='none', label='$d^E_{l, n}$', alpha=0.7))
            handles.append(*ax.semilogy(dofs[key], supinfsE[key], 'ro', mec='r', mfc='none', label=r'$\Psi_l^E(\mathcal{V}_l)$'))
            handles.append(*ax.semilogy(dofrange, nwidths1[dofrange], 'b-', mfc='none', label='$d^1_{l, n}$', alpha=0.7))
            handles.append(*ax.semilogy(dofs[key], supinfs1[key], 'bs', mec='b', mfc='none', label=r'$\Psi_l^1(\mathcal{V}_l)$'))
            handles.append(*ax.semilogy(dofrange, nwidths0[dofrange], 'g-', mfc='none', label='$d^0_{l, n}$', alpha=0.7))
            handles.append(*ax.semilogy(dofs[key], supinfs0[key], 'g+', mec='g', mfc='none', label=r'$\Psi_l^0(\mathcal{V}_l)$'))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis=True, semilog=True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_title(names[key])
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^i(\mathcal{V}_l)$')
            fig.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_semilogy.pdf')
            
            figlegend = plt.figure(figsize=(3.5*sa_utils.legendx, 2*sa_utils.legendy), frameon=False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc=10, ncol=3)
            figlegend.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_legend_semilogy.pdf', bbox_extra_artists=(lgd, ))
 
            plt.close('all')
    
    del fig, ax, figlegend, handles, labels, lgd
    logger.info('matplotlib done')

    del extended_facet_dirichlet
    del extended_facet_neumann
    logger.info('localizations_nitsche patch done')

def patch_script(meshdir, prefix, level, patch_level, outdir, dofmax=100, write_shapes=False, patches=None, 
                 logger=None, minimal=1, ortho=True, heat=True, elasticity=True, 
                 spe=False, contrast=1e4, beta=2.0,
                 max_procs=1, patch_procs=None, max_keys=None, batch_procs=None,
                 batch_logging=False):
    prefix_level = prefix+'_'+str(level)
    logger.info('Starting patch_script ['+prefix_level+', '+str(patch_level)+']')
    patch_dir = meshdir+'/'+prefix_level+'_patches'

    patch_boxes = np.genfromtxt(meshdir+'/'+prefix+'_patch_descriptors/'+str(patch_level)+'.csv', delimiter=',', names=True)
    patch_num = len(patch_boxes['idx'])
    patch_fill = int(np.log(patch_num)/np.log(10.))+1

    all_args = range(patch_num)

    if patches is None:
        args = all_args
    else:
        args = []
        for ii in patches:
            if ii in all_args:
                args.append(ii)
    len_args = len(args)
    if len_args < 1:
        logger.info('patches [{:s}] do not exist in patches [{:s}] for [{:s}]'.format(str(patches), str(all_args), patch_dir))
        return
    del all_args

    def compute_fun(ii, elasticity, qq):
        patch_name = str(patch_level)+'/patch_'+str(ii).zfill(patch_fill)

        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] started')
        basename = patch_dir+'/'+patch_name
        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] loading ['+basename+']')
        if spe:
            extended_mesh, extended_domains, extended_coeff, extended_facets = spe.load_patch(basename)
        else:
            extended_mesh, extended_domains, extended_coeff, extended_facets = create_patches.load_patch_h5py(basename, contrast=contrast, beta=beta)
        basedim = extended_mesh.geometry().dim()

        basename = prefix+'_'+str(level)+'_'+str(patch_level)+'_patch_'+str(ii).zfill(patch_fill)
        localization_test(extended_mesh, extended_domains, extended_coeff, extended_facets,
                          dofmax=dofmax*(basedim if elasticity else 1), elasticity=elasticity, write_shapes=write_shapes, prefix=basename, matplotlib=True, 
                          outdir=outdir+('/elasticity/' if elasticity else '/heat/')+patch_name, 
                          logger=logger, minimal=minimal, ortho=ortho, spe=spe, max_procs=patch_procs, batch_procs=batch_procs, max_keys=max_keys,
                          batch_logging=batch_logging)
        qq.put(None)
        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] finished')

    block = multiprocessing.cpu_count() if max_procs is None else np.min([max_procs,multiprocessing.cpu_count()])

    start = sa_utils.get_time()
    if heat:
        logger.info('Starting Heat')
        queue = multiprocessing.Queue()
        if block > 1:
            compute_processes = []
            for ii in args:
                compute_processes.append(multiprocessing.Process(target=compute_fun, args=(ii, False, queue)))
            block_low = 0; block_high = np.min([block, len_args])
            while(block_low < block_high):
                for kk in range(block_low, block_high):
                    compute_processes[kk].start()
                for kk in range(block_low, block_high):
                    queue.get()
                for kk in range(block_low, block_high):
                    compute_processes[kk].join()
                    compute_processes[kk] = None
                block_low = block_high
                block_high = np.min([block_high+block, len_args])
                gc.collect()
            del compute_processes
        else:
            for ii in args:
                compute_fun(ii, False, queue)
                gc.collect()
        del queue
        logger.info('Finished Heat')

    gc.collect()

    if elasticity:
        logger.info('Starting Elasticity')
        queue = multiprocessing.Queue()
        if block > 1:
            compute_processes = []
            for ii in args:
                compute_processes.append(multiprocessing.Process(target=compute_fun, args=(ii, True, queue)))
            block_low = 0; block_high = np.min([block, len_args])
            while(block_low < block_high):
                for kk in range(block_low, block_high):
                    compute_processes[kk].start()
                for kk in range(block_low, block_high):
                    queue.get()
                for kk in range(block_low, block_high):
                    compute_processes[kk].join()
                    compute_processes[kk] = None
                block_low = block_high
                block_high = np.min([block_high+block, len_args])
                gc.collect()
            del compute_processes
        else:
            for ii in args:
                compute_fun(ii, True, queue)
                gc.collect()
        del queue
        logger.info('Finished Elasticity')

    gc.collect()
    end = sa_utils.get_time()
    logger.info('Finished patch script ['+prefix_level+', '+str(patch_level)+'] in ['+str(end-start)+']')
