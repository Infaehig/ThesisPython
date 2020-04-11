from dolfin import *

import logging
import sa_utils
from sa_utils import comm, rank, size

import sa_hdf5
import create_patches
import spe

import sympy as smp
import numpy as np
import scipy as sp
import scipy.linalg as la
import numpy.random as rnd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import time
import operator
import sys, os
import io
import gc
import multiprocessing

sys.setrecursionlimit(5000)
PETScOptions.set('mat_mumps_icntl_14', 100)

parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

from sa_utils import myeps
myeps2 = myeps*myeps

ortho_eps = myeps 
ortho_eps2 = ortho_eps*ortho_eps

efendiev_eps = myeps

solid_harmonics = True

direct_params = ('pastix', )
krylov_params = ('cg', 'hypre_amg')

log_solves = False

plotcolors = dict()
plotcolors['efendiev'] = 'g'
plotcolors['efendiev_extended'] = 'g'
plotcolors['efendiev_extended_noweight'] = 'g'
plotcolors['laplace_neumann_extended'] = 'm'
plotcolors['liptonE'] = 'y'
plotcolors['lipton1'] = 'y'
plotcolors['lipton0'] = 'y'
plotcolors['subspaceE'] = 'c'
plotcolors['subspace1'] = 'c'
plotcolors['subspace0'] = 'c'
plotcolors['harmonic_polys'] = 'r'
plotcolors['harmonic_polys_neumann_extended'] = 'r'

plotmarkers = dict()
plotmarkers['efendiev'] = 'go:'
plotmarkers['efendiev_extended'] = 'g+:'
plotmarkers['efendiev_extended_noweight'] = 'gx:'
plotmarkers['liptonE'] = 'yo:'
plotmarkers['lipton1'] = 'yx:'
plotmarkers['lipton0'] = 'ys:'
plotmarkers['subspaceE'] = 'co:'
plotmarkers['subspace1'] = 'cx:'
plotmarkers['subspace0'] = 'cs:'
plotmarkers['harmonic_polys'] = 'ro:'
plotmarkers['harmonic_polys_neumann_extended'] = 'r+:'

names = dict()
names['efendiev'] = r'$\mathcal{V}^{e,-}_l$'
names['efendiev_extended'] = r'$\mathcal{V}^{e}_l$'
names['efendiev_extended_noweight'] = r'$\mathcal{V}^{e, +, 0}_l$'
names['liptonE'] = r'$\mathcal{V}^{E}_l$'
names['lipton1'] = r'$\mathcal{V}^{1}_l$'
names['lipton0'] = r'$\mathcal{V}^{0}_l$'
names['subspaceE'] = r'$\mathcal{V}^{E, s}_l$'
names['subspace1'] = r'$\mathcal{V}^{1, s}_l$'
names['subspace0'] = r'$\mathcal{V}^{0, s}_l$'
names['harmonic_polys'] = r'$\mathcal{V}^{h,-}_l$'
names['harmonic_polys_neumann_extended'] = r'$\mathcal{V}^{h}_l$'
max_degree = 4

legend_rows = 2

def linalg_solve_krylov(AA, xx, bb):
    solver = PETScKrylovSolver(*krylov_params)
    solver.parameters['relative_tolerance'] = 1e-13
    solver.parameters['absolute_tolerance'] = 0.
    solver.parameters['maximum_iterations'] = 1000
    solver.solve(AA, xx, bb)

def linalg_solve_direct(AA, xx, bb):
    solve(AA, xx, bb, *direct_params)

class MySympyHarmonicFactory:
    def __init__(self, vector = False, low = np.array([-1, -1]), high = np.array([1, 1]), numdofs = 1, logger = None, noz = False):
        if logger is not None:
            logger.info('SymPy Harmonic constructor, low={:s}, high={:s}, numdofs=[{:d}]'.format(str(low), str(high), numdofs))
        basedim = len(low)
        old = sa_utils.get_time()
        center = (low+high)*.5
        lengths = (high-low)*.5
        radius = np.sqrt(lengths.dot(lengths))
#       x_expr = smp.symbols('((x[0]-('+str(center[0])+'))/'+str(radius)+')', real = True)
        x_expr = smp.symbols('dx', real = True)
        self.x_expr = x_expr
#       y_expr = smp.symbols('((x[1]-('+str(center[1])+'))/'+str(radius)+')', real = True)
        y_expr = smp.symbols('dy', real = True)
        self.y_expr = y_expr
        if basedim == 2:
            z_expr = x_expr+smp.I*y_expr
            last_power = z_expr
            p_exprs = [z_expr**0, x_expr, y_expr]
            p_offsets = [0, 1, 3]
            new = sa_utils.get_time()
            p_times = [new-old, new-old]
            self.degree = 1
            while(p_offsets[-1] < numdofs):
                old = sa_utils.get_time()
                self.degree += 1
                last_power *= z_expr
                newre = smp.re(last_power); newim = smp.im(last_power)
                p_exprs += [newre, newim]
                new = sa_utils.get_time()
                p_times.append(p_times[-1]+new-old)
                p_offsets.append(len(p_exprs))
            p_exprs = [smp.ccode(smp.horner(smp.horner(pp, wrt = x_expr), wrt = y_expr)) for pp in p_exprs]
        elif basedim == 3:
            z_expr = smp.symbols('dz', real = True)
#           z_expr = smp.symbols('((x[2]-('+str(center[2])+'))/'+str(radius)+')', real = True)
            theta, phi = smp.symbols('theta phi', real = True)
            rr = smp.sqrt(x_expr*x_expr+y_expr*y_expr+z_expr*z_expr)
            theta_ext = smp.acos(z_expr/rr)
            phi_ext = smp.Piecewise((0, x_expr*x_expr+y_expr*y_expr < myeps2), (smp.atan2(y_expr, x_expr), y_expr >= 0), (2*smp.pi+smp.atan2(y_expr, x_expr), True))
            self.z_expr = z_expr
            p_offsets = [0, 1]
            p_exprs = [smp.ccode(smp.nsimplify(smp.simplify(smp.Ynm(0, 0, theta, phi).expand(func = True, trig = True)).subs(smp.pi, pi), rational = True))]
            new = sa_utils.get_time()
            p_times = [new-old]
            self.degree = 0
            pi_exp = smp.symbols('mypi', real = True, positive = True)

            while(p_offsets[-1] < numdofs):
                old = sa_utils.get_time()
                self.degree += 1
                ll = self.degree
                for mm in range(-ll, ll+1):
                    if mm < 0:
                        tmp = (-1)**mm*smp.sqrt(2)*smp.im(smp.Ynm(ll, abs(mm), theta, phi).expand(func = True))
                    elif mm == 0:
                        tmp = smp.re(smp.Ynm(ll, mm, theta, phi).expand(func = True))
                    else:
                        tmp = (-1)**mm*smp.sqrt(2)*smp.re(smp.Ynm(ll, abs(mm), theta, phi).expand(func = True))
                    if solid_harmonics:
                        tmp = tmp*rr**ll
                    tmp = tmp.subs(theta, theta_ext).subs(phi, phi_ext).subs(smp.pi, pi)
                    p_exprs.append(smp.ccode(smp.nsimplify(smp.simplify(tmp), rational = True)))
                    del tmp
                new = sa_utils.get_time()
                p_times.append(p_times[-1]+new-old)
                p_offsets.append(len(p_exprs))
        else:
            raise NameError('Sympy Harmonic Constructor for basedim = ['+str(basedim)+'] not implemented')
        assert(self.degree == len(p_times)-1)
        self.scalar_expressions = []
        count = 0
        self.products = []
        self.degree_offsets = [0]
        times = [0]
        if vector:
            if noz:
                sub_range = list(range(2))
                add = 2
            else:
                sub_range = list(range(basedim))
                add = basedim
            for deg in range(self.degree+1):
                old = sa_utils.get_time()
                for jj in range(p_offsets[deg], p_offsets[deg+1]):
                    cpp_string = '[](double dx, double dy {:s}) {{ return {:s}; }} ((x[0]-xx)/rr, (x[1]-yy)/rr {})'.format('' if basedim == 2 else ', double dz', p_exprs[jj], '' if basedim == 2 else ', (x[2]-zz)/rr')
                    for kk in sub_range:
#                        cpp_string = """
##include <pybind11/pybind11.h>
##include <pybind11/eigen.h>
#namespace py = pybind11;
#
##include <dolfin/function/Expression.h>
##include <dolfin/function/Constant.h>
#
#class MyCppExpression : public dolfin::Expression {{
#    public:
#        size_t ii;
#
#        MyCppExpression(): dolfin::Expression({:d}), xx(0), yy(0), {} rr(1), ii(0) {{}}
#
#        double xx, yy, {} rr;
#
#        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> arg) const {{
#            double dx = (arg[0]-xx)/rr, dy = (arg[1]-yy)/rr{};
#            {}
#        }}
#}};
#
#PYBIND11_MODULE(SIGNATURE, m) {{
#    py::class_<MyCppExpression, std::shared_ptr<MyCppExpression>, dolfin::Expression>
#    (m, "MyCppExpression")
#    .def(py::init<>());
#}}
#""".format(basedim, 
#           *(' zz(0), ', ' zz, ', ', dz = (arg[2]-zz)/rr') if basedim == 3 else ('', '', ''), 
#           ' '.join(['values[{:d}] = 0;'.format(ll) for ll in range(kk)]+['values[{:d}] = {};'.format(kk, p_exprs[jj])]+['values[{:d}] = 0;'.format(ll) for ll in range(kk+1, basedim)]))
##                       print(cpp_string)
#                        exp = CompiledExpression(compile_cpp_code(cpp_string).MyCppExpression(), degree = 1)
                        if basedim == 2:
                            exp = Expression(['0' for ll in range(kk)]+[cpp_string]+['0' for ll in range(kk+1, basedim)], xx = center[0], yy = center[1], rr = radius, degree = 1)
                        elif basedim == 3:
                            exp = Expression(['0' for ll in range(kk)]+[cpp_string]+['0' for ll in range(kk+1, basedim)], xx = center[0], yy = center[1], zz = center[2], rr = radius, degree = 1)

                        exp.xx = center[0]
                        exp.yy = center[1]
                        if basedim == 3:
                            exp.zz = center[2]
                        exp.rr = radius
                        exp.ii = kk
                        self.products.append(exp)
                        del exp
                    count += add
                self.degree_offsets.append(count)
                new = sa_utils.get_time()
                times.append(times[-1]+new-old)
        else:
            for deg in range(self.degree+1):
                old = sa_utils.get_time()
                for jj in range(p_offsets[deg], p_offsets[deg+1]):
                    cpp_string = '[](double dx, double dy {:s}) {{ return {:s}; }} ((x[0]-xx)/rr, (x[1]-yy)/rr {})'.format('' if basedim == 2 else ', double dz', p_exprs[jj], '' if basedim == 2 else ', (x[2]-zz)/rr')
                    if basedim == 2:
                        exp = Expression(cpp_string, xx = center[0], yy = center[1], rr = radius, degree = 1)
                    elif basedim == 3:
                        exp = Expression(cpp_string, xx = center[0], yy = center[1], zz = center[2], rr = radius, degree = 1)
#                    cpp_string = """
##include <pybind11/pybind11.h>
##include <pybind11/eigen.h>
#namespace py = pybind11;
#
##include <dolfin/function/Expression.h>
##include <dolfin/function/Constant.h>
#
#class MyCppExpression : public dolfin::Expression {{
#    public:
#        double xx, yy, {} rr;
#
#        MyCppExpression(): dolfin::Expression(), xx(0), yy(0), {} rr(1) {{}}
#
#        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> arg) const {{
#            double dx = (arg[0]-xx)/rr, dy = (arg[1]-yy)/rr{};
#            values[0] = {};
#        }}
#}};
#
#PYBIND11_MODULE(SIGNATURE, m) {{
#    py::class_<MyCppExpression, std::shared_ptr<MyCppExpression>, dolfin::Expression>
#    (m, "MyCppExpression")
#    .def(py::init<>());
#}}
#""".format(*(' zz, ', ' zz(0), ', ', dz = (arg[2]-zz)/rr') if basedim == 3 else ('', '', ''), p_exprs[jj])
##                   print(cpp_string)
#                    exp = CompiledExpression(compile_cpp_code(cpp_string).MyCppExpression(), degree = 1)
#                   exp.xx = center[0]
#                   exp.yy = center[1]
#                   if basedim == 3:
#                       exp.zz = center[2]
#                   exp.rr = radius
                    self.products.append(exp)
                    del exp
                    count += 1
                self.degree_offsets.append(count)
                new = sa_utils.get_time()
                times.append(times[-1]+new-old)

        self.times = np.array(times[1:])+np.array(p_times)
        self.num_products = len(self.products)
        del p_times
        del p_exprs
        gc.collect()

def localization_test(extended_mesh, extended_domains, extended_kappa, extended_facets, 
                      dofmax = 55, gamma = 1.5, elasticity = False, orthotropic = False, 
                      write_shapes = False, write_paraview = False, matplotlib = True, 
                      prefix = 'test', outdir = 'heat', minimal = 2, ortho = True, 
                      logger = None, spe = False, compute_supinfs = True, orthogonalize_hats = True, 
                      interior_orth_matrix = False,
                      patch_test = False, strain_stress = False, 
                      max_procs = None, batch_procs = None, max_keys = None, beta = 2.0, ldomain = False, 
                      batch_logging = False, krylov = False, krylov_harmonic = False, krylov_neumann = False, debug=False, compute_particular_solutions = True,
                      kappa_is_EE = True, kappa_is_poisson = False, poisson = 0.3, EE = 1e11,
                      bc_cases = None, hole_only_neumann = True, only_lr_dirichlet = False,
                      lagrange_neumann_harmonic = True, normalize_null_orthogonalize = False,
                      nonzero_matrices = True, noz = False, ortho_axis = [0, 0, 1],
                      ortho_params = [[1.5e7, 7.5e5, 7.5e5], [0.3, 1e6, 1e6/(2*(1+0.44))], [0.3, 0.44, 1e6]]):
    if orthotropic:
        elasticity = True
    if elasticity and not orthotropic:
        assert(bool(kappa_is_EE) ^ bool(kappa_is_poisson))
    coeffs = np.unique(extended_kappa.vector().get_local())
    cmin = np.min(coeffs)
    cmax = np.max(coeffs)

    linalg_solve = linalg_solve_direct
    if krylov_harmonic and krylov_neumann:
        krylov = True
    if krylov:
        krylov_harmonic = True
        krylov_neumann = True
        linalg_solve = linalg_solve_krylov
   #krylov_neumann = False
    logger.info('''
Computing enrichments and supinfs START
    dofmax                          {:d}
    contrast                        {:.2e}-{:.2e}={:s}
    beta                            {:.2e}
    gamma                           {:.2e}
    elasticity                      {:s}
    orthotropic                     {:s}
    ldomain                         {:s}
    compute supinfs                 {:s}
    matplotlib                      {:s}
    compute particular solutions    {:s}
    krylov                          {:s}
    krylov_harmonic                 {:s}
    krylov_neumann                  {:s}
    debug                           {:s}
'''.format(dofmax, cmin, cmax, str(cmax/cmin) if cmin > 0 else 'inf', beta, gamma, str(elasticity), str(orthotropic), str(ldomain), str(compute_supinfs), str(matplotlib), str(compute_particular_solutions),
           str(krylov), str(krylov_harmonic), str(krylov_neumann), str(debug)))
    del coeffs
    basedim = extended_mesh.geometry().dim()
    if basedim < 3:
        noz = False
    logger.info('['+str(basedim)+'] dimension mesh')
    if batch_logging:
        batch_logger = logger
    else:
        batch_logger = None
    if log_solves:
        solve_log = logger.info
    else:
        solve_log = lambda xx: 1

    cpu_count = multiprocessing.cpu_count() if max_procs is None else np.min([multiprocessing.cpu_count(), max_procs])

    sa_utils.makedirs_norace(outdir)

    extended_bmesh = BoundaryMesh(extended_mesh, 'exterior')
    extended_bmap = extended_bmesh.entity_map(basedim-1)

    fine_mesh = SubMesh(extended_mesh, extended_domains, 1)
    logger.info('submesh extracted')
    fine_bmesh = BoundaryMesh(fine_mesh, 'exterior')
    fine_bmap = fine_bmesh.entity_map(basedim-1)
   
    logger.info('moving facet markers to submesh')
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
    logger.info('submesh with [{:d}/{:d} = {:.2f}%] cells'.format(fine_mesh.num_cells(), extended_mesh.num_cells(), 100*fine_mesh.num_cells()/extended_mesh.num_cells()))

    if debug:
        sa_utils.makedirs_norace('{:s}/debug'.format(outdir))
        File('{:s}/debug/ext_mesh.pvd'.format(outdir)) << extended_mesh
        File('{:s}/debug/fine_mesh.pvd'.format(outdir)) << fine_mesh
        File('{:s}/debug/ext_facets.pvd'.format(outdir)) << extended_facets
        File('{:s}/debug/fine_facets.pvd'.format(outdir)) << fine_facets

    nodes = fine_mesh.coordinates()
    low = np.min(nodes, axis = 0)
    high = np.max(nodes, axis = 0)
    del nodes
    lengths = high-low
    center = (low+high)*.5
    radius = np.sqrt(lengths.dot(lengths))*.5

    extended_nodes = extended_mesh.coordinates()
    extended_low = np.min(extended_nodes, axis = 0)
    extended_high = np.max(extended_nodes, axis = 0)
    del extended_nodes
    extended_lengths = extended_high-extended_low
    extended_center = (extended_high-extended_low)*.5
    extended_radius = np.sqrt(extended_lengths.dot(extended_lengths))*.5

    if minimal <= 1:
        keys = ['liptonE', 'subspaceE', 'lipton1', 'subspace1', 'lipton0', 'subspace0', 
                'efendiev', 'efendiev_extended', 'efendiev_extended_noweight', 
                'harmonic_polys_neumann_extended']
    elif minimal <= 2:
        keys = ['liptonE', 'subspaceE', 'efendiev_extended', 'harmonic_polys_neumann_extended']
    elif minimal <= 3:
        keys = ['liptonE', 'harmonic_polys_neumann_extended']
    elif minimal <= 4:
        keys = ['liptonE', 'efendiev_extended']
    else:
        keys = ['liptonE']

    numkeys = len(keys)
    efendiev = False
    if 'efendiev' in keys:
        efendiev = True
    efendiev_extended = False
    if 'efendiev_extended' in keys:
        efendiev_extended = True
    efendiev_extended_noweight = False
    if 'efendiev_extended_noweight' in keys:
        efendiev_extended_noweight = True
    liptonE = False
    if 'liptonE' in keys:
        liptonE = True
    lipton0 = False
    if 'lipton0' in keys:
        lipton0 = True
    lipton1 = False
    if 'lipton1' in keys:
        lipton1 = True
    subspaceE = False
    if 'subspaceE' in keys:
        subspaceE = True
    subspace1 = False
    if 'subspace1' in keys:
        subspace1 = True
    subspace0 = False
    if 'subspace0' in keys:
        subspace0 = True
    harmonic_polys_neumann_extended = False
    if 'harmonic_polys_neumann_extended' in keys:
        harmonic_polys_neumann_extended = True

    polys_needed = False
    if harmonic_polys_neumann_extended or subspaceE or subspace1 or subspace0:
        polys_needed = True

    logger.info('Setup')
    # Global Test problem
    dofs = dict()

    if elasticity:
        fine_VV = VectorFunctionSpace(fine_mesh, 'CG', 1, basedim)
    else:
        fine_VV = FunctionSpace(fine_mesh, 'CG', 1)
    fine_dim = fine_VV.dim()
    fine_hh = fine_mesh.hmax()
    fine_normal = FacetNormal(fine_mesh)

    fine_ds = Measure('ds', domain = fine_mesh, subdomain_data = fine_facets)
    fine_dx = Measure('dx', domain = fine_mesh)

    fine_nullspace = sa_utils.build_nullspace(fine_VV, elasticity = elasticity)

    # On extended mesh
    extended_V0 = FunctionSpace(extended_mesh, 'DG', 0)
    if elasticity:
        extended_VV = VectorFunctionSpace(extended_mesh, 'CG', 1, basedim)
    else:
        extended_VV = FunctionSpace(extended_mesh, 'CG', 1)
    extended_dim = extended_VV.dim()
    extended_normal = FacetNormal(extended_mesh)
    
    extended_ds = Measure('ds', domain = extended_mesh, subdomain_data = extended_facets)
    extended_dx = Measure('dx', domain = extended_mesh, subdomain_data = extended_domains)

    def fine_interpolate(uu):
        vv = interpolate(uu, fine_VV)
        vv.rename('u', 'label')
        return vv

    def extended_interpolate(uu):
        vv = interpolate(uu, extended_VV)
        vv.rename('u', 'label')
        return vv
 
    logger.info('Forms')
    
    if elasticity:
        epsilon = lambda uu: sym(grad(uu))
        
        if orthotropic:
            ortho_cos = cos(extended_kappa)
            ortho_sin = sin(extended_kappa)
            RR = as_matrix([
                [ortho_cos + ortho_axis[0]*ortho_axis[0]*(1 - ortho_cos), ortho_axis[0]*ortho_axis[1]*(1 - ortho_cos) - ortho_axis[2]*ortho_sin, ortho_axis[0]*ortho_axis[2]*(1 - ortho_cos) + ortho_axis[1]*ortho_sin],
                [ortho_axis[0]*ortho_axis[1]*(1 - ortho_cos) + ortho_axis[2]*ortho_sin, ortho_cos + ortho_axis[1]*ortho_axis[1]*(1 - ortho_cos), ortho_axis[1]*ortho_axis[2]*(1 - ortho_cos) - ortho_axis[0]*ortho_sin],
                [ortho_axis[0]*ortho_axis[2]*(1 - ortho_cos) - ortho_axis[1]*ortho_sin, ortho_axis[1]*ortho_axis[2]*(1 - ortho_cos) + ortho_axis[0]*ortho_sin, ortho_cos + ortho_axis[2]*ortho_axis[2]*(1 - ortho_cos)]
            ])

            nu_xy = ortho_params[1][0]
            nu_xz = ortho_params[2][0]
            nu_yz = ortho_params[2][1]
            EE = [ortho_params[0][0], ortho_params[1][1], ortho_params[2][2]]
            nu_yx = nu_xy*EE[1]/EE[0]
            nu_zx = nu_xz*EE[2]/EE[0]
            nu_zy = nu_yz*EE[2]/EE[1]
            kk = 1 - nu_xy*nu_yx - nu_yz*nu_zy - nu_xz*nu_zx - 2*nu_xy*nu_yz*nu_zx
            DD_00 = EE[0]*(1 - nu_yz*nu_zy)/kk; DD_01 = EE[0]*(nu_yz*nu_zx + nu_yx)/kk; DD_02 = EE[0]*(nu_zy*nu_yx + nu_zx)/kk
            DD_10 = EE[1]*(nu_xz*nu_zy + nu_xy)/kk; DD_11 = EE[1]*(1 - nu_xz*nu_zx)/kk; DD_12 = EE[1]*(nu_zx*nu_xy + nu_zy)/kk
            DD_20 = EE[2]*(nu_xy*nu_yz + nu_xz)/kk; DD_21 = EE[2]*(nu_yx*nu_xz + nu_yz)/kk; DD_22 = EE[2]*(1 - nu_xy*nu_yx)/kk
            del EE, kk, nu_xy, nu_xz, nu_yz, nu_yx, nu_zx, nu_zy

            def sigma(uu):
                eps = RR*epsilon(uu)*RR.T
                ss_00 = DD_00*eps[0,0]+DD_01*eps[1,1]+DD_02*eps[2,2]
                ss_11 = DD_10*eps[0,0]+DD_11*eps[1,1]+DD_12*eps[2,2]
                ss_22 = DD_20*eps[0,0]+DD_21*eps[1,1]+DD_22*eps[2,2]
                ss_01 = 2*ortho_params[0][1]*eps[0,1]
                ss_02 = 2*ortho_params[0][2]*eps[0,2]
                ss_12 = 2*ortho_params[1][2]*eps[1,2]
                return RR.T*as_matrix([[ss_00, ss_01, ss_02], [ss_01, ss_11, ss_12], [ss_02, ss_12, ss_22]])*RR
        else:
            if kappa_is_EE:
                EE = extended_kappa
            elif kappa_is_poisson:
                poisson = extended_kappa
            ll = (poisson*EE)/((1.+poisson)*(1.-2.*poisson))
            mu = EE/(2.*(1.+poisson))

            poisson = 0.3
            ll = (poisson*extended_kappa)/((1.+poisson)*(1.-2.*poisson))
            mu = extended_kappa/(2.*(1.+poisson))
            sigma = lambda uu: 2.*mu*epsilon(uu)+ll*tr(epsilon(uu))*Identity(basedim)

        zero = Constant([0]*basedim)
        ones = Constant([1]*basedim)

        write_functions = lambda name, uus: sa_hdf5.write_dolfin_vector_cg1(name, uus, True)

        VT = TensorFunctionSpace(fine_mesh, 'DG', 0)
    else:
        epsilon = lambda uu: grad(uu)
        sigma = lambda uu: extended_kappa*epsilon(uu)
        zero = Constant(0)
        ones = Constant(1)
        
        write_functions = lambda name, uus: sa_hdf5.write_dolfin_scalar_cg1(name, uus, True)
        
        VT = VectorFunctionSpace(fine_mesh, 'DG', 0, basedim)

    extended_nullspace = sa_utils.build_nullspace(extended_VV, elasticity = elasticity)

    extended_trial = TrialFunction(extended_VV)
    extended_test = TestFunction(extended_VV)
   
    extended_kk = inner(sigma(extended_trial), epsilon(extended_test))*extended_dx
    extended_l2 = inner(extended_trial, extended_test)*dx

    interior_kk = inner(sigma(extended_trial), epsilon(extended_test))*extended_dx(1)
    interior_l2 = inner(extended_trial, extended_test)*extended_dx(1)
    interior_h1 = inner(grad(extended_trial), grad(extended_test))*extended_dx(1)

    extended_zero_form = inner(zero, extended_test)*extended_dx
    extended_zero = assemble(extended_zero_form)
    extended_ones = interpolate(ones, extended_VV)
  
    old = sa_utils.get_time()
    extended_KK_nonzero = PETScMatrix()
    assemble(extended_kk, tensor = extended_KK_nonzero)
    extended_KK_nonzero.set_nullspace(extended_nullspace)
    new = sa_utils.get_time()
    time_extended_KK_nonzero = new-old
    old = sa_utils.get_time()
    interior_KK_nonzero = PETScMatrix()
    assemble(interior_kk, tensor = interior_KK_nonzero)
    new = sa_utils.get_time()
    time_interior_KK_nonzero = new-old
    old = sa_utils.get_time()
    interior_MM_nonzero = PETScMatrix()
    assemble(interior_l2, tensor = interior_MM_nonzero)
    new = sa_utils.get_time()
    time_interior_MM_nonzero = new-old
    old = sa_utils.get_time()
    interior_H1_nonzero = PETScMatrix()
    assemble(interior_h1, tensor = interior_H1_nonzero)
    new = sa_utils.get_time()
    time_interior_H1_nonzero = new-old

    fine_trial = TrialFunction(fine_VV)
    fine_test = TestFunction(fine_VV)

    if strain_stress:
        fine_vt_trial = TrialFunction(VT)
        fine_vt_test = TestFunction(VT)
        fine_vt_ones = Function(VT)
        fine_vt_ones.vector()[:] = 1
        fine_vt_mass = (assemble(inner(fine_vt_trial, fine_vt_test)*dx)*fine_vt_ones.vector()).get_local()
        fine_vt_epsilon = assemble(inner(epsilon(fine_trial), fine_vt_test)*fine_dx)
        fine_vt_sigma = assemble(inner(sigma(fine_trial), fine_vt_test)*fine_dx)

        def fine_strain_stress(uu):
            strain = Function(VT, name = 'epsilon')
            strain.vector().set_local((fine_vt_epsilon*uu.vector()).get_local()/fine_vt_mass)
            stress = Function(VT, name = 'sigma')
            stress.vector().set_local((fine_vt_sigma*uu.vector()).get_local()/fine_vt_mass)
            return strain, stress

        def write_strain_stress(uus, basename):
            if isinstance(uus, list):
                strainfile = XDMFFile(fine_mesh.mpi_comm(), basename+'_strain.xdmf')
                strainfile.parameters['rewrite_function_mesh'] = False
                stressfile = XDMFFile(fine_mesh.mpi_comm(), basename+'_stress.xdmf')
                stressfile.parameters['rewrite_function_mesh'] = False

                plotnumber = np.min([20, len(uus)])
                for uu in uus[:plotnumber]:
                    strain, stress = fine_strain_stress(uu)
                    strainfile.write(strain)
                    stressfile.write(stress)
                del strainfile
                del stressfile
            else:
                strain, stress = fine_strain_stress(uus)
                XDMFFile(fine_mesh.mpi_comm(), basename+'_strain.xdmf').write(strain)
                XDMFFile(fine_mesh.mpi_comm(), basename+'_stress.xdmf').write(stress)
    else:
        def write_strain_stress(uus, basename):
            return
    
    fine_kk = inner(sigma(fine_trial), epsilon(fine_test))*fine_dx
    fine_l2 = inner(fine_trial, fine_test)*fine_dx
    fine_h1 = inner(grad(fine_trial), grad(fine_test))*fine_dx
 
    fine_zero = assemble(inner(zero, fine_test)*dx)
    fine_ones = interpolate(ones, fine_VV)

    old = sa_utils.get_time()
    fine_KK_nonzero = PETScMatrix()
    assemble(fine_kk, tensor = fine_KK_nonzero)
    new = sa_utils.get_time()
    time_fine_KK_nonzero = new-old
    old = sa_utils.get_time()
    fine_MM_nonzero = PETScMatrix()
    assemble(fine_l2, tensor = fine_MM_nonzero)
    new = sa_utils.get_time()
    time_fine_MM_nonzero = new-old
    old = sa_utils.get_time()
    fine_H1_nonzero = PETScMatrix()
    assemble(fine_h1, tensor = fine_H1_nonzero)
    new = sa_utils.get_time()
    time_fine_H1_nonzero = new-old

    extended_facet_plain = np.unique(facets_array).astype(int)
    logger.info('Extended, plain: '+str(extended_facet_plain))
    del facets_array

    mask_domain_boundary = np.zeros(len(extended_facet_plain), dtype = bool)
    mask_patch_boundary = np.zeros(len(extended_facet_plain), dtype = bool)
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
                                Expression(('-x[1]', 'x[0]'), degree = 1)]
        elif basedim == 3:
            null_expressions = [Constant((1, 0, 0)), 
                                Constant((0, 1, 0)), 
                                Constant((0, 0, 1)), 
                                Expression(('0', '-x[2]', 'x[1]'), degree = 1), 
                                Expression(('x[2]', '0', '-x[0]'), degree = 1), 
                                Expression(('-x[1]', 'x[0]', '0'), degree = 1)]
        else:
            raise NameError('Dimension ['+str(basedim)+'] null space not available')
    else:
        null_expressions = [Constant(1)]
    const_expressions = null_expressions
    null_dim = len(null_expressions)
    logger.info('Null space dimension [{:d}]'.format(null_dim))

    logger.info('Null vectors')
    fine_const_vectors = [fine_interpolate(nu).vector() for nu in const_expressions]
    extended_const_vectors = [extended_interpolate(nu).vector() for nu in const_expressions]
    for ii in range(null_dim):
        for jj in range(ii):
            fine_const_vectors[ii].axpy(-fine_const_vectors[ii].inner(fine_MM_nonzero*fine_const_vectors[jj]), fine_const_vectors[jj])
            extended_const_vectors[ii].axpy(-extended_const_vectors[ii].inner(interior_MM_nonzero*extended_const_vectors[jj]), extended_const_vectors[jj])
        fine_const_vectors[ii] /= np.sqrt(fine_const_vectors[ii].inner(fine_MM_nonzero*fine_const_vectors[ii]))
        extended_const_vectors[ii] /= np.sqrt(extended_const_vectors[ii].inner(interior_MM_nonzero*extended_const_vectors[ii]))

    def fine_orthogonalize_null(vec):
        for vv in fine_const_vectors:
            vec.axpy(-vec.inner(fine_MM_nonzero*vv), vv)

    def extended_orthogonalize_null(vec):
        for vv in extended_const_vectors:
            vec.axpy(-vec.inner(interior_MM_nonzero*vv), vv)

    logger.info('Dirichlet preparatory assembly')
    extended_boundary_form = lambda uu, vv: sum([inner(uu, vv)*extended_ds(int(kk)) for kk in extended_facet_patch_boundary])
    extended_neumann_KK_nonzero = PETScMatrix()
    assemble(extended_boundary_form(dot(sigma(extended_trial), extended_normal), extended_test), tensor = extended_neumann_KK_nonzero, keep_diagonal = True)

    logger.info('Pure Neumann preparatory assembly')
    if elasticity:
        fine_WV = VectorElement('CG', fine_mesh.ufl_cell(), 1, basedim)
    else:
        fine_WV = FiniteElement('CG', fine_mesh.ufl_cell(), 1)
    fine_WR = VectorElement('R', fine_mesh.ufl_cell(), 0, null_dim)
    fine_WW = FunctionSpace(fine_mesh, fine_WV*fine_WR)

    fine_uw, fine_cw = TrialFunctions(fine_WW)
    fine_vw, fine_dw = TestFunctions(fine_WW)
    fine_KW = PETScMatrix()
    assemble((inner(sigma(fine_uw), epsilon(fine_vw))-
              sum(fine_cw[ii]*inner(fine_vw, null_expressions[ii]) for ii in range(null_dim))-
              sum(fine_dw[ii]*inner(fine_uw, null_expressions[ii]) for ii in range(null_dim)))*fine_dx, 
              tensor = fine_KW)

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
    VV_to_WW_assigner = FunctionAssigner(extended_WW.sub(0), extended_VV)
    WW_to_VV_assigner = FunctionAssigner(extended_VV, extended_WW.sub(0))
    extended_RW = PETScMatrix()
    assemble(extended_boundary_form(dot(sigma(extended_uw), extended_normal), extended_vw), tensor=extended_RW)


    if compute_particular_solutions:
        def dirichlet_extended_solve(dirichlet, neumann = None, ff = zero):
            solve_log('            dirichlet mixed setup')
            AA = PETScMatrix()
            bb = PETScVector()
            assemble_system(extended_kk,
                            inner(ff, extended_test)*dx+sum(inner(hh, extended_test)*extended_ds(kk) for (hh,kk) in neumann) if neumann else inner(ff, extended_test)*dx,
                            [DirichletBC(extended_VV, gg, extended_facets, kk) for (gg, kk) in dirichlet],
                            A_tensor = AA, b_tensor = bb)
            vv = Function(extended_VV, name = 'u')
            solve_log('            dirichlet mixed solve')
            linalg_solve(AA, vv.vector(), bb)
            del AA, bb
            solve_log('            dirichlet mixed return')
            return vv

        if krylov_neumann:
            def neumann_extended_solve(neumann = None, ff = zero):
                solve_log('            neumann mixed setup')
                bb = assemble(inner(ff, extended_test)*dx)
                if neumann is not None:
                    for (hh, kk) in neumann:
                        bb += assemble(inner(hh, extended_test)*extended_ds(kk))
                extended_nullspace.orthogonalize(bb)
                solve_log('            neumann mixed solve')
                vv = Function(extended_VV, name = 'u')
                linalg_solve_krylov(extended_KK_nonzero, vv.vector(), bb)
                extended_orthogonalize_null(vv.vector())
                del bb
                solve_log('            neumann mixed return')
                return vv
        else:
            def neumann_extended_solve(neumann = None, ff = zero):
                solve_log('            neumann mixed setup')
                bb = assemble(inner(ff, extended_vw)*dx)
                if neumann is not None:
                    for (hh, kk) in neumann:
                        bb += assemble(inner(hh, extended_vw)*extended_ds(kk))
                ww = Function(extended_WW)
                solve_log('            neumann mixed solve')
                linalg_solve_direct(extended_KW, ww.vector(), bb)
                del bb
                vv = Function(extended_VV, name = 'u')
                solve_log('            neumann mixed assign')
                WW_to_VV_assigner.assign(vv, ww.sub(0))
                del ww
                solve_log('            neumann mixed return')
                return vv

#       particular_dir = '{:s}/../'.format(outdir)
        particular_dir = outdir
        sa_utils.makedirs_norace(particular_dir+'/coeffs')
        logger.info('Computing fine scaled patch solutions for inhomogeneities, basedim = ['+str(basedim)+']')
        all_bcs = sa_utils.get_bcs(basedim, elasticity, ldomain=ldomain)
        null_basis = [fine_interpolate(uu) for uu in null_expressions]
        write_strain_stress(null_basis, outdir+'/strain_stress/null_basis')
        write_functions(outdir+'/coeffs/null_basis', null_basis)

        for ii, (dirichlet, neumann, ff) in enumerate(all_bcs):
            logger.info('    [{:d}/{:d}]'.format(ii+1, len(all_bcs)))
            basename = prefix+'_'+str(ii)
            dirichlet_here = []
            dirichlet_zero = []
            for (gg, kk) in dirichlet:
                if kk in extended_facet_domain_boundary:
                    dirichlet_here.append((gg, kk))
                    dirichlet_zero.append((zero, kk))
            neumann_here = []
            for (hh, kk) in neumann:
                if kk in extended_facet_domain_boundary:
                    neumann_here.append((hh, kk))
            if ff is not None:
                if len(dirichlet_zero):
                    vv = fine_interpolate(dirichlet_extended_solve(dirichlet = dirichlet_zero, ff = ff))
                else:
                    vv = fine_interpolate(neumann_extended_solve(ff = ff))
                logger.info('Pair ['+str(ii)+'] rhs addition computed')
                write_strain_stress(vv, particular_dir+'/strain_stress/'+basename+'_f')
                write_functions(particular_dir+'/coeffs/'+basename+'_f', [vv])
                logger.info('Pair ['+str(ii)+'] rhs addition written')
            if len(dirichlet_here):
                vv = fine_interpolate(dirichlet_extended_solve(dirichlet = dirichlet_here))
                logger.info('Pair ['+str(ii)+'] dirichlet addition computed')
                write_strain_stress(vv, particular_dir+'/strain_stress/'+basename+'_g')
                write_functions(particular_dir+'/coeffs/'+basename+'_g', [vv])
                logger.info('Pair ['+str(ii)+'] dirichlet addition written')
            if len(neumann_here):
                if len(dirichlet_zero):
                    vv = fine_interpolate(dirichlet_extended_solve(dirichlet = dirichlet_zero, neumann = neumann_here))
                else:
                    vv = fine_interpolate(neumann_extended_solve(neumann = neumann_here))
                logger.info('Pair ['+str(ii)+'] neumann addition computed')
                write_strain_stress(vv, particular_dir+'/strain_stress/'+basename+'_h')
                write_functions(particular_dir+'/coeffs/'+basename+'_h', [vv])
                logger.info('Pair ['+str(ii)+'] neumann addition written')
            del dirichlet_here, dirichlet_zero, neumann_here
 
    logger.info('Extracting free boundary dofs')
    old = sa_utils.get_time()
    domain_boundary_dofs_dict = dict()
    patch_boundary_dofs_dict = dict()
    not_domain_boundary_dofs_dict = dict()
    num_vertices = extended_mesh.num_vertices()
    dofmap = vertex_to_dof_map(extended_VV)
    if elasticity:
        if noz:
            sub_range = list(range(2))
        else:
            sub_range = list(range(basedim))
        dofmap = [[dofmap[vertex*basedim+sub] for vertex in range(num_vertices)] for sub in range(basedim)]
        for kk in extended_facet_domain_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    for sub in sub_range:
                        tmp = dofmap[sub][vertex.index()]
                        domain_boundary_dofs_dict[tmp] = None
        for kk in extended_facet_patch_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    for sub in sub_range:
                        tmp = dofmap[sub][vertex.index()]
                        patch_boundary_dofs_dict[tmp] = None
                        if not tmp in domain_boundary_dofs_dict:
                            not_domain_boundary_dofs_dict[tmp] = None
    else:
        for kk in extended_facet_domain_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    tmp = dofmap[vertex.index()]
                    domain_boundary_dofs_dict[tmp] = None
        for kk in extended_facet_patch_boundary:
            for ff in SubsetIterator(extended_facets, kk):
                for vertex in vertices(ff):
                    tmp = dofmap[vertex.index()]
                    patch_boundary_dofs_dict[tmp] = None
                    if not tmp in domain_boundary_dofs_dict:
                        not_domain_boundary_dofs_dict[tmp] = None
    patch_boundary_dofs = sorted(patch_boundary_dofs_dict.keys())
    not_domain_boundary_dofs = sorted(not_domain_boundary_dofs_dict.keys())
    del domain_boundary_dofs_dict, patch_boundary_dofs_dict, not_domain_boundary_dofs_dict
    logger.info("""Boundary dofs extracted
    patch boundary          {:d}
    not domain boundary     {:d}""".format(len(patch_boundary_dofs), len(not_domain_boundary_dofs)))
    new = sa_utils.get_time()
    time_dofmap = new-old
    
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

    if krylov_neumann:
        def neumann_extended_neumann(uu):
            solve_log('            neumann setup')
            uinterp = extended_interpolate(uu)
            bb = extended_zero+extended_neumann_KK_nonzero*uinterp.vector()
            del uinterp
            extended_nullspace.orthogonalize(bb)
            vv = Function(extended_VV, name = 'u')
            solve_log('            neumann solve')
            linalg_solve_krylov(extended_KK_nonzero, vv.vector(), bb)
            extended_orthogonalize_null(vv.vector())
            del bb
            solve_log('            neumann return')
            return vv
    else:
        def neumann_extended_neumann(uu):
            solve_log('            neumann setup')
            uinterp = Function(extended_WW)
            VV_to_WW_assigner.assign(uinterp.sub(0), interpolate(uu, extended_VV))
            bb = extended_bw+extended_RW*uinterp.vector()
            del uinterp
            ww = Function(extended_WW)
            solve_log('            neumann solve')
            linalg_solve_direct(extended_KW, ww.vector(), bb)
            del bb
            solve_log('            neumann assign')
            vv = Function(extended_VV, name = 'u')
            WW_to_VV_assigner.assign(vv, ww.sub(0))
            del ww
            solve_log('            neumann return')
            return vv

    if krylov_harmonic:
        def neumann_extended_harmonic(uu):
            solve_log('            neumann dirichlet setup')
           #AA = extended_KK_nonzero.copy()
           #bb = extended_zero.copy()
           #for kk in extended_facet_patch_boundary:
           #    bc = DirichletBC(extended_VV, uu, extended_facets, kk)
           #    bc.apply(AA, bb)
            AA = PETScMatrix()
            bb = PETScVector()
            assemble_system(extended_kk, extended_zero_form,
                            [DirichletBC(extended_VV, uu, extended_facets, kk) for kk in extended_facet_patch_boundary],
                            A_tensor = AA, b_tensor = bb)
            vv = Function(extended_VV, name = 'u')
            solve_log('            neumann dirichlet solve')
            linalg_solve_krylov(AA, vv.vector(), bb)
            extended_orthogonalize_null(vv.vector())
            del AA, bb
            solve_log('            neumann dirichlet return')
            return vv
    else:
        if lagrange_neumann_harmonic:
            def neumann_extended_harmonic(uu):
                solve_log('            neumann dirichlet setup')
                AA = extended_KW.copy()
                bb = extended_bw.copy()
                for kk in extended_facet_patch_boundary:
                    bc = DirichletBC(extended_WW.sub(0), uu, extended_facets, kk)
                    bc.apply(AA, bb)
                ww = Function(extended_WW)
                solve_log('            neumann dirichlet solve')
                linalg_solve_direct(AA, ww.vector(), bb)
                del AA, bb
                vv = Function(extended_VV, name = 'u')
                solve_log('            neumann dirichlet assign')
                WW_to_VV_assigner.assign(vv, ww.sub(0))
                del ww
                solve_log('            neumann dirichlet return')
                return vv
        else:
            def neumann_extended_harmonic(uu):
                solve_log('            neumann dirichlet setup')
                AA = extended_KK_nonzero.copy()
                bb = extended_zero.copy()
                for kk in extended_facet_patch_boundary:
                    bc = DirichletBC(extended_VV, uu, extended_facets, kk)
                    bc.apply(AA, bb)
                vv = Function(extended_VV, name = 'u')
                solve_log('            neumann dirichlet solve')
                linalg_solve_direct(AA, vv.vector(), bb)
                del AA, bb
                solve_log('            neumann dirichlet return')
                return vv

    numdofs = int(gamma*dofmax/(basedim if elasticity else 1))
    logger.info('Harmonic polynomials nobc elasticity=[{:s}], gamma=[{:.1e}], dofmax=[{:d}], numdofs=[{:d}]'.format(str(elasticity), gamma, dofmax, numdofs))
    harmonic_factory_nobc = MySympyHarmonicFactory(vector = elasticity, low = extended_low, high = extended_high, numdofs = numdofs, logger = logger, noz = noz)
    del numdofs
    harmonic_times_tmp_nobc = harmonic_factory_nobc.times

    harmonic_degree_nobc = harmonic_factory_nobc.degree
    harmonic_offsets_nobc = harmonic_factory_nobc.degree_offsets
    harmonic_offset_nobc = harmonic_offsets_nobc[1]
    harmonic_times_nobc = np.array(harmonic_factory_nobc.times[1:])
    harmonic_dofs_nobc = np.array(harmonic_factory_nobc.degree_offsets[2:])-harmonic_offset_nobc
    assert(len(harmonic_dofs_nobc) == len(harmonic_times_nobc))
    opt_dofs = harmonic_dofs_nobc

    logger.info('Extended harmonic polynomials')
    extended_harmonic_polys_neumann_root = [extended_interpolate(uu) for uu in harmonic_factory_nobc.products[:harmonic_offset_nobc]]
    logger.info('initialized until degree ['+str(harmonic_factory_nobc.degree)+']')
     
    logger.info('Looping bc cases')
    domain_boundary_num = len(extended_facet_domain_boundary)
    num_cases = 2**domain_boundary_num
    for bc_ii in range(num_cases):
        gc.collect()
        mask_dirichlet = np.zeros(domain_boundary_num, dtype = bool)
        mask_neumann = np.zeros(domain_boundary_num, dtype = bool)
        binary = format(bc_ii, '0{:d}b'.format(domain_boundary_num))
        for jj in range(domain_boundary_num):
            if int(binary[jj]):
                mask_neumann[jj] = True
            else:
                mask_dirichlet[jj] = True
        extended_facet_dirichlet = extended_facet_domain_boundary[mask_dirichlet]
        extended_facet_neumann = extended_facet_domain_boundary[mask_neumann]
        bc_suffix = bytearray('__________'.encode())
        DD = 'D'.encode()[0]
        NN = 'N'.encode()[0]
        for ii in range(1, 11):
            if ii in extended_facet_dirichlet:
                bc_suffix[ii-1] = DD
            elif ii in extended_facet_neumann:
                bc_suffix[ii-1] = NN
        bc_suffix = bc_suffix.decode()
        del mask_dirichlet, mask_neumann, DD, NN

        skip = False
        if bc_cases is not None and not bc_suffix in bc_cases:
            skip = True
        if hole_only_neumann and bc_suffix[-1] == 'D':
            skip = True
        if only_lr_dirichlet:
            for ii in range(3, 11):
                if ii in extended_facet_dirichlet:
                    skip = True
                    break

        if skip:
            logger.info('bc case [{:d}/{:d}]: {:s} SKIP'.format(bc_ii+1, num_cases, bc_suffix))
            continue
        else:
            logger.info('bc case [{:d}/{:d}]: {:s} START'.format(bc_ii+1, num_cases, bc_suffix))
            pass

        if len(extended_facet_dirichlet):
            boundary_dofs = not_domain_boundary_dofs
        else:
            boundary_dofs = patch_boundary_dofs

        num_boundary = len(boundary_dofs)

        outdir_bc = '{:s}/{:s}'.format(outdir, bc_suffix)
        sa_utils.makedirs_norace(outdir_bc)

        logger.info('extended mesh matrix assembly')
       
        extended_zero_bcs = [DirichletBC(extended_VV, zero, extended_facets, ii) for ii in extended_facet_dirichlet]
        
        _dummy = inner(ones, extended_test)*extended_dx
        _bb = PETScVector()
     
        old = sa_utils.get_time()
        extended_KK_zero = PETScMatrix()
        assemble_system(extended_kk, _dummy, extended_zero_bcs, A_tensor = extended_KK_zero, b_tensor = _bb)
        if not len(extended_facet_dirichlet):
            extended_KK_zero.set_nullspace(extended_nullspace)
        new = sa_utils.get_time()
        time_extended_KK_zero = new-old

        if nonzero_matrices or not len(extended_facet_dirichlet):
            extended_KK = extended_KK_nonzero
            time_extended_KK= time_extended_KK_zero
            interior_KK = interior_KK_nonzero
            time_interior_KK = time_interior_KK_nonzero
            interior_MM = interior_MM_nonzero
            time_interior_MM = time_interior_MM_nonzero
            interior_H1 = interior_H1_nonzero
            time_interior_H1 = time_interior_H1_nonzero

            fine_KK = fine_KK_nonzero
            time_fine_KK = time_fine_KK_nonzero
            fine_MM = fine_MM_nonzero
            time_fine_MM = time_fine_MM_nonzero
            fine_H1 = fine_H1_nonzero
            time_fine_H1 = time_fine_H1_nonzero

        else:
            extended_KK = extended_KK_zero
            time_extended_KK = time_extended_KK_zero
            old = sa_utils.get_time()
            interior_KK = PETScMatrix()
            assemble_system(interior_kk, _dummy, extended_zero_bcs, A_tensor = interior_KK, b_tensor = _bb)
            new = sa_utils.get_time()
            time_interior_KK = new-old
            old = sa_utils.get_time()
            interior_MM = PETScMatrix()
            assemble_system(interior_l2, _dummy, extended_zero_bcs, A_tensor = interior_MM, b_tensor = _bb)
            new = sa_utils.get_time()
            time_interior_MM = new-old
            old = sa_utils.get_time()
            interior_H1 = PETScMatrix()
            assemble_system(interior_h1, _dummy, extended_zero_bcs, A_tensor = interior_H1, b_tensor = _bb)
            new = sa_utils.get_time()
            time_interior_H1 = new-old

            _bb = PETScVector()
            _fine_dummy = inner(ones, fine_test)*dx
            old = sa_utils.get_time()
            fine_MM = PETScMatrix()
            assemble_system(fine_l2, _fine_dummy, fine_zero_bcs, A_tensor = fine_MM, b_tensor = _bb)
            new = sa_utils.get_time()
            time_fine_MM = new-old
            old = sa_utils.get_time()
            fine_KK = PETScMatrix()
            assemble_system(fine_kk, _fine_dummy, fine_zero_bcs, A_tensor = fine_KK, b_tensor = _bb)
            new = sa_utils.get_time()
            time_fine_KK = new-old
            old = sa_utils.get_time()
            fine_H1 = PETScMatrix()
            assemble_system(fine_h1, _fine_dummy, fine_zero_bcs, A_tensor = fine_H1, b_tensor = _bb)
            new = sa_utils.get_time()
            time_fine_H1 = new-old

            del _fine_dummy

        del _bb, _dummy

        logger.info('fine mesh matrix assembly')
        fine_zero_bcs = [DirichletBC(fine_VV, zero, fine_facets, ii) for ii in extended_facet_dirichlet]

        fine_orthogonalize = lambda uu, uus: orthogonalize(uu, uus, fine_KK)
        if interior_orth_matrix:
            extended_orthogonalize = lambda uu, uus: orthogonalize(uu, uus, interior_KK)
        else:
            extended_orthogonalize = lambda uu, uus: orthogonalize(uu, uus, extended_KK)

        logger.info('Computing coarse bases')
       
        uus = dict()
        coarse_spaces = dict()

        if len(extended_facet_dirichlet):
            extended_neumann_KK = extended_neumann_KK_nonzero.copy()

            if krylov_neumann:
                def extended_neumann(uu):
                    solve_log('            dirichlet neumann setup')
                    uinterp = extended_interpolate(uu)
                   #AA = extended_KK_nonzero.copy()
                    bb = extended_zero+extended_neumann_KK*uinterp.vector()
                    for bc in extended_zero_bcs:
                        bc.apply(bb)
                   #    bc.apply(AA, bb)
                    del uinterp
                    vv = Function(extended_VV, name = 'u')
                    solve_log('            dirichlet neumann solve')
                  # linalg_solve_krylov(AA, vv.vector(), bb)
                    linalg_solve_krylov(extended_KK_zero, vv.vector(), bb)
                   #del AA, bb
                    solve_log('            dirichlet neumann return')
                    return vv
            else:
                def extended_neumann(uu):
                    solve_log('            dirichlet neumann setup')
                    uinterp = extended_interpolate(uu)
                    bb = extended_zero+extended_neumann_KK*uinterp.vector()
                    for bc in extended_zero_bcs:
                        bc.apply(bb)
                    del uinterp
                    vv = Function(extended_VV, name = 'u')
                    solve_log('            dirichlet neumann solve')
                    linalg_solve_direct(extended_KK_zero, vv.vector(), bb)
                    del bb
                    solve_log('            dirichlet neumann return')
                    return vv
            
            if krylov_harmonic:
                def extended_harmonic(uu):
                    solve_log('            dirichlet dirichlet setup')
                   #AA = extended_KK_nonzero.copy()
                   #bb = extended_zero.copy()
                   #for kk in extended_facet_patch_boundary:
                   #    bc = DirichletBC(extended_VV, uu, extended_facets, kk)
                   #    bc.apply(AA, bb)
                   #for bc in extended_zero_bcs:
                   #    bc.apply(AA, bb)
                    AA = PETScMatrix()
                    bb = PETScVector()
                    assemble_system(extended_kk, extended_zero_form,
                                    [DirichletBC(extended_VV, uu, extended_facets, kk) for kk in extended_facet_patch_boundary]+
                                    extended_zero_bcs,
                                    A_tensor = AA, b_tensor = bb)
                    vv = Function(extended_VV, name = 'u')
                    solve_log('            dirichlet dirichlet solve')
                    linalg_solve_krylov(AA, vv.vector(), bb)
                    del AA, bb
                    solve_log('            dirichlet dirichlet return')
                    return vv
            else:
                def extended_harmonic(uu):
                    solve_log('            dirichlet dirichlet setup')
                    AA = extended_KK_nonzero.copy()
                    bb = extended_zero.copy()
                    for kk in extended_facet_patch_boundary:
                        bc = DirichletBC(extended_VV, uu, extended_facets, kk)
                        bc.apply(AA, bb)
                    for bc in extended_zero_bcs:
                        bc.apply(AA, bb)
                    vv = Function(extended_VV, name = 'u')
                    solve_log('            dirichlet dirichlet solve')
                    linalg_solve_direct(AA, vv.vector(), bb)
                    del AA, bb
                    solve_log('            dirichlet dirichlet return')
                    return vv
        else:
            extended_neumann = neumann_extended_neumann
            extended_harmonic = neumann_extended_harmonic

        if patch_test:
            solve_log('patch test')
            solve_log('    expression')
            if elasticity:
                exp = Expression(('1+x[0]*x[0]+2.*x[1]*x[1]', '0'), degree = 2)
            else:
                exp = Expression('1+x[0]*x[0]+2.*x[1]*x[1]', degree = 2)
            queue = multiprocessing.Queue()
            def test(qq, exp):
                solve_log('        interpolate')
                uu = extended_interpolate(exp)
                solve_log('        interpolate write')
                write_functions(outdir_bc+'/debug/uu', [uu])
                solve_log('    expression done, dirichlet')
                bb = extended_harmonic(uu)
                solve_log('        dirichlet write')
                write_functions(outdir_bc+'/debug/uu_dirichlet', [bb])
                solve_log('    dirichlet done, neumann')
                aa = extended_neumann(uu)
                solve_log('        neumann write')
                write_functions(outdir_bc+'/debug/uu_neumann', [aa])
                solve_log('    neumann done')
                qq.put(None)
                del exp, uu, aa, bb
            solve_log('    without process')
            test(queue, exp)
            queue.get()
            solve_log('    with process')
            process = multiprocessing.Process(target = test, args = (queue, exp))
            process.start()
            queue.get()
            del queue
            process.join()
            continue

        times_direct = dict()
        times_eigen = dict()
        times_compute = dict()

        if polys_needed: 
            extended_harmonic_times_neumann = [0]
            logger.info('    computing harmonic extensions')
            extended_harmonic_polys_neumann = list(extended_harmonic_polys_neumann_root)
            sa_utils.batch_fun(harmonic_factory_nobc.products, extended_VV, low = 1, high = harmonic_degree_nobc+1, offsets = harmonic_offsets_nobc, 
                               fun = extended_neumann, polys = extended_harmonic_polys_neumann, times = extended_harmonic_times_neumann, logger = batch_logger, 
                               max_procs = batch_procs)
            if debug:
                write_functions(outdir_bc+'/debug/extended_harmonic_polys', extended_harmonic_polys_neumann)
     
            logger.info('    computed harmonic extensions, Length: ['+str(len(extended_harmonic_polys_neumann))+']')
            extended_harmonic_times_neumann = harmonic_times_nobc+np.array(extended_harmonic_times_neumann[1:])

        if harmonic_polys_neumann_extended:
            interior_laplace_polys_nobc = [fine_interpolate(uu) for uu in harmonic_factory_nobc.products[:harmonic_offset_nobc]]
            laplace_times_interior_nobc = [0]
            logger.info('    computing restrictions')
            sa_utils.batch_fun(extended_harmonic_polys_neumann, fine_VV, low = 1, high = harmonic_degree_nobc+1, offsets = harmonic_offsets_nobc, 
                               fun = fine_interpolate, polys = interior_laplace_polys_nobc, times = laplace_times_interior_nobc, logger = batch_logger, 
                               max_procs = batch_procs)
            logger.info('    computed restrictions')
            dofs['harmonic_polys_neumann_extended'] = harmonic_dofs_nobc.copy()
            uus['harmonic_polys_neumann_extended'] = interior_laplace_polys_nobc[harmonic_offset_nobc:]
            times_direct['harmonic_polys_neumann_extended'] = extended_harmonic_times_neumann+np.array(laplace_times_interior_nobc[1:])
            times_eigen['harmonic_polys_neumann_extended'] = np.zeros(len(dofs['harmonic_polys_neumann_extended']))
            logger.info('laplace nobc neumann extended done')
            del laplace_times_interior_nobc
            del interior_laplace_polys_nobc


        if matplotlib and (efendiev or efendiev_extended or efendiev_extended_noweight):
            efendiev_fig = plt.figure()
            efendiev_ax = efendiev_fig.add_subplot(111)
            efendiev_handles = []
            efendiev_fig_loglog = plt.figure()
            efendiev_ax_loglog = efendiev_fig_loglog.add_subplot(111)
            efendiev_handles_loglog = []

            ymin = 1
            ymax = 0
            xmax = 0

        if efendiev:
            logger.info('Efendiev')
            fine_MM_weighted = PETScMatrix()
            old = sa_utils.get_time()
            if len(extended_facet_dirichlet):
                AA = fine_KK
                assemble(extended_kappa*inner(fine_trial, fine_test)*fine_dx, tensor = fine_MM_weighted)
            else:
                AA = fine_KW
                assemble(extended_kappa*inner(fine_uw, fine_vw)*fine_dx, tensor = fine_MM_weighted)
            new = sa_utils.get_time()
            time_fine_MM_weighted = new-old
            logger.info('    assembled')

            eigensolver = None
            times_eigensolver = [None]
            broken = False
            for deg in range(len(opt_dofs)):
                old = sa_utils.get_time()
                tmp_eigensolver = SLEPcEigenSolver(AA, fine_MM_weighted)
                tmp_eigensolver.parameters['problem_type'] = 'gen_hermitian'
                tmp_eigensolver.parameters['spectrum'] = 'smallest magnitude'
                tmp_eigensolver.parameters['spectral_shift'] = 0.
                tmp_eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
                tmp_eigensolver.parameters['tolerance'] = efendiev_eps
                tmp_eigensolver.solve(opt_dofs[deg]+null_dim)
                if tmp_eigensolver.get_number_converged() <  opt_dofs[deg]+null_dim:
                    del tmp_eigensolver
                    broken = True
                    break
                eigensolver = tmp_eigensolver
                del tmp_eigensolver
                new = sa_utils.get_time()
                times_eigensolver.append(time_fine_MM_weighted+time_fine_KK+new-old)
            if broken:
                if deg > 0:
                    tmp_dofs = opt_dofs[:deg].copy()
                    broken = False
                else:
                    tmp_dofs = None
            else:
                tmp_dofs = opt_dofs.copy()
            del AA
            del fine_MM_weighted
            times_eigensolver = np.array(times_eigensolver[1:])
            logger.info('    eigensolver ran')

            if eigensolver is not None:
                fine_converged = eigensolver.get_number_converged()
            else:
                fine_converged = 0
            if broken or fine_converged < tmp_dofs[-1]:
                del tmp_dofs
                efendiev = False
                logger.critical('EFENDIEV: not enough eigenvectors converged')
                keys.remove('efendiev')
            else:
                logger.info('solved, ['+str(fine_converged)+'/'+str(tmp_dofs[-1])+'] converged')
                efendiev_eigenvalues = []
                interior_efendiev = []
                uus['efendiev'] = interior_efendiev
                times_efendiev = [0]
                dofs['efendiev'] = tmp_dofs
                last = 0
                count = 0
                ok = True
                for deg in range(len(tmp_dofs)):
                    current = tmp_dofs[deg]
                    old = sa_utils.get_time()
                    for ii in range(last, current):
                        while count < fine_converged:
                            rr, cc, rv, rc = eigensolver.get_eigenpair(count)
                            if rr > myeps:
                                break
                            else:
                                if debug:
                                    logger.info('Efendiev [{:d}] skip: {:.2e}'.format(count, rr))
                                count += 1
                        if debug:
                            logger.info('Efendiev [{:d}]: {:.2e}'.format(ii, rr))
                        count += 1
                        efendiev_eigenvalues.append(1/np.sqrt(rr))
                        interior_efendiev.append(Function(fine_VV, name = 'u'))
                        if len(extended_facet_dirichlet):
                            interior_efendiev[-1].vector()[:] = rv
                        else:
                            vv = Function(fine_WW)
                            vv.vector()[:] = rv
                            assign(interior_efendiev[-1], vv.sub(0))
                    new = sa_utils.get_time()
                    times_efendiev.append(times_efendiev[-1]+new-old)
                    if current > last:
                        last = current
                del tmp_dofs
                del count
                del eigensolver
                times_direct['efendiev'] = np.array(times_efendiev[1:])
                times_eigen['efendiev'] = times_eigensolver
                del interior_efendiev
                if debug:
                    write_functions(outdir_bc+'/debug/efendiev', uus['efendiev'])
            del broken

            if matplotlib and efendiev:
                efendiev_eigenvalues = np.array(efendiev_eigenvalues)
                efendiev_len = len(efendiev_eigenvalues)
                ymin = np.min([ymin, np.min(efendiev_eigenvalues)])
                ymax = np.max([ymax, np.max(efendiev_eigenvalues)])
                xmax = np.max([xmax, efendiev_len])
                efendiev_handles.append(*efendiev_ax.semilogy(1+np.arange(efendiev_len), efendiev_eigenvalues, 'go:', mec = 'g', mfc = 'none', label = r'$\omega_l$'))
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog(1+np.arange(efendiev_len), efendiev_eigenvalues, 'go:', mec = 'g', mfc = 'none', label = r'$\omega_l$'))
                xx = np.log(1.+np.arange(efendiev_len))
                yy = np.log(efendiev_eigenvalues)
                AA = np.vstack([xx, np.ones(efendiev_len)]).T
                rate, offset = la.lstsq(AA, yy)[0]
                del AA
                ydiff = np.exp(0.1*np.log(ymax/ymin))
                efendiev_log = (ymin/ydiff)
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog([1, efendiev_len], [efendiev_log/efendiev_len**rate, efendiev_log], 'k:', label = r'$c n^{'+("%.2g"%rate)+r'}$'))
                del efendiev_log, rate, offset, ydiff

            logger.info('Efendiev done')

        if efendiev_extended:
            if debug:
                tmp = []
            logger.info('Efendiev extended')
            extended_MM_weighted = PETScMatrix()
            old = sa_utils.get_time()
            if len(extended_facet_dirichlet):
                AA = extended_KK_zero.copy()
                if noz:
                    noz_bc = DirichletBC(extended_VV.sub(2), 0, extended_facets, 100)
                    noz_bc.apply(AA)
                assemble(extended_kappa*inner(extended_trial, extended_test)*extended_dx, tensor = extended_MM_weighted)
            else:
                AA = extended_KW.copy()
                if noz:
                    noz_bc = DirichletBC(extended_WW.sub(0).sub(2), 0, extended_facets, 100)
                    noz_bc.apply(AA)
                assemble(extended_kappa*inner(extended_uw, extended_vw)*extended_dx, tensor = extended_MM_weighted)
            new = sa_utils.get_time()
            time_extended_MM_weighted = new-old
            logger.info('    assembled')

            eigensolver = None
            times_eigensolver_extended = [None]
            broken = False
            for deg in range(len(opt_dofs)):
                old = sa_utils.get_time()
                tmp_eigensolver = SLEPcEigenSolver(AA, extended_MM_weighted)
                tmp_eigensolver.parameters['problem_type'] = 'gen_hermitian'
                tmp_eigensolver.parameters['spectrum'] = 'smallest magnitude'
                tmp_eigensolver.parameters['spectral_shift'] = 0.
                tmp_eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
                tmp_eigensolver.parameters['tolerance'] = efendiev_eps
                tmp_eigensolver.solve(opt_dofs[deg]+null_dim)
                if tmp_eigensolver.get_number_converged() <  opt_dofs[deg]+null_dim:
                    del tmp_eigensolver
                    broken = True
                    break
                eigensolver = tmp_eigensolver
                del tmp_eigensolver
                new = sa_utils.get_time()
                times_eigensolver_extended.append(time_extended_MM_weighted+time_extended_KK_zero+new-old)
            if broken:
                if deg > 0:
                    tmp_dofs = opt_dofs[:deg].copy()
                    broken = False
                else:
                    tmp_dofs = None
            else:
                tmp_dofs = opt_dofs.copy()
            del AA, extended_MM_weighted
            times_eigensolver_extended = np.array(times_eigensolver_extended[1:])
            logger.info('    eigensolver ran')

            if eigensolver is not None:
                extended_converged = eigensolver.get_number_converged()
            else:
                extended_converged = 0
            if broken or extended_converged < tmp_dofs[-1]:
                efendiev_extended = False
                logger.critical('EFENDIEV: extended, not enough eigenvectors converged')
                keys.remove('efendiev_extended')
            else:
                logger.info('solved, ['+str(extended_converged)+'/'+str(tmp_dofs[-1])+'] converged')
                efendiev_extended_eigenvalues = []
                interior_efendiev = []
                uus['efendiev_extended'] = interior_efendiev
                times_efendiev_extended = [0]
                dofs['efendiev_extended'] = tmp_dofs
                last = 0
                ok = True
                shift = 0 if len(extended_facet_dirichlet) else null_dim
                count = 0
                for deg in range(len(tmp_dofs)):
                    current = tmp_dofs[deg]
                    old = sa_utils.get_time()
                    for ii in range(last, current):
                        while count < extended_converged:
                            rr, cc, rv, rc = eigensolver.get_eigenpair(count)
                            if rr > myeps:
                                break
                            else:
                                if debug:
                                    logger.info('Efendiev extended [{:d}] skip: {:.2e}'.format(count, rr))
                                count += 1
                        if debug:
                            logger.info('Efendiev extended [{:d}]: {:.2e}'.format(ii, rr))
                        count += 1
                        uu = Function(extended_VV, name = 'u')
                        if len(extended_facet_dirichlet):
                            uu.vector()[:] = rv
                        else:
                            vv = Function(extended_WW)
                            vv.vector()[:] = rv
                            WW_to_VV_assigner.assign(uu, vv.sub(0))
                        if debug:
                            tmp.append(uu)
                        efendiev_extended_eigenvalues.append(1./np.sqrt(rr))
                        interior_efendiev.append(fine_interpolate(uu))
                        del uu
                    new = sa_utils.get_time()
                    times_efendiev_extended.append(times_efendiev_extended[-1]+new-old)
                    if current > last:
                        last = current
                del tmp_dofs
                del count
                del eigensolver
                times_direct['efendiev_extended'] = np.array(times_efendiev_extended[1:])
                times_eigen['efendiev_extended'] = times_eigensolver_extended
                del interior_efendiev, times_eigensolver_extended
            del broken

            if debug and len(tmp):
                write_functions(outdir_bc+'/debug/efendiev_extended_uncut', tmp)
                write_functions(outdir_bc+'/debug/efendiev_extended', uus['efendiev_extended'])
                del tmp

            if matplotlib and efendiev_extended:
                efendiev_extended_eigenvalues = np.array(efendiev_extended_eigenvalues)
                efendiev_extended_len = len(efendiev_extended_eigenvalues)
                ymin = np.min([ymin, np.min(efendiev_extended_eigenvalues)])
                ymax = np.max([ymax, np.max(efendiev_extended_eigenvalues)])
                xmax = np.max([xmax, efendiev_extended_len])
                efendiev_handles.append(*efendiev_ax.semilogy(1+np.arange(efendiev_extended_len), efendiev_extended_eigenvalues, 'g+:', mec = 'g', mfc = 'none', label = r'$\omega_l^+$'))
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog(1+np.arange(efendiev_extended_len), efendiev_extended_eigenvalues, 'g+:', mec = 'g', mfc = 'none', label = r'$\omega_l^+$'))
                xx = np.log(1.+np.arange(efendiev_extended_len))
                yy = np.log(efendiev_extended_eigenvalues)
                AA = np.vstack([xx, np.ones(efendiev_extended_len)]).T
                rate, offset = la.lstsq(AA, yy)[0]
                del AA
                ydiff = np.exp(0.1*np.log(ymax/ymin))
                efendiev_log = (np.min(efendiev_extended_eigenvalues)*ydiff)
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog([1, efendiev_extended_len], [efendiev_log/efendiev_extended_len**rate, efendiev_log], 'k--', label = r'$c n^{'+("%.2g"%rate)+r'}$'))
                del efendiev_log, rate, offset, ydiff
            
            logger.info('Efendiev extended done')

        if efendiev_extended_noweight:
            if debug:
                tmp = []
            logger.info('Efendiev extended noweight')
            extended_MM_weighted = PETScMatrix()
            old = sa_utils.get_time()
            if len(extended_facet_dirichlet):
                AA = extended_KK_zero
                assemble(inner(extended_trial, extended_test)*extended_dx, tensor = extended_MM_weighted)
            else:
                AA = extended_KW
                assemble(inner(extended_uw, extended_vw)*extended_dx, tensor = extended_MM_weighted)
            new = sa_utils.get_time()
            time_extended_MM_weighted = new-old
            logger.info('    Assembled')

            eigensolver = None
            times_eigensolver_extended = [None]
            broken = False
            for deg in range(len(opt_dofs)):
                old = sa_utils.get_time()
                tmp_eigensolver = SLEPcEigenSolver(AA, extended_MM_weighted)
                tmp_eigensolver.parameters['problem_type'] = 'gen_hermitian'
                tmp_eigensolver.parameters['spectrum'] = 'smallest magnitude'
                tmp_eigensolver.parameters['spectral_shift'] = 0.
                tmp_eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
                tmp_eigensolver.parameters['tolerance'] = efendiev_eps
                tmp_eigensolver.solve(opt_dofs[deg]+null_dim)
                if tmp_eigensolver.get_number_converged() <  opt_dofs[deg]+null_dim:
                    del tmp_eigensolver
                    broken = True
                    break
                eigensolver = tmp_eigensolver
                del tmp_eigensolver
                new = sa_utils.get_time()
                times_eigensolver_extended.append(time_extended_MM_weighted+time_extended_KK_zero+new-old)
            if broken:
                if deg > 0:
                    tmp_dofs = opt_dofs[:deg].copy()
                    broken = False
                else:
                    tmp_dofs = None
            else:
                tmp_dofs = opt_dofs.copy()
            del AA
            del extended_MM_weighted
            times_eigensolver_extended = np.array(times_eigensolver_extended[1:])
            logger.info('    eigensolver ran')

            if eigensolver is not None:
                extended_converged = eigensolver.get_number_converged()
            else:
                extended_converged = 0
            if broken or extended_converged < tmp_dofs[-1]:
                efendiev_extended_noweight = False
                logger.critical('EFENDIEV: extended noweight, not enough eigenvectors converged')
                keys.remove('efendiev_extended_noweight')
            else:
                logger.info('solved, ['+str(extended_converged)+'/'+str(tmp_dofs[-1])+'] converged')
                efendiev_extended_eigenvalues = []
                interior_efendiev = []
                uus['efendiev_extended_noweight'] = interior_efendiev
                times_efendiev_extended = [0]
                dofs['efendiev_extended_noweight'] = tmp_dofs
                last = 0
                ok = True
                shift = 0 if len(extended_facet_dirichlet) else null_dim
                count = 0
                for deg in range(len(tmp_dofs)):
                    current = tmp_dofs[deg]
                    old = sa_utils.get_time()
                    for ii in range(last, current):
                        while count < extended_converged:
                            rr, cc, rv, rc = eigensolver.get_eigenpair(count)
                            if rr > myeps:
                                break
                            else:
                                if debug:
                                    logger.info('Efendiev [{:d}] skip: {:.2e}'.format(count, rr))
                                count += 1
                        if debug:
                            logger.info('Efendiev [{:d}]: {:.2e}'.format(ii, rr))
                        count += 1
                        uu = Function(extended_VV, name = 'u')
                        if len(extended_facet_dirichlet):
                            uu.vector()[:] = rv
                        else:
                            vv = Function(extended_WW)
                            vv.vector()[:] = rv
                            WW_to_VV_assigner.assign(uu, vv.sub(0))
                        if debug:
                            tmp.append(uu)
                        efendiev_extended_eigenvalues.append(1./np.sqrt(rr))
                        interior_efendiev.append(fine_interpolate(uu))
                        del uu
                    new = sa_utils.get_time()
                    times_efendiev_extended.append(times_efendiev_extended[-1]+new-old)
                    if current > last:
                        last = current
                del tmp_dofs
                del count
                del eigensolver
                times_direct['efendiev_extended_noweight'] = np.array(times_efendiev_extended[1:])
                times_eigen['efendiev_extended_noweight'] = times_eigensolver_extended
                del interior_efendiev, times_efendiev_extended

            if matplotlib and efendiev_extended_noweight:
                efendiev_extended_eigenvalues = np.array(efendiev_extended_eigenvalues)
                efendiev_len = len(efendiev_extended_eigenvalues)
                ymin = np.min([ymin, np.min(efendiev_extended_eigenvalues)])
                ymax = np.max([ymax, np.max(efendiev_extended_eigenvalues)])
                xmax = np.max([xmax, efendiev_len])
                efendiev_handles.append(*efendiev_ax.semilogy(1+np.arange(efendiev_len), efendiev_extended_eigenvalues, 'gx:', mec = 'g', mfc = 'none', label = r'$\omega_l^+$, noweight $M$'))
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog(1+np.arange(efendiev_len), efendiev_extended_eigenvalues, 'gx:', mec = 'g', mfc = 'none', label = r'$\omega_l^+$, noweight $M$'))
                xx = np.log(1.+np.arange(efendiev_len))
                yy = np.log(efendiev_extended_eigenvalues)
                AA = np.vstack([xx, np.ones(efendiev_len)]).T
                rate, offset = la.lstsq(AA, yy)[0]
                del AA
                ydiff = np.exp(0.1*np.log(ymax/ymin))
                efendiev_log = (np.min(efendiev_extended_eigenvalues)*ydiff)
                efendiev_handles_loglog.append(*efendiev_ax_loglog.loglog([1, efendiev_len], [efendiev_log/efendiev_len**rate, efendiev_log], 'k--', label = r'$c n^{'+("%.2g"%rate)+r'}$'))
                del efendiev_log, efendiev_len, rate, offset, ydiff, efendiev_extended_eigenvalues
 
            del times_eigensolver_extended
            logger.info('Efendiev extended noweight done')

        if matplotlib and (efendiev or efendiev_extended or efendiev_extended_noweight):
            logger.info('Plotting Efendiev eigenvalues')
            sa_utils.set_log_ticks(efendiev_ax_loglog, 1, xmax+1, xaxis = True)
            sa_utils.set_log_ticks(efendiev_ax_loglog, ymin, ymax)
            efendiev_ax_loglog.set_xlabel(r'$n$')
            efendiev_ax_loglog.set_ylabel(r'$\frac{1}{\sqrt{\lambda_n}}$')
            efendiev_fig_loglog.savefig(outdir_bc+'/'+prefix+'_efendiev_loglog.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in efendiev_handles_loglog]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(efendiev_handles_loglog, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_efendiev_loglog_legend.pdf', bbox_extra_artists = (lgd, ))

            del efendiev_handles_loglog, efendiev_ax_loglog, efendiev_fig_loglog

            sa_utils.set_log_ticks(efendiev_ax, 1, xmax+1, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(efendiev_ax, ymin, ymax)
            efendiev_ax.set_xlabel(r'$n$')
            efendiev_ax.set_ylabel(r'$\frac{1}{\sqrt{\lambda_n}}$')
            efendiev_fig.savefig(outdir_bc+'/'+prefix+'_efendiev.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in efendiev_handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(efendiev_handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_efendiev_legend.pdf', bbox_extra_artists = (lgd, ))

            del efendiev_fig, efendiev_ax, efendiev_handles, figlegend, ax, lgd, xmax, ymax, ymin
            logger.info('Efendiev eigenvalues plotted')

        logger.info('Building basis orthogonal to null space')
        old = sa_utils.get_time()
        def get_harmonic(kk):
            uu = Function(extended_VV)
            uu.vector()[kk] = 1.
            return extended_harmonic(uu)
        harmonic_hats = sa_utils.simple_batch_fun(boundary_dofs, extended_VV, low = 0, high = num_boundary, fun = get_harmonic, logger = batch_logger, max_procs = batch_procs)
        new = sa_utils.get_time()
        time_harmonic_hats = new-old
        logger.info('Boundary basis constructed [{:d}] functions in [{:.2f}s = {:s}]'.format(len(harmonic_hats), time_harmonic_hats, sa_utils.human_time(time_harmonic_hats)))
        if debug:
            write_functions(outdir_bc+'/debug/harmonic_hats', harmonic_hats)

        old = sa_utils.get_time()
        extended_orthogonalized = []
        skipped = 0
        if orthogonalize_hats:
            logger.info('orthogonalizing hats')
            for uu in harmonic_hats:
                if extended_orthogonalize(uu, extended_orthogonalized):
                    extended_orthogonalized.append(uu)
                else:
                    skipped += 1
        else:
            logger.info('normalizing hats')
            for uu in harmonic_hats:
                if extended_orthogonalize(uu, []):
                    extended_orthogonalized.append(uu)
                else:
                    skipped += 1
        del harmonic_hats
        extended_harmonic_dim = len(extended_orthogonalized)
        interior_orthogonalized = sa_utils.simple_batch_fun(extended_orthogonalized, fine_VV, low = 0, high = extended_harmonic_dim, fun = fine_interpolate, logger = batch_logger, max_procs = batch_procs)
        new = sa_utils.get_time()
        time_boundary_basis = new-old+time_dofmap+time_harmonic_hats
        logger.info('{:d}-dimensional Basis of harmonic functions constructed in [{:.2f}s = {:s}], [{:d}] functions skipped'.format(extended_harmonic_dim, time_boundary_basis, sa_utils.human_time(time_boundary_basis), skipped))
        del skipped
        if debug:
            print('lengths: extended [{:d}], interior [{:d}]'.format(len(extended_orthogonalized), len(interior_orthogonalized)))
            write_functions(outdir_bc+'/debug/extended_orthogonalized', extended_orthogonalized)
            write_functions(outdir_bc+'/debug/interior_orthogonalized', interior_orthogonalized)
 
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
                processes.append(multiprocessing.Process(target = product_fun, args = (low, high, times_queue)))
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
        logger.info(r'Harmonic hat matrices set up in [{:.2f}s = {:s}]'.format(time_boundary_matrices, sa_utils.human_time(time_boundary_matrices)))
        if debug:
            spec_extended_KK = la.eigvalsh(extended_KK_harmonic)
            spec_interior_KK = la.eigvalsh(interior_KK_harmonic)
            spec_interior_MM = la.eigvalsh(interior_MM_harmonic)
            spec_interior_H1 = la.eigvalsh(interior_H1_harmonic)
            logger.info(r'diagonals:\rextended KK: {:s}\rinterior KK: {:s}\rinterior MM: {:s}\rinterior H1: {:s}'.format(str(np.diag(extended_KK_harmonic)), str(np.diag(interior_KK_harmonic)), str(np.diag(interior_MM_harmonic)), str(np.diag(interior_H1_harmonic))))
            logger.info(r'Spectrum:\rextended KK: {:s}\rinterior KK: {:s}\rinterior MM: {:s}\rinterior H1: {:s}'.format(str(spec_extended_KK), str(spec_interior_KK), str(spec_interior_MM), str(spec_interior_H1)))

        if subspaceE or subspace1 or subspace0:
            old = sa_utils.get_time()
            extended_subspace = []
            interior_subspace = []
            count = 0
            count_null = 0

            for kk, uu in enumerate(extended_harmonic_polys_neumann[harmonic_offset_nobc:]):
                ret = extended_orthogonalize(uu, extended_subspace)
                if ret < ortho_eps:
                    logger.info('harmonic poly skip ['+str(kk)+', '+str(ret)+']')
                    count_null += 1
                else:
                    extended_subspace.append(uu)
                    interior_subspace.append(fine_interpolate(uu))
                    count += 1
            extended_subspace_dim = len(extended_subspace)
            logger.info('extended subspace dim: {:d}, count: {:d}, count_null: {:d}'.format(extended_subspace_dim, count, count_null))
            new = sa_utils.get_time()
            time_subspace_basis = extended_harmonic_times_neumann[-1]+new-old

            old = sa_utils.get_time()
            extended_KK_subspace = sa_utils.get_shared_array((extended_subspace_dim, extended_subspace_dim))
            interior_KK_subspace = sa_utils.get_shared_array((extended_subspace_dim, extended_subspace_dim))
            interior_MM_subspace = sa_utils.get_shared_array((extended_subspace_dim, extended_subspace_dim))
            interior_H1_subspace = sa_utils.get_shared_array((extended_subspace_dim, extended_subspace_dim))

            times_dict = dict()
            times_queue = multiprocessing.Queue()
            def product_fun(low, high, times_q):
                old = sa_utils.get_time()
                for ii in range(low, high):
                    extended_KK_subspace[ii, ii] = extended_subspace[ii].vector().inner(extended_KK*extended_subspace[ii].vector())
                    interior_KK_subspace[ii, ii] = extended_subspace[ii].vector().inner(interior_KK*extended_subspace[ii].vector())
                    interior_MM_subspace[ii, ii] = extended_subspace[ii].vector().inner(interior_MM*extended_subspace[ii].vector())
                    interior_H1_subspace[ii, ii] = extended_subspace[ii].vector().inner(interior_H1*extended_subspace[ii].vector())
                    for jj in range(ii):
                        extended_KK_subspace[ii, jj] = extended_subspace[jj].vector().inner(extended_KK*extended_subspace[ii].vector())
                        interior_KK_subspace[ii, jj] = extended_subspace[jj].vector().inner(interior_KK*extended_subspace[ii].vector())
                        interior_MM_subspace[ii, jj] = extended_subspace[jj].vector().inner(interior_MM*extended_subspace[ii].vector())
                        interior_H1_subspace[ii, jj] = extended_subspace[jj].vector().inner(interior_H1*extended_subspace[ii].vector())
                        extended_KK_subspace[jj, ii] = extended_KK_subspace[ii, jj]
                        interior_KK_subspace[jj, ii] = interior_KK_subspace[ii, jj]
                        interior_MM_subspace[jj, ii] = interior_MM_subspace[ii, jj]
                        interior_H1_subspace[jj, ii] = interior_H1_subspace[ii, jj]
                new = sa_utils.get_time()
                times_dict[low] = new-old
                times_q.put(times_dict)
            if cpu_count > 1:
                processes = []
                block = 20
                low = 0
                high = np.min([low+block, extended_subspace_dim])
                while(low < high):
                    processes.append(multiprocessing.Process(target = product_fun, args = (low, high, times_queue)))
                    low = high
                    high = np.min([low+block, extended_subspace_dim])
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
                product_fun(0, extended_subspace_dim, times_queue)
                times_dict.update(times_queue.get())
            for tt in times_dict.values():
                old -= tt
            del product_fun, times_dict, times_queue
            new = sa_utils.get_time()
            time_subspace_matrices = new-old
            gc.collect()
            logger.info('Subspace matrices assembled in [{:.2f}s = {:s}]'.format(time_subspace_matrices, sa_utils.human_time(time_subspace_matrices)))

        if liptonE:
            dofs['liptonE'] = opt_dofs.copy()
        if lipton1:
            dofs['lipton1'] = opt_dofs.copy()
        if lipton0:
            dofs['lipton0'] = opt_dofs.copy()
        if subspaceE:
            dofs['subspaceE'] = opt_dofs.copy()
        if subspace1:
            dofs['subspace1'] = opt_dofs.copy()
        if subspace0:
            dofs['subspace0'] = opt_dofs.copy()

        min_dof = np.min([dofs[key][0] for key in keys])
        max_dof = np.max([dofs[key][-1] for key in keys])

        logger.info('Energy optimal')
        old = sa_utils.get_time()
        eigvals, eigvecs = la.eigh(interior_KK_harmonic, extended_KK_harmonic)
        new = sa_utils.get_time()
        time_eigensolver = new-old
        idx = np.argsort(eigvals)[::-1]
        eigvals_max = eigvals[idx[0]]
        max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
        logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
        nwidthsE = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
        nwidthE_limit = len(nwidthsE)

        if liptonE:
            ev_harmonic = eigvecs[:, idx]
            optimal_E = []
            times_assemble = [0] 
            logger.info(' creating functions')
            last = 0

            for deg in range(len(opt_dofs)):
                current = opt_dofs[deg]
                if current >= nwidthE_limit:
                    dofs['liptonE'] = opt_dofs[:deg].copy()
                    break
                old = sa_utils.get_time()
                for kk in range(last, current):
                    uu = Function(fine_VV, name = 'u')
                    for jj in range(extended_harmonic_dim):
                        uu.vector().axpy(ev_harmonic[jj, kk], interior_orthogonalized[jj].vector())
                    optimal_E.append(uu)
                    del uu
                new = sa_utils.get_time()
                times_assemble.append(times_assemble[-1]+new-old)
                last = current
            
            times_direct['liptonE'] = np.array(times_assemble[1:])+time_boundary_basis+time_boundary_matrices
            times_eigen['liptonE'] = np.ones(len(opt_dofs))*time_eigensolver
            uus['liptonE'] = optimal_E

            if debug:
                optimal_E_extended = []
                last = 0
                for deg in range(len(opt_dofs)):
                    current = opt_dofs[deg]
                    if current >= nwidthE_limit:
                        break
                    for kk in range(last, current):
                        uu = Function(extended_VV, name = 'u')
                        for jj in range(extended_harmonic_dim):
                            uu.vector().axpy(ev_harmonic[jj, kk], extended_orthogonalized[jj].vector())
                        optimal_E_extended.append(uu)
                        del uu
                    last = current
                print('lengths: liptonE_extended [{:d}], liptonE [{:d}]'.format(len(optimal_E_extended), len(optimal_E)))
                write_functions(outdir_bc+'/debug/liptonE_extended', optimal_E_extended)
                write_functions(outdir_bc+'/debug/liptonE', optimal_E)
                del optimal_E_extended

            del optimal_E
            logger.info('created')

        logger.info('L2 optimal')
        old = sa_utils.get_time()
        eigvals, eigvecs = la.eigh(interior_MM_harmonic, extended_KK_harmonic)
        new = sa_utils.get_time()
        time_eigensolver = new-old
        idx = np.argsort(eigvals)[::-1]
        eigvals_max = eigvals[idx[0]]
        max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
        logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
        nwidths0 = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
        nwidth0_limit = len(nwidths0)
        if lipton0:
            ev_harmonic = eigvecs[:, idx]
            optimal_0 = []
            times_assemble = [0] 
            logger.info(' creating functions')
            last = 0
            for deg in range(len(opt_dofs)):
                current = opt_dofs[deg]
                if current > nwidth0_limit:
                    dofs['lipton0'] = opt_dofs[:deg].copy()
                    break
                old = sa_utils.get_time()
                for kk in range(last, current):
                    uu = Function(fine_VV, name = 'u')
                    for jj in range(extended_harmonic_dim):
                        uu.vector().axpy(ev_harmonic[jj, kk], interior_orthogonalized[jj].vector())
                    optimal_0.append(uu)
                new = sa_utils.get_time()
                times_assemble.append(times_assemble[-1]+new-old)
                last = current
            times_direct['lipton0'] = np.array(times_assemble[1:])+time_boundary_basis+time_boundary_matrices
            times_eigen['lipton0'] = np.ones(len(opt_dofs))*time_eigensolver
            uus['lipton0'] = optimal_0
            del optimal_0
            logger.info('created')

        logger.info('H1 optimal')
        old = sa_utils.get_time()
        eigvals, eigvecs = la.eigh(interior_H1_harmonic, extended_KK_harmonic)
        new = sa_utils.get_time()
        time_eigensolver = new-old
        idx = np.argsort(eigvals)[::-1]
        eigvals_max = eigvals[idx[0]]
        max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
        logger.info('last ['+str(extended_harmonic_dim-max_idx-1)+'] eigvecs vanish')
        nwidths1 = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
        nwidth1_limit = len(nwidths1)
        if lipton1:
            ev_harmonic = eigvecs[:, idx]
            optimal_1 = []
            times_assemble = [0] 
            logger.info(' creating functions')
            last = 0

            for deg in range(len(opt_dofs)):
                current = opt_dofs[deg]
                if current >= nwidth1_limit:
                    dofs['lipton1'] = opt_dofs[:deg].copy()
                    break
                old = sa_utils.get_time()
                for kk in range(last, current):
                    uu = Function(fine_VV, name = 'u')
                    for jj in range(extended_harmonic_dim):
                        uu.vector().axpy(ev_harmonic[jj, kk], interior_orthogonalized[jj].vector())
                    optimal_1.append(uu)
                    del uu
                new = sa_utils.get_time()
                times_assemble.append(times_assemble[-1]+new-old)
                last = current

            times_direct['lipton1'] = np.array(times_assemble[1:])+time_boundary_basis+time_boundary_matrices
            times_eigen['lipton1'] = np.ones(len(opt_dofs))*time_eigensolver
            uus['lipton1'] = optimal_1
            del optimal_1
            logger.info('created')
        
        if subspaceE:
            logger.info('Energy subspace optimal')
            try:
                old = sa_utils.get_time()
                eigvals, eigvecs = la.eigh(interior_KK_subspace, extended_KK_subspace)
                new = sa_utils.get_time()
            except:
                subspaceE = False
                if 'subspaceE' in keys:
                    keys.remove('subspaceE')
                logger.info('Energy subspace optimal eigenvalues not computed')
            else:
                time_eigensolver = new-old
                idx = np.argsort(eigvals)[::-1]
                eigvals_max = eigvals[idx[0]]
                max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
                logger.info('last ['+str(extended_subspace_dim-max_idx-1)+'] eigvecs vanish')
                nwidthsE_subspace = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
                nwidthE_limit_subspace = len(nwidthsE_subspace)
                if subspaceE:
                    ev_harmonic = eigvecs[:, idx]
                    optimal_E_subspace = []
                    times_assemble = [0] 
                    logger.info(' creating functions')
                    last = 0
                    for deg in range(len(opt_dofs)):
                        current = opt_dofs[deg]
                        if current >= nwidthE_limit_subspace:
                            dofs['subspaceE'] = opt_dofs[:deg].copy()
                            break
                        old = sa_utils.get_time()
                        for kk in range(last, current):
                            uu = Function(fine_VV, name = 'u')
                            for jj in range(extended_subspace_dim):
                                uu.vector().axpy(ev_harmonic[jj, kk], interior_subspace[jj].vector())
                            optimal_E_subspace.append(uu)
                        new = sa_utils.get_time()
                        times_assemble.append(times_assemble[-1]+new-old)
                        last = current
                    times_direct['subspaceE'] = np.array(times_assemble[1:])+time_subspace_basis+time_subspace_matrices
                    times_eigen['subspaceE'] = np.ones(len(dofs['subspaceE']))*time_eigensolver
                    uus['subspaceE'] = optimal_E_subspace
                    del optimal_E_subspace
                    logger.info('created')
            finally:
                del interior_KK_subspace

        if subspace1: 
            logger.info('H1 subspace optimal')
            try:
                old = sa_utils.get_time()
                eigvals, eigvecs = la.eigh(interior_H1_subspace, extended_KK_subspace)
                new = sa_utils.get_time()
            except:
                subspace1 = False
                if 'subspace1' in keys:
                    keys.remove('subspace1')
                logger.info('H1 subspace optimal eigenvalues not computed')
            else:
                time_eigensolver = new-old
                idx = np.argsort(eigvals)[::-1]
                eigvals_max = eigvals[idx[0]]
                max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
                logger.info('last ['+str(extended_subspace_dim-max_idx-1)+'] eigvecs vanish')
                nwidths1_subspace = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
                nwidth1_limit_subspace = len(nwidths1_subspace)
                if subspace1:
                    ev_harmonic = eigvecs[:, idx]
                    optimal_1_subspace = []
                    times_assemble = [0] 
                    logger.info(' creating functions')
                    last = 0
                    for deg in range(len(opt_dofs)):
                        current = opt_dofs[deg]
                        if current >= nwidth1_limit_subspace:
                            dofs['subspace1'] = opt_dofs[:deg].copy()
                            break
                        old = sa_utils.get_time()
                        for kk in range(last, current):
                            uu = Function(fine_VV, name = 'u')
                            for jj in range(extended_subspace_dim):
                                uu.vector().axpy(ev_harmonic[jj, kk], interior_subspace[jj].vector())
                            optimal_1_subspace.append(uu)
                        new = sa_utils.get_time()
                        times_assemble.append(times_assemble[-1]+new-old)
                        last = current
                    times_direct['subspace1'] = np.array(times_assemble[1:])+time_subspace_basis+time_subspace_matrices
                    times_eigen['subspace1'] = np.ones(len(dofs['subspace1']))*time_eigensolver
                    uus['subspace1'] = optimal_1_subspace
                    del optimal_1_subspace
                    logger.info('created')
            finally:
                del interior_H1_subspace
       
        if subspace0:
            logger.info('L2 subspace optimal')
            try:
                old = sa_utils.get_time()
                eigvals, eigvecs = la.eigh(interior_MM_subspace, extended_KK_subspace)
                new = sa_utils.get_time()
            except:
                subspace0 = False
                if 'subspace0' in keys:
                    keys.remove('subspace0')
                logger.info('L2 subspace optimal eigenvalues not computed')
            else:
                time_eigensolver = new-old
                idx = np.argsort(eigvals)[::-1]
                eigvals_max = eigvals[idx[0]]
                max_idx = np.max(np.where(eigvals[idx] >= ortho_eps2*eigvals_max)[0])
                logger.info('last ['+str(extended_subspace_dim-max_idx-1)+'] eigvecs vanish')
                nwidths0_subspace = np.sqrt(eigvals[idx[:np.min([max_idx+1, max_dof+1])]])
                nwidth0_limit_subspace = len(nwidths0_subspace)
                if subspace0:
                    ev_harmonic = eigvecs[:, idx]
                    optimal_0_subspace = []
                    times_assemble = [0] 
                    logger.info(' creating functions')
                    last = 0
                    for deg in range(len(opt_dofs)):
                        current = opt_dofs[deg]
                        if current > nwidth0_limit_subspace:
                            dofs['subspace0'] = opt_dofs[:deg].copy()
                            break
                        old = sa_utils.get_time()
                        for kk in range(last, current):
                            uu = Function(fine_VV, name = 'u')
                            for jj in range(extended_subspace_dim):
                                uu.vector().axpy(ev_harmonic[jj, kk], interior_subspace[jj].vector())
                            optimal_0_subspace.append(uu)
                        new = sa_utils.get_time()
                        times_assemble.append(times_assemble[-1]+new-old)
                        last = current
                    times_direct['subspace0'] = np.array(times_assemble[1:])+time_subspace_basis+time_subspace_matrices
                    times_eigen['subspace0'] = np.ones(len(dofs['subspace0']))*time_eigensolver
                    uus['subspace0'] = optimal_0_subspace
                    del optimal_0_subspace
                    logger.info('created')
            finally:
                del interior_MM_subspace
                
        if subspace0 or subspace1 or subspaceE:
            del extended_KK_subspace
        
        numkeys = len(keys)
        peterseim_keys = []
        peterseim_keys_inv = []
        for key in keys:
            if 'peterseim' in key:
                peterseim_keys.append(key)
            else:
                peterseim_keys_inv.append(key)

        logger.info('Removing linearly dependent shape functions')
        logger.info('Ortho: '+str(ortho))
        times_ortho = dict()
        for key in peterseim_keys_inv:
            count = 0
            orthogonalized = []
            new_uus = []
            new_dofs = []
            tmp_dofs = dofs[key]
            tmp_uus = uus[key]
            logger.info('['+key+'] starting, ['+str(len(tmp_uus))+'] functions preset')
            last = 0
            count_null = 0
            tmp_times_direct = times_direct[key]
            tmp_times_eigen = times_eigen[key]
            new_times_direct = []
            new_times_eigen = []
            times_ortho[key] = [0]
            for deg in range(len(tmp_dofs)):
                old = sa_utils.get_time()
                current = tmp_dofs[deg]
                last_count = count
                for kk in range(last, current):
                    uu = tmp_uus[kk].copy(True)
                    ret = fine_orthogonalize(uu, orthogonalized)
                    if not len(extended_facet_dirichlet):
                        fine_orthogonalize_null(uu.vector())
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
                    new_times_direct.append(tmp_times_direct[deg])
                    new_times_eigen.append(tmp_times_eigen[deg])
                    times_ortho[key].append(times_ortho[key][-1]+new-old)
                last = current
            logger.info('['+key+'] finished, ['+str(count_null)+'] functions removed ['+str(count)+'] remaining')
            dofs[key] = np.array(new_dofs)
            uus[key] = new_uus
            times_ortho[key] = np.array(times_ortho[key][:-1])
            times_direct[key] = np.array(new_times_direct)
            times_eigen[key] = np.array(new_times_eigen)
            coarse_spaces[key] = []
            for coarse_deg in dofs[key]:
                coarse_spaces[key].append(uus[key][:coarse_deg])
            del orthogonalized
            del new_uus
            del new_dofs
            del tmp_uus
            logger.info('['+key+'] old dofs:'+str(tmp_dofs))
            del tmp_dofs
            logger.info('['+key+'] dofs:    '+str(dofs[key]))
        for key in peterseim_keys:
            logger.info('['+key+'] starting')
            tmp_dofs = dofs[key]
            new_dofs = []
            times_ortho[key] = []
            last_count = 0
            tmp_times_direct = times_direct[key]
            tmp_times_eigen = times_eigen[key]
            new_times_direct = []
            new_times_eigen = []
            for deg in range(len(tmp_dofs)):
                logger.info('['+key+'], degree ['+str(deg)+'], ['+str(tmp_dofs[deg])+'] present')
                old = sa_utils.get_time()
                orthogonalized = []
                count = 0
                count_null = 0
                tmp_uus = coarse_spaces[key][deg]
                new_uus = []
                for kk in range(tmp_dofs[deg]):
                    uu = tmp_uus[kk].copy(True)
                    ret = fine_orthogonalize(uu, orthogonalized)
                    if ret < ortho_eps:
                        count_null += 1
                    else:
                        orthogonalized.append(uu)
                        if ortho:
                            new_uus.append(uu)
                        else:
                            new_uus.append(tmp_uus[kk])
                        count += 1
                del orthogonalized
                new = sa_utils.get_time()
                if count > last_count:
                    new_dofs.append(count)
                    new_times_direct.append(tmp_times_direct[deg])
                    new_times_eigen.append(tmp_times_eigen[deg])
                    coarse_spaces[key][deg] = new_uus
                    times_ortho[key].append(new-old)
                del new_uus
                del tmp_uus
                last_count = count
                logger.info('['+key+'], degree ['+str(deg)+'], ['+str(count_null)+'] functions removed ['+str(count)+'] remaining')
            dofs[key] = np.array(new_dofs)
            times_direct[key] = np.array(new_times_direct)
            times_eigen[key] = np.array(new_times_eigen)
            del new_dofs
            del tmp_dofs
            logger.info('['+key+'] dofs: '+str(dofs[key]))

        logger.info('Filtering with respect to nwidths and stiffness condition')
        times_eigen_filtered = dict()
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
                coarse_spaces[key] = coarse_spaces[key][:coarse_deg]
                logger.info('['+key+'] shortened by ['+str(count)+'] degrees from ['+str(dof0)+'] to ['+str(dof1)+']')
                times_direct[key] = times_direct[key][:coarse_deg]
                times_eigen[key] = times_eigen[key][:coarse_deg]
                if np.min(times_eigen[key]) > 0:
                    times_eigen_filtered[key] = times_eigen[key]
                times_compute[key] = np.array(times_direct[key]+times_eigen[key])
                logger.info('['+key+'] lengths: ['+str(dofs[key][-1])+'] < ['+str(nwidthE_limit)+', '+str(nwidth0_limit)+', '+str(nwidth1_limit)+']')
        for key in remove_keys:
            keys.remove(key)
        numkeys = len(keys)
        keys_filtered = list(times_eigen_filtered.keys())

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
        for ii in range(1, np.min([nwidthE_limit, nwidth0_limit, nwidth1_limit])):
        #for ii in range(min_dof+1, np.min([nwidthE_limit, nwidth0_limit, nwidth1_limit])):
            denom = 1./np.log(1.*(ii+1)/ii)
            suprrE = ln(nwidthsE[ii]/nwidthsE[ii-1])*denom
            suprr1 = ln(nwidths0[ii]/nwidths0[ii-1])*denom
            suprr0 = ln(nwidths0[ii]/nwidths0[ii-1])*denom
            matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n".format(ii, nwidthsE[ii], suprrE, nwidths1[ii], suprr1, nwidths0[ii], suprr0))
        matfile.close()

        supinfsE = dict()
        supinfs0 = dict()
        supinfs1 = dict()
        conditions_stiffness = dict()
        supinfsE_rates = dict()
        supinfs0_rates = dict()
        supinfs1_rates = dict()
        times_direct_rates = dict()
        times_eigen_rates = dict()
        times_compute_rates = dict()
        supinfsE_compute_rates = dict()
        supinfs0_compute_rates = dict()
        supinfs1_compute_rates = dict()
        conditions_stiffness_rates = dict()

        gc.collect()
        
        logger.info('Computing stuff with coarse bases')
        keys_count = multiprocessing.cpu_count() if max_keys is None else np.min([multiprocessing.cpu_count(), max_keys])

        def compute_stuff_for_key(key, supinfsE_q, supinfs0_q, supinfs1_q, 
                                  conditions_stiffness_q, 
                                  supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, 
                                  times_direct_rates_q, times_eigen_rates_q, times_compute_rates_q, 
                                  supinfsE_compute_rates_q, supinfs0_compute_rates_q, supinfs1_compute_rates_q, 
                                  conditions_stiffness_rates_q, 
                                  done_q):
            if write_shapes:
                hardname = prefix+'_shapes_'+key
                logger.info('['+key+'] plotting examples of shape functions')
                write_strain_stress(coarse_spaces[key][-1], outdir_bc+'/strain_stress/'+hardname)
                write_functions(outdir_bc+'/coeffs/'+hardname, coarse_spaces[key][-1])
                logger.info('['+key+'] plotted examples of shape functions')

            if not compute_supinfs:
                done_q.put(None)
                return
            
            logger.info('['+key+'] major assembly starting')

            finest_dim = dofs[key][-1]
            finest_space = coarse_spaces[key][-1]
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
                    matrix_processes.append(multiprocessing.Process(target = matrix_populate, args = (low, high, matrix_queue)))
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
                    eval_processes.append(multiprocessing.Process(target = embedded_fun, args = (low, high, eval_queue)))
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

            logger.info('['+key+'] Computing rates etc for plots')

            basename = prefix+'_'+key
            supinfE = supinfsE[key]
            supinf0 = supinfs0[key]
            supinf1 = supinfs1[key]
            dof = dofs[key]
            time_direct = times_direct[key]
            time_eigen = times_eigen[key]
            time_compute = times_compute[key]

            if_eigen = key in keys_filtered

            denom = 1./(np.log(dof[-1])-np.log(dof[0]))
            suprrE = (np.log(supinfE[-1])-np.log(supinfE[0]))*denom
            supinfsE_rates[key] = suprrE
            suprr0 = (np.log(supinf0[-1])-np.log(supinf0[0]))*denom
            supinfs0_rates[key] = suprr0
            suprr1 = (np.log(supinf1[-1])-np.log(supinf1[0]))*denom
            supinfs1_rates[key] = suprr1
            time_direct_r = (np.log(time_direct[-1])-np.log(time_direct[0]))*denom
            times_direct_rates[key] = time_direct_r
            if if_eigen:
                time_eigen_r = (np.log(time_eigen[-1])-np.log(time_eigen[0]))*denom
            else:
                time_eigen_r = 0
            times_eigen_rates[key] = time_eigen_r
            time_compute_r = (np.log(time_compute[-1])-np.log(time_compute[0]))*denom
            times_compute_rates[key] = time_compute_r
            supinfsE_compute_r = (np.log(supinfE[-1])-np.log(supinfE[0]))/(np.log(time_compute[-1])-np.log(time_compute[0]))
            supinfsE_compute_rates[key] = supinfsE_compute_r
            supinfs0_compute_r = (np.log(supinf0[-1])-np.log(supinf0[0]))/(np.log(time_compute[-1])-np.log(time_compute[0]))
            supinfs0_compute_rates[key] = supinfs0_compute_r
            supinfs1_compute_r = (np.log(supinf1[-1])-np.log(supinf1[0]))/(np.log(time_compute[-1])-np.log(time_compute[0]))
            supinfs1_compute_rates[key] = supinfs1_compute_r
            conditions_stiffness_r = (np.log(conditions_stiffness[key][-1])-np.log(conditions_stiffness[key][0]))*denom
            conditions_stiffness_rates[key] = conditions_stiffness_r
            
            matfile = open(outdir_bc+'/'+basename+'_supinfs.csv', 'w')
            matfile.write('dof, supinfE, lambdaE, supinfEr, supinf1, lambda1, supinf1r, supinf0, lambda0, supinf0r, direct, directr, eigen, eigenr, compute, computer, supinfEcomputer, supinf0computer, condstiffness\n')
            matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3f}, {:.3f}, {:.3e}\n'.format(fine_dim, 0, 0, suprrE, 0, 0, suprr1, 0, 0, suprr0, 0, time_direct_r, 0, time_eigen_r, 0, time_compute_r, supinfsE_compute_r, supinfs0_compute_r, 0))
            matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3f}, {:.3f}, {:.3e}\n'.format(dof[0], supinfE[0], supinfE[0]/nwidthsE[dof[0]], 0, supinf1[0], supinf1[0]/nwidths1[dof[0]], 0, supinf0[0], supinf0[0]/nwidths0[dof[0]], 0, time_direct[0], 0, time_eigen[0], 0, time_compute[0], 0, 0, 0, conditions_stiffness[key][0]))
            for ii in range(1, len(dof)):
                denom = 1./np.log(1.*dof[ii]/dof[ii-1])
                suprrE = ln(supinfE[ii]/supinfE[ii-1])*denom
                suprr1 = ln(supinf1[ii]/supinf1[ii-1])*denom
                suprr0 = ln(supinf0[ii]/supinf0[ii-1])*denom
                time_direct_r = (np.log(time_direct[ii])-np.log(time_direct[ii-1]))*denom
                if if_eigen:
                    time_eigen_r = (np.log(time_eigen[ii])-np.log(time_eigen[ii-1]))*denom
                else:
                    time_eigen_r = 0
                time_compute_r = (np.log(time_compute[ii])-np.log(time_compute[ii-1]))*denom
                supinfsE_compute_r = (np.log(supinfE[ii])-np.log(supinfE[ii-1]))/(np.log(time_compute[ii])-np.log(time_compute[ii-1]))
                supinfs1_compute_r = (np.log(supinf1[ii])-np.log(supinf1[ii-1]))/(np.log(time_compute[ii])-np.log(time_compute[ii-1]))
                supinfs0_compute_r = (np.log(supinf0[ii])-np.log(supinf0[ii-1]))/(np.log(time_compute[ii])-np.log(time_compute[ii-1]))
                matfile.write("{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3f}, {:.3f}, {:.3e}\n".format(dof[ii], supinfE[ii], supinfE[ii]/nwidthsE[dof[ii]], suprrE, supinf1[ii], supinf1[ii]/nwidths1[dof[ii]], suprr1, supinf0[ii], supinf0[ii]/nwidths0[dof[ii]], suprr0, time_direct[ii], time_direct_r, time_eigen[ii], time_eigen_r, time_compute[ii], time_compute_r, supinfsE_compute_r, supinfs0_compute_r, conditions_stiffness[key][ii]))
            matfile.close()

            logger.info('['+key+'] global rates computed')

            supinfsE_q.put(supinfsE); supinfs0_q.put(supinfs0); supinfs1_q.put(supinfs1)
            conditions_stiffness_q.put(conditions_stiffness)
            supinfsE_rates_q.put(supinfsE_rates); supinfs0_rates_q.put(supinfs0_rates); supinfs1_rates_q.put(supinfs1_rates)
            times_direct_rates_q.put(times_direct_rates); times_eigen_rates_q.put(times_eigen_rates); times_compute_rates_q.put(times_compute_rates)
            supinfsE_compute_rates_q.put(supinfsE_compute_rates); supinfs0_compute_rates_q.put(supinfs0_compute_rates); supinfs1_compute_rates_q.put(supinfs1_compute_rates)
            conditions_stiffness_rates_q.put(conditions_stiffness_rates)
            logger.info('['+key+'] everything put in queues')

            done_q.put(None)


        done_q = multiprocessing.Queue()

        compute_processes = []
        supinfsE_q = multiprocessing.Queue(); supinfs0_q = multiprocessing.Queue(); supinfs1_q = multiprocessing.Queue(); 
        conditions_stiffness_q = multiprocessing.Queue()
        supinfsE_rates_q = multiprocessing.Queue(); supinfs0_rates_q = multiprocessing.Queue(); supinfs1_rates_q = multiprocessing.Queue()
        times_direct_rates_q = multiprocessing.Queue(); times_eigen_rates_q = multiprocessing.Queue(); times_compute_rates_q = multiprocessing.Queue()
        supinfsE_compute_rates_q = multiprocessing.Queue(); supinfs0_compute_rates_q = multiprocessing.Queue(); supinfs1_compute_rates_q = multiprocessing.Queue()
        conditions_stiffness_rates_q = multiprocessing.Queue()
        if keys_count > 1:
            for key in keys:
                proc = multiprocessing.Process(target = compute_stuff_for_key, args = (key, supinfsE_q, supinfs0_q, supinfs1_q, 
                                                                                   conditions_stiffness_q, 
                                                                                   supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, 
                                                                                   times_direct_rates_q, times_eigen_rates_q, times_compute_rates_q, 
                                                                                   supinfsE_compute_rates_q, supinfs0_compute_rates_q, supinfs1_compute_rates_q, 
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
                    times_direct_rates.update(times_direct_rates_q.get()); times_eigen_rates.update(times_eigen_rates_q.get()); times_compute_rates.update(times_compute_rates_q.get())
                    supinfsE_compute_rates.update(supinfsE_compute_rates_q.get()); supinfs0_compute_rates.update(supinfs0_compute_rates_q.get()); supinfs1_compute_rates.update(supinfs1_compute_rates_q.get())
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
                                      times_direct_rates_q, times_eigen_rates_q, times_compute_rates_q, 
                                      supinfsE_compute_rates_q, supinfs0_compute_rates_q, supinfs1_compute_rates_q, 
                                      conditions_stiffness_rates_q, 
                                      done_q)
                done_q.get()
                logger.info('process [{:d}] done'.format(kk+1))
                if compute_supinfs:
                    supinfsE.update(supinfsE_q.get()); supinfs0.update(supinfs0_q.get()); supinfs1.update(supinfs1_q.get()); 
                    conditions_stiffness.update(conditions_stiffness_q.get())
                    supinfsE_rates.update(supinfsE_rates_q.get()); supinfs0_rates.update(supinfs0_rates_q.get()); supinfs1_rates.update(supinfs1_rates_q.get())
                    times_direct_rates.update(times_direct_rates_q.get()); times_eigen_rates.update(times_eigen_rates_q.get()); times_compute_rates.update(times_compute_rates_q.get())
                    supinfsE_compute_rates.update(supinfsE_compute_rates_q.get()); supinfs0_compute_rates.update(supinfs0_compute_rates_q.get()); supinfs1_compute_rates.update(supinfs1_compute_rates_q.get())
                    conditions_stiffness_rates.update(conditions_stiffness_rates_q.get())
                    logger.info('process [{:d}] data fetches'.format(kk+1))
            logger.info('all keys done')

        del supinfsE_q, supinfs0_q, supinfs1_q, conditions_stiffness_q, supinfsE_rates_q, supinfs0_rates_q, supinfs1_rates_q, times_direct_rates_q, times_eigen_rates_q, times_compute_rates_q
        del supinfsE_compute_rates_q, supinfs0_compute_rates_q, supinfs1_compute_rates_q, conditions_stiffness_rates_q, done_q
        del compute_processes, uus, coarse_spaces, fine_KK, interior_H1_harmonic, interior_MM_harmonic, interior_KK_harmonic, extended_KK_harmonic
        gc.collect()

        if not compute_supinfs:
            return

        if matplotlib:
            logger.info('matplotlib plots')
            minx = min_dof; maxx = max_dof
            doflimits = np.array([min_dof, np.sqrt(min_dof*max_dof), max_dof], dtype = float)
            dofrange = np.arange(min_dof, max_dof+1, dtype = int)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidthsE[dofrange]); maxy = np.max(nwidthsE[dofrange]);
            for key in keys:
                supinf = supinfsE[key]
                rr = supinfsE_rates[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            miny0 = np.min([supinfsE[key][0] for key in keys]); maxy0 = np.max([supinfsE[key][0] for key in keys])
            minr = np.min([supinfsE_rates[key] for key in keys]); maxr = np.max([supinfsE_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidthsE[dofrange], 'k-', mfc = 'none', label = '$d^E_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfsE.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfsE_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.max(nwidthsE[dofrange]); maxy = np.max(nwidthsE[dofrange]);
            for key in keys:
                supinf = supinfsE[key]
                rr = supinfsE_rates[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf)
                miny = np.min([ymin, miny])
            miny0 = np.min([supinfsE[key][0] for key in keys]); maxy0 = np.max([supinfsE[key][0] for key in keys])
            minr = np.min([supinfsE_rates[key] for key in keys]); maxr = np.max([supinfsE_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidthsE[dofrange], 'k-', mfc = 'none', label = '$d^E_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy0)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfsE_rescaled.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = 0.9; maxy = 2.;
            for key in keys:
                supinf = supinfsE[key]
                nE = nwidthsE[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/nE, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/nE); ymax = np.max(supinf/nE)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambdaE.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil(numkeys/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil(numkeys/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = 0.9; maxy = 2.;
            for key in keys:
                supinf = supinfsE[key]
                nE = nwidthsE[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/nE, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/nE); ymax = np.max(supinf/nE)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            maxy = np.min([miny*1e2, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambdaE_rescaled.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths0[dofrange]); maxy = np.max(nwidths0[dofrange]);
            for key in keys:
                supinf = supinfs0[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            miny0 = np.min([supinfs0[key][0] for key in keys]); maxy0 = np.max([supinfs0[key][0] for key in keys])
            minr = np.min([supinfs0_rates[key] for key in keys]); maxr = np.max([supinfs0_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidths0[dofrange], 'k-', mfc = 'none', label = '$d^0_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs0.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs0_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.max(nwidths0[dofrange]); maxy = np.max(nwidths0[dofrange]);
            for key in keys:
                supinf = supinfs0[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf)
                miny = np.min([ymin, miny])
            miny0 = np.min([supinfs0[key][0] for key in keys]); maxy0 = np.max([supinfs0[key][0] for key in keys])
            minr = np.min([supinfs0_rates[key] for key in keys]); maxr = np.max([supinfs0_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidths0[dofrange], 'k-', mfc = 'none', label = '$d^0_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy0)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs0_rescaled.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs0[key]
                n0 = nwidths0[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/n0, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n0); ymax = np.max(supinf/n0)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda0.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs0[key]
                n2 = nwidths0[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/n2, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n2); ymax = np.max(supinf/n2)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            maxy = np.min([miny*1e2, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda0_rescaled.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths0[dofrange]);
            for key in keys:
                supinf = supinfs1[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            miny0 = np.min([supinfs1[key][0] for key in keys]); maxy0 = np.max([supinfs1[key][0] for key in keys])
            minr = np.min([supinfs1_rates[key] for key in keys]); maxr = np.max([supinfs1_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'k-', mfc = 'none', label = '$d^1_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs1.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs1_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.max(nwidths1[dofrange]); maxy = np.max(nwidths1[dofrange]);
            for key in keys:
                supinf = supinfs1[key]
                handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf)
                miny = np.min([ymin, miny])
            miny0 = np.min([supinfs1[key][0] for key in keys]); maxy0 = np.max([supinfs1[key][0] for key in keys])
            minr = np.min([supinfs1_rates[key] for key in keys]); maxr = np.max([supinfs1_rates[key] for key in keys])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'k-', mfc = 'none', label = '$d^1_{l, n}$'))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy0)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs1_rescaled.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs1[key]
                n1 = nwidths1[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/n1, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n1); ymax = np.max(supinf/n1)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda1.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs1[key]
                n1 = nwidths1[dofs[key]]
                handles.append(*ax.loglog(dofs[key], supinf/n1, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n1); ymax = np.max(supinf/n1)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            maxy = np.min([miny*1e2, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda1_rescaled.pdf')

            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_compute[key]) for key in keys]); ymax = np.max([np.max(times_compute[key]) for key in keys])
            miny0 = np.min([times_compute[key][0] for key in keys]); maxy0 = np.max([times_compute[key][0] for key in keys]); 
            minr = np.min([times_compute_rates[key] for key in keys]); maxr = np.max([times_compute_rates[key] for key in keys]);
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            for key in keys:
                handles.append(*ax.loglog(dofs[key], times_compute[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'\text{compute time}')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_compute.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_compute_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_direct[key]) for key in keys]); ymax = np.max([np.max(times_direct[key]) for key in keys])
            miny0 = np.min([times_direct[key][0] for key in keys]); maxy0 = np.max([times_direct[key][0] for key in keys]); 
            minr = np.min([times_direct_rates[key] for key in keys]); maxr = np.max([times_direct_rates[key] for key in keys]);
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            for key in keys:
                handles.append(*ax.loglog(dofs[key], times_direct[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'assembly, direct solver time')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_direct.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_direct_legend.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_eigen_filtered[key]) for key in keys_filtered]);
            ymax = np.max([np.max(times_eigen_filtered[key]) for key in keys_filtered])
            miny0 = np.min([times_eigen[key][0] for key in keys_filtered]); maxy0 = np.max([times_eigen[key][0] for key in keys_filtered]); 
            minr = np.min([times_eigen_rates[key] for key in keys_filtered])
            maxr = np.max([times_eigen_rates[key] for key in keys_filtered])
            minlog0 = miny0/(min_dof**minr); maxlog0 = (maxy0)/min_dof**maxr
            for key in keys_filtered:
                handles.append(*ax.loglog(dofs[key], times_eigen_filtered[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'eigensolver time')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_eigen.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((len(keys_filtered)+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((len(keys_filtered)+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_eigen_legend.pdf', bbox_extra_artists = (lgd, ))

            plt.close('all')

            mint = np.min([np.min(times_compute[key]) for key in keys]); maxt = np.max([np.max(times_compute[key]) for key in keys])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min([np.min(supinfsE[key]) for key in keys]); maxy = np.max([np.max(supinfsE[key]) for key in keys])
            for key in keys:
                supinf = supinfsE[key]
                rr = supinfsE_rates[key]
                handles.append(*ax.loglog(supinf, times_compute[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, mint, maxt)
            sa_utils.set_log_ticks(ax, miny, maxy, True)
            ax.set_ylabel(r'computation time')
            ax.set_xlabel(r'$\Psi_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_compute_vs_supinfsE.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min([np.min(supinfs0[key]) for key in keys]); maxy = np.max([np.max(supinfs0[key]) for key in keys])
            for key in keys:
                supinf = supinfs0[key]
                handles.append(*ax.loglog(supinf, times_compute[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, mint, maxt)
            sa_utils.set_log_ticks(ax, miny, maxy, True)
            ax.set_ylabel(r'computation time')
            ax.set_xlabel(r'$\Psi_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_compute_vs_supinfs0.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min([np.min(supinfs1[key]) for key in keys]); maxy = np.max([np.max(supinfs1[key]) for key in keys])
            for key in keys:
                supinf = supinfs1[key]
                handles.append(*ax.loglog(supinf, times_compute[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, mint, maxt)
            sa_utils.set_log_ticks(ax, miny, maxy, True)
            ax.set_ylabel(r'computation time')
            ax.set_xlabel(r'$\Psi_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_compute_vs_supinfs1.pdf')

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
                handles.append(*ax.loglog(dofs[key], conditions_stiffness[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label = r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label = '$c n^{'+str("%.2g"%maxr)+r'}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'$\operatorname{cond}(K)$')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_legend.pdf', bbox_extra_artists = (lgd, ))

            plt.close('all')
            logger.info('individual key plots')

            for key in keys:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                miny = np.min([np.min(nwidthsE[dofrange]), np.min(nwidths0[dofrange]), \
                               np.min(supinfsE[key]), np.min(supinfs0[key])])
                maxy = np.max([np.max(nwidthsE[max_dof:max_dof+1]), np.max(nwidths0[max_dof:max_dof+1]), \
                               np.max(supinfsE[key]), np.max(supinfs0[key])])
                handles = []
                handles.append(*ax.loglog(dofrange, nwidthsE[dofrange], 'r-', mfc = 'none', label = '$d^E_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfsE[key], 'ro', mec = 'r', mfc = 'none', label = r'$\Psi_l^E(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfsE[key][0]/dofs[key][0]**supinfsE_rates[key])*dofs[key]**supinfsE_rates[key], 'r:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfsE_rates[key])+r'}$'))
                handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'b-', mfc = 'none', label = '$d^1_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfs1[key], 'bs', mec = 'b', mfc = 'none', label = r'$\Psi_l^1(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfs1[key][0]/dofs[key][0]**supinfs1_rates[key])*dofs[key]**supinfs1_rates[key], 'b:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfs1_rates[key])+r'}$'))
                handles.append(*ax.loglog(dofrange, nwidths0[dofrange], 'g-', mfc = 'none', label = '$d^0_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfs0[key], 'g+', mec = 'g', mfc = 'none', label = r'$\Psi_l^0(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfs0[key][0]/dofs[key][0]**supinfs0_rates[key])*dofs[key]**supinfs0_rates[key], 'g:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfs0_rates[key])+r'}$'))
                sa_utils.set_log_ticks(ax, minx, maxx, True)
                sa_utils.set_log_ticks(ax, miny, maxy)
                ax.set_title(names[key])
                ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
                ax.set_ylabel(r'$\Psi_l^i(\mathcal{V}_l)$')
                ax.set_title(outdir_bc, {'fontsize': 6})
                fig.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs.pdf')
                
                figlegend = plt.figure(figsize = (3.5*sa_utils.legendx, 3*sa_utils.legendy), frameon = False)
                labels = [hh.get_label() for hh in handles]
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_legend.pdf', bbox_extra_artists = (lgd, ))
     
                fig = plt.figure()
                ax = fig.add_subplot(111)
                miny = np.min([np.min(supinfsE[key]), np.min(supinfs1[key]), np.min(supinfs0[key])])
                maxy = np.max([np.max(supinfsE[key]), np.max(supinfs1[key]), np.max(supinfs0[key])])
                handles = []
                handles.append(*ax.loglog(dofrange, nwidthsE[dofrange], 'r-', mfc = 'none', label = r'$d^E_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfsE[key], 'ro', mec = 'r', mfc = 'none', label = r'$\Psi_l^E(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfsE[key][0]/dofs[key][0]**supinfsE_rates[key])*dofs[key]**supinfsE_rates[key], 'r:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfsE_rates[key])+r'}$'))
                handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'b-', mfc = 'none', label = r'$d^1_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfs1[key], 'bs', mec = 'b', mfc = 'none', label = r'$\Psi_l^1(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfs1[key][0]/dofs[key][0]**supinfs1_rates[key])*dofs[key]**supinfs1_rates[key], 'b:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfs1_rates[key])+r'}$'))
                handles.append(*ax.loglog(dofrange, nwidths0[dofrange], 'g-', mfc = 'none', label = r'$d^0_{l, n}$', alpha = 0.7))
                handles.append(*ax.loglog(dofs[key], supinfs0[key], 'g+', mec = 'g', mfc = 'none', label = r'$\Psi_l^0(\mathcal{V}_l)$'))
                handles.append(*ax.loglog(dofs[key], (supinfs0[key][0]/dofs[key][0]**supinfs0_rates[key])*dofs[key]**supinfs0_rates[key], 'g:', alpha = 0.7, label = r'$c n^{'+str("%.2g"%supinfs0_rates[key])+r'}$'))
                sa_utils.set_log_ticks(ax, minx, maxx, True)
                sa_utils.set_log_ticks(ax, miny, maxy)
                ax.set_title(names[key])
                ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
                ax.set_ylabel(r'$\Psi_l^i(\mathcal{V}_l)$')
                ax.set_title(outdir_bc, {'fontsize': 6})
                fig.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_rescaled.pdf')

                fig = plt.figure()
                ax = fig.add_subplot(111)
                handles = []
                miny = np.min([np.min(supinfsE[key]), np.min(supinfs1[key]), np.min(supinfs0[key])])
                maxy = np.max([np.max(supinfsE[key]), np.max(supinfs1[key]), np.max(supinfs0[key])])
                mint = np.min(times_compute[key]); maxt = np.max(times_compute[key])
                ax.loglog(supinfsE[key], times_compute[key], 'ro', mec = 'r', mfc = 'none')
                handles.append(*ax.loglog(supinfsE[key], (times_compute[key][0]/supinfsE[key][0]**(1./supinfsE_compute_rates[key]))*supinfsE[key]**(1./supinfsE_compute_rates[key]), 'r:', alpha = 0.7, label = r'$c \Psi_l^E(\mathcal{V}_l)^{'+str("%.2g"%(1./supinfsE_compute_rates[key]))+r'}$'))
                ax.loglog(supinfs1[key], times_compute[key], 'b+', mec = 'b', mfc = 'none')
                handles.append(*ax.loglog(supinfs1[key], (times_compute[key][0]/supinfs1[key][0]**(1./supinfs1_compute_rates[key]))*supinfs1[key]**(1./supinfs1_compute_rates[key]), 'b:', alpha = 0.7, label = r'$c \Psi_l^1(\mathcal{V}_l)^{'+str("%.2g"%(1./supinfs1_compute_rates[key]))+r'}$'))
                ax.loglog(supinfs0[key], times_compute[key], 'g+', mec = 'g', mfc = 'none')
                handles.append(*ax.loglog(supinfs0[key], (times_compute[key][0]/supinfs0[key][0]**(1./supinfs0_compute_rates[key]))*supinfs0[key]**(1./supinfs0_compute_rates[key]), 'g:', alpha = 0.7, label = r'$c \Psi_l^0(\mathcal{V}_l)^{'+str("%.2g"%(1./supinfs0_compute_rates[key]))+r'}$'))
                sa_utils.set_log_ticks(ax, miny, maxy, True)
                sa_utils.set_log_ticks(ax, mint, maxt)
                ax.set_title(names[key])
                ax.set_xlabel(r'$\Psi_l^i(\mathcal{V}_l)$')
                ax.set_ylabel(r'computation time')
                ax.set_title(outdir_bc, {'fontsize': 6})
                fig.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_compute_vs_supinfs.pdf')

                figlegend = plt.figure(figsize = (3.2*sa_utils.legendx, sa_utils.legendy), frameon = False)
                labels = [hh.get_label() for hh in handles]
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_compute_vs_supinfs_legend.pdf', bbox_extra_artists = (lgd, ))
     
                plt.close('all')

            logger.info('semilog plots')
            minx = min_dof; maxx = max_dof
            doflimits = np.array([min_dof, np.sqrt(min_dof*max_dof), max_dof], dtype = float)
            dofrange = np.arange(min_dof, max_dof+1, dtype = int)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidthsE[dofrange]); maxy = np.max(nwidthsE[dofrange]);
            for key in keys:
                supinf = supinfsE[key]
                rr = supinfsE_rates[key]
                handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            handles.append(*ax.semilogy(dofrange, nwidthsE[dofrange], 'k-', mfc = 'none', label = '$d^E_{l, n}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, 1e-9, 1)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfsE_semilogy.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfsE_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = 0.9; maxy = 2.;
            for key in keys:
                supinf = supinfsE[key]
                nE = nwidthsE[dofs[key]]
                handles.append(*ax.semilogy(dofs[key], supinf/nE, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/nE); ymax = np.max(supinf/nE)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^E(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambdaE_semilogy.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths0[dofrange]); maxy = np.max(nwidths0[dofrange]);
            for key in keys:
                supinf = supinfs0[key]
                handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            handles.append(*ax.semilogy(dofrange, nwidths0[dofrange], 'k-', mfc = 'none', label = '$d^0_{l, n}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, 1e-9, 1)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs0_semilogy.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs0_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs0[key]
                n0 = nwidths0[dofs[key]]
                handles.append(*ax.semilogy(dofs[key], supinf/n0, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n0); ymax = np.max(supinf/n0)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^0(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda0_semilogy.pdf')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths0[dofrange]);
            for key in keys:
                supinf = supinfs1[key]
                handles.append(*ax.semilogy(dofs[key], supinf, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf); ymax = np.max(supinf)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            handles.append(*ax.semilogy(dofrange, nwidths1[dofrange], 'k-', mfc = 'none', label = '$d^1_{l, n}$'))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, 1e-9, 1)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Psi_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_supinfs1_semilogy.pdf')

            figlegend = plt.figure(figsize = (np.ceil((numkeys+3)/legend_rows)*sa_utils.legendx*1.05, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+3)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_supinfs1_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = .9; maxy = 2.;
            for key in keys:
                supinf = supinfs1[key]
                n1 = nwidths1[dofs[key]]
                handles.append(*ax.semilogy(dofs[key], supinf/n1, plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
                ymin = np.min(supinf/n1); ymax = np.max(supinf/n1)
                miny = np.min([ymin, miny]); maxy = np.max([ymax, maxy])
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_ylabel(r'$\Lambda_l^1(\mathcal{V}_l)$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_lambda1_semilogy.pdf')

            plt.close('all')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_compute[key]) for key in keys]); ymax = np.max([np.max(times_compute[key]) for key in keys])
            for key in keys:
                handles.append(*ax.semilogy(dofs[key], times_compute[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'\text{compute time}')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_compute_semilogy.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_compute_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_direct[key]) for key in keys]); ymax = np.max([np.max(times_direct[key]) for key in keys])
            for key in keys:
                handles.append(*ax.semilogy(dofs[key], times_direct[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'assembly, direct solver time')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_direct_semilogy.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_direct_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(times_eigen_filtered[key]) for key in keys_filtered]);
            ymax = np.max([np.max(times_eigen_filtered[key]) for key in keys_filtered])
            for key in keys_filtered:
                handles.append(*ax.semilogy(dofs[key], times_eigen_filtered[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'eigensolver time')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_times_eigen_semilogy.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((len(keys_filtered)+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((len(keys_filtered)+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_times_eigen_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            plt.close('all')

            mint = np.min([np.min(times_compute[key]) for key in keys]); maxt = np.max([np.max(times_compute[key]) for key in keys])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            ymin = np.min([np.min(conditions_stiffness[key]) for key in keys])
            ymax = np.max([np.max(conditions_stiffness[key]) for key in keys])
            for key in keys:
                handles.append(*ax.semilogy(dofs[key], conditions_stiffness[key], plotmarkers[key], mec = plotcolors[key], mfc = 'none', label = names[key]))
            sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
            sa_utils.set_log_ticks(ax, ymin, ymax)
            ax.set_ylabel(r'$\operatorname{cond}(K)$')
            ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
            ax.set_title(outdir_bc, {'fontsize': 6})
            fig.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_semilogy.pdf')

            figlegend = plt.figure(figsize = (1.05*np.ceil((numkeys+2)/legend_rows)*sa_utils.legendx, legend_rows*sa_utils.legendy), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil((numkeys+2)/legend_rows)))
            figlegend.savefig(outdir_bc+'/'+prefix+'_conditions_stiffness_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))

            plt.close('all')

            for key in keys:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                miny = np.min([np.min(nwidthsE[dofrange]), np.min(nwidths0[dofrange]), 
                               np.min(supinfsE[key]), np.min(supinfs0[key])])
                maxy = np.max([np.max(nwidthsE[max_dof:max_dof+1]), np.max(nwidths0[max_dof:max_dof+1]), 
                               np.max(supinfsE[key]), np.max(supinfs0[key])])
                handles = []
                handles.append(*ax.semilogy(dofrange, nwidthsE[dofrange], 'r-', mfc = 'none', label = '$d^E_{l, n}$', alpha = 0.7))
                handles.append(*ax.semilogy(dofs[key], supinfsE[key], 'ro', mec = 'r', mfc = 'none', label = r'$\Psi_l^E(\mathcal{V}_l)$'))
                handles.append(*ax.semilogy(dofrange, nwidths1[dofrange], 'b-', mfc = 'none', label = '$d^1_{l, n}$', alpha = 0.7))
                handles.append(*ax.semilogy(dofs[key], supinfs1[key], 'bs', mec = 'b', mfc = 'none', label = r'$\Psi_l^1(\mathcal{V}_l)$'))
                handles.append(*ax.semilogy(dofrange, nwidths0[dofrange], 'g-', mfc = 'none', label = '$d^0_{l, n}$', alpha = 0.7))
                handles.append(*ax.semilogy(dofs[key], supinfs0[key], 'g+', mec = 'g', mfc = 'none', label = r'$\Psi_l^0(\mathcal{V}_l)$'))
                sa_utils.set_log_ticks(ax, minx, maxx, xaxis = True, semilog = True)
                sa_utils.set_log_ticks(ax, miny, maxy)
                ax.set_title(names[key])
                ax.set_xlabel(r'$n = \operatorname{dim} \mathcal{V}_l$')
                ax.set_ylabel(r'$\Psi_l^i(\mathcal{V}_l)$')
                ax.set_title(outdir_bc, {'fontsize': 6})
                fig.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_semilogy.pdf')
                
                figlegend = plt.figure(figsize = (3.5*sa_utils.legendx, 2*sa_utils.legendy), frameon = False)
                labels = [hh.get_label() for hh in handles]
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(outdir_bc+'/'+prefix+'_'+str(key)+'_supinfs_legend_semilogy.pdf', bbox_extra_artists = (lgd, ))
     
                plt.close('all')
        
        del fig, ax, figlegend, handles, labels, lgd
        logger.info('matplotlib done')
        logger.info('bc case [{:d}/{:d}]: {:s} END'.format(bc_ii+1, num_cases, bc_suffix))

    del extended_facet_dirichlet
    del extended_facet_neumann
    logger.info('localizations_nitsche patch END')
    logger.info('''
    dofmax                          {:d}
    contrast                        {:.2e}-{:.2e}={:s}
    beta                            {:.2e}
    gamma                           {:.2e}
    elasticity                      {:s}
    orthotropic                     {:s}
    ldomain                         {:s}
    compute supinfs                 {:s}
    matplotlib                      {:s}
    compute particular solutions    {:s}
    krylov                          {:s}
    debug                           {:s}
Computing enrichments and supinfs END
'''.format(dofmax, cmin, cmax, str(cmax/cmin) if cmin > 0 else 'inf', beta, gamma, str(elasticity), str(orthotropic), str(ldomain), str(compute_supinfs), str(matplotlib), str(compute_particular_solutions), str(krylov), str(debug)))

def patch_script(meshdir, prefix, level, patch_level, outdir, dofmax = 100, write_shapes = False, patches = None, 
                 logger = None, minimal = 1, ortho = True, heat = True, elasticity = True, orthotropic = False,
                 spe = False, contrast = 1e4, beta = 2.0, ldomain = False, 
                 max_procs = 1, patch_procs = None, max_keys = None, batch_procs = None, 
                 batch_logging = False, debug = False, values = None,
                 kappa_is_EE = True, kappa_is_poisson = False, EE = 1e11, poisson = 0.3,
                 hole_only_neumann = True, only_lr_dirichlet = False,
                 krylov = False, krylov_neumann = False, krylov_harmonic = False,
                 lagrange_neumann_harmonic = True, normalize_null_orthogonalize = False,
                 noz = False, orthogonalize_hats = True, interior_orth_matrix = False):
    prefix_level = prefix+'_'+str(level)
    logger.info('Starting patch_script ['+prefix_level+', '+str(patch_level)+']')
    patch_dir = meshdir+'/'+prefix_level+'_patches'

    patch_boxes = np.genfromtxt(meshdir+'/'+prefix+'_patch_descriptors/'+str(patch_level)+'.csv', delimiter = ', ', names = True)
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

    if orthotropic:
        elasticity = True

    def compute_fun(ii, elasticity, qq):
        patch_name = str(patch_level)+'/patch_'+str(ii).zfill(patch_fill)

        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] started')
        basename = patch_dir+'/'+patch_name
        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] loading ['+basename+']')
        if spe:
            extended_mesh, extended_domains, extended_coeff, extended_facets = spe.load_patch(basename)
        else:
            extended_mesh, extended_domains, extended_coeff, extended_facets = create_patches.load_patch_h5py(basename, contrast = contrast, beta = beta, values = values, debug = debug)
        basedim = extended_mesh.geometry().dim()

        basename = prefix+'_'+str(level)+'_'+str(patch_level)+'_patch_'+str(ii).zfill(patch_fill)
        localization_test(extended_mesh, extended_domains, extended_coeff, extended_facets, 
                          dofmax = dofmax*(basedim if elasticity else 1), elasticity = elasticity, write_shapes = write_shapes, prefix = basename, matplotlib = True, 
                          outdir = '{:s}/{:s}/contrast_{:.2e}/patchlevel_{:s}/res_{:d}/beta_{:.2e}'.format(outdir, ('orthotropic' if orthotropic else 'elasticity') if elasticity else 'heat', contrast, patch_name, level, beta), 
                          logger = logger, minimal = minimal, ortho = ortho, spe = spe, max_procs = patch_procs, batch_procs = batch_procs, max_keys = max_keys, 
                          batch_logging = batch_logging, debug = debug, ldomain = ldomain, kappa_is_EE = kappa_is_EE, kappa_is_poisson = kappa_is_poisson, EE = EE, poisson = poisson,
                          hole_only_neumann = hole_only_neumann, only_lr_dirichlet = only_lr_dirichlet, krylov = krylov, krylov_harmonic = krylov_harmonic, krylov_neumann = krylov_neumann, 
                          lagrange_neumann_harmonic = lagrange_neumann_harmonic, normalize_null_orthogonalize = normalize_null_orthogonalize,
                          noz = noz, orthotropic = orthotropic, orthogonalize_hats = orthogonalize_hats, interior_orth_matrix = interior_orth_matrix)
        qq.put(None)
        logger.info('patch ['+str(ii+1)+'/'+str(patch_num)+'] finished')

    block = multiprocessing.cpu_count() if max_procs is None else np.min([max_procs, multiprocessing.cpu_count()])

    start = sa_utils.get_time()
    if elasticity:
        logger.info('Starting Elasticity')
        queue = multiprocessing.Queue()
        if block > 1:
            compute_processes = []
            for ii in args:
                compute_processes.append(multiprocessing.Process(target = compute_fun, args = (ii, True, queue)))
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

    if heat:
        logger.info('Starting Heat')
        queue = multiprocessing.Queue()
        if block > 1:
            compute_processes = []
            for ii in args:
                compute_processes.append(multiprocessing.Process(target = compute_fun, args = (ii, False, queue)))
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
    end = sa_utils.get_time()
    logger.info('Finished patch script ['+prefix_level+', '+str(patch_level)+'] in ['+sa_utils.human_time(end-start)+']')
