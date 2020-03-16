from dolfin import *
from mshr import *

import_matplotlib = False

import sa_utils
import sa_hdf5

import numpy as np
import sympy as smp
import gc
import datetime
comm = MPI.comm_world

eps = 1e-10

now = datetime.datetime.now()
logger = sa_utils.LogWrapper('fenics_pum_{:s}'.format(now.strftime("%Y-%m-%d_%H:%M")), stdout=True)

def sympy_get_monomials(dimension, degree, vector = False):
    xs = smp.symbols(['x[{:d}]'.format(ii) for ii in range(dimension)], real=True)
    monos = smp.polys.monomials.itermonomials(xs, degree)
    for xx in xs:
        monos = [smp.horner(mm, wrt=xx) for mm in monos]
    monos = sorted(monos, key=smp.polys.orderings.monomial_key('grlex', xs[::-1]))[1:]
    if vector:
        expressions = [Constant(
            [0 for ll in range(kk)]+
            [1] +
            [0 for ll in range(kk+1, dimension)]
        ) for kk in range(dimension)]
        expressions += [Expression(
            ['0' for ll in range(kk)]+
            [smp.ccode(mm)]+
            ['0' for ll in range(kk+1, dimension)],
            degree=int(smp.degree(mm))) for mm in monos for kk in range(dimension)
        ]
    else:
        expressions = [Constant(1)]
        expressions += [
            Expression(smp.ccode(mm), degree=int(smp.degree(mm))) for mm in monos
        ]
    return expressions

def get_interpolants(spaces, expressions):
    return [[interpolate(exp, space) for exp in expressions] for space in spaces]

def distance_points_box(points, low, high):
    dd = np.max([low-points, np.zeros_like(points), points-high], 0)
    return np.sqrt(np.sum(dd*dd, 1))

def inside_points_box(points, low, high):
    return (low-eps < points).all(1)*(points < high+eps).all(1)
    
def construct_pu(
        mesh, pum_low, pum_high, *,
        pum_res = 2, pum_alpha = .25, oversampling = [2.0],
        cutoff = eps
):
    basedim = mesh.geometric_dimension()
    num_oversampling = len(oversampling)
    
    logger.info("Constructing PU")
    pum_diff = pum_high-pum_low
    pum_hh = np.max(pum_diff)*2.**(-pum_res)
    pum_max = np.array(np.ceil(pum_diff/pum_hh), dtype=int)

    logger.info("Constructing flat top regions")
    patches_flattop = []
    patches_oversampling = [[]  for ii in range(num_oversampling)]
    for kk in range(pum_max[2]):
        zz_low = pum_low[2]+(kk+0.5)*pum_hh
        for jj in range(pum_max[1]):
            yy_low = pum_low[1]+(jj+0.5)*pum_hh
            for ii in range(pum_max[0]):
                xx_low = pum_low[0]+(ii+0.5)*pum_hh
                pt = np.array([xx_low, yy_low, zz_low])
                patches_flattop.append([pt-pum_hh*(1.-pum_alpha)*0.5, pt+pum_hh*(1.-pum_alpha)*0.5])
                for ll, oo in enumerate(oversampling):
                    patches_oversampling[ll].append([pt-pum_hh*(1.+pum_alpha)*oo*0.5, pt+pum_hh*(1.+pum_alpha)*oo*0.5])
    patches_flattop = np.array(patches_flattop)
    num_patches = len(patches_flattop)
        
    pum_alpha_hh = pum_hh*pum_alpha
    mesh_vertices = mesh.coordinates()
    num_vertices = mesh.num_vertices()
    vertices_done = np.zeros(num_vertices, dtype=int)
    patches_ok = []
    vertices_ok = []
    for ii, patch in enumerate(patches_flattop):
        res = inside_points_box(mesh_vertices, *patch)
        if res.any():
            patches_ok.append(ii)
            vertices_done += res
            vertices_ok.append(np.where(res > 0)[0])
    vertices_undone = np.where(vertices_done < 1)[0]
    patches_ok = np.array(patches_ok, dtype=int)
    patches_flattop = patches_flattop[patches_ok]
    for ll in range(num_oversampling):
        patches_oversampling[ll] = np.array(patches_oversampling[ll])[patches_ok]
    logger.info("[{:d}/{:d}] patches left".format(len(patches_flattop), num_patches))
    num_patches = len(patches_flattop)
    
    coeffs = np.zeros((num_patches, num_vertices))
    for ii, patch in enumerate(patches_flattop):
        vs = vertices_ok[ii]
        ci = coeffs[ii]
        ci[vertices_undone] = distance_points_box(mesh_vertices[vertices_undone], *patch)
        ci[np.where(ci > pum_alpha_hh)] = 0
        ci[np.where(ci < cutoff)] = 0
        ww = np.where(ci > 0)
        ci[ww] = (pum_alpha_hh-ci[ww])/pum_alpha_hh
        coeffs[ii, vs] = 1.

    coeffs_sum = sum(coeffs, 0)
    coeffs /= coeffs_sum
        
    VV = FunctionSpace(mesh, 'CG', 1)
    pu = []
    mm = dof_to_vertex_map(VV)
    for ii, patch in enumerate(patches_flattop):
        pu.append(Function(VV, name='u'))
        pu[-1].vector().set_local(coeffs[ii][mm])

    mesh_cells = mesh.cells()
    markers = [[] for ii in range(num_oversampling)]
    for ll, pts in enumerate(patches_oversampling):
        for pt in pts:
            inside_vertices = inside_points_box(mesh_vertices, *pt)
            mf = MeshFunction('size_t', mesh, basedim, 0)
            mf.set_values(np.apply_along_axis(lambda cc: inside_vertices[cc].any(), 1, mesh_cells))
            markers[ll].append(mf)
    return pu, markers

def make_subdomains(
    mesh, pu, markers, *,
    cell_function = None, facet_function = None,
    vector = False, cutoff = eps, debug = False
):
    logger.info('make_subdomains, [{:d}] patches, mesh [{:d}] dof, start'.format(len(pu), mesh.num_vertices()));
    assert(len(pu) == len(markers))
    basedim = mesh.geometric_dimension()
    num = len(pu)
    mesh_scalar_VV = pu[0].function_space()
    mesh_vertex_to_scalar_dof_map = vertex_to_dof_map(mesh_scalar_VV)
    if vector:
        mesh_VV = VectorFunctionSpace(mesh, 'CG', 1)
        mesh_vertex_to_dof_map = vertex_to_dof_map(mesh_VV)
        if debug:
            debug_uu = interpolate(Expression(['x[0]*x[1]']+['0' for ii in range(basedim-1)], degree=1), mesh_VV)
    elif debug:
        debug_uu = interpolate(Expression('x[0]*x[1]', degree=1), mesh_scalar_VV)

    oversampled_submeshes = []
    oversampled_submeshes_VV = []
    subsubmesh_markers = []
    local_to_global_data = []
    if cell_function is not None:
        submesh_cell_functions = []
    if facet_function is not None:
        submesh_facet_functions = []
        bmesh = BoundaryMesh(mesh, 'exterior')
        bmesh_map = bmesh.entity_map(basedim-1)
        exterior_subdomain = AutoSubDomain(lambda xx, on: on)
    count = 0
    for pp, mk in zip(pu, markers):
        logger.info('    patch [{:d}/{:d}] start'.format(count+1, num))
        logger.info('        creating submesh')
        coeff = pp.vector()
    
        submesh = SubMesh(mesh, mk, 1)
        oversampled_submeshes.append(submesh)
        logger.info('        created submesh')
        logger.info('        creating maps')
        vertex_parent_map = submesh.data().array("parent_vertex_indices", 0)
        
        mf = MeshFunction('size_t', submesh, basedim, 0)
        mf.set_values(np.apply_along_axis(lambda cc: (coeff[mesh_vertex_to_scalar_dof_map[vertex_parent_map[cc]]] > 0).any(), 1, submesh.cells()))
        subsubmesh_markers.append(mf)
        
        submesh_scalar_VV = FunctionSpace(submesh, 'CG', 1)
        submesh_scalar_dof_to_vertex_map = dof_to_vertex_map(submesh_scalar_VV)
        submesh_scalar_dof_to_mesh_scalar_dof_map = mesh_vertex_to_scalar_dof_map[vertex_parent_map[submesh_scalar_dof_to_vertex_map]]
        submesh_scalar_nonzero = np.where(coeff[submesh_scalar_dof_to_mesh_scalar_dof_map] > cutoff)[0]
        
        if not vector:
            submesh_VV = submesh_scalar_VV
            submesh_dof_to_mesh_dof_map = submesh_scalar_dof_to_mesh_scalar_dof_map
            submesh_nonzero = submesh_scalar_nonzero
            submesh_nonzero_pu = coeff[submesh_dof_to_mesh_dof_map[submesh_nonzero]]
        else:
            submesh_VV = VectorFunctionSpace(submesh, 'CG', 1)
            submesh_dof_to_vertex_map = dof_to_vertex_map(submesh_VV)
            submesh_dof_to_vertex_map_index = submesh_dof_to_vertex_map//basedim
            submesh_dof_to_vertex_map_offset = submesh_dof_to_vertex_map%basedim
            submesh_dof_to_mesh_dof_map = mesh_vertex_to_dof_map[basedim*vertex_parent_map[submesh_dof_to_vertex_map_index]+submesh_dof_to_vertex_map_offset]
            submesh_nonzero = np.where(coeff[mesh_vertex_to_scalar_dof_map[vertex_parent_map[submesh_dof_to_vertex_map_index]]] > cutoff)[0]
            submesh_nonzero_pu = coeff[mesh_vertex_to_scalar_dof_map[vertex_parent_map[submesh_dof_to_vertex_map_index[submesh_nonzero]]]]
        oversampled_submeshes_VV.append(submesh_VV)
        local_to_global_data.append((submesh_dof_to_mesh_dof_map, submesh_nonzero, submesh_nonzero_pu))
        logger.info('        created maps')
        
        if cell_function is not None:
            logger.info('        copying cell function to submesh')
            cell_parent_map = submesh.data().array("parent_cell_indices", basedim)
            sub_V0 = FunctionSpace(submesh, 'DG', 0)
            cf = Function(sub_V0)
            cf.vector().set_local(cell_function.vector()[cell_parent_map])
            submesh_cell_functions.append(cf)
            logger.info('        copyied cell function to submesh')
        else:
            cf = None
        
        if facet_function is not None:
            logger.info('        copying facet function to submesh')
            ff = MeshFunction('size_t', submesh, basedim-1, 0)
            bsubmesh = BoundaryMesh(submesh, 'exterior')
            bsubmesh_map = bsubmesh.entity_map(basedim-1)
            logger.info('        submesh exterior')
            exterior_subdomain.mark(ff, 100)
            
            logger.info('        boundary mesh submesh')
            submesh_min = submesh.coordinates().min(0)
            submesh_max = submesh.coordinates().max(0)
            submesh_bounding = AutoSubDomain(lambda xx, on: (submesh_min-eps <= xx).all() and (xx <= submesh_max+eps).all())
            subbmarker = MeshFunction('size_t', bmesh, basedim-1, 0)
            submesh_bounding.mark(subbmarker, 1)
            subbmesh = SubMesh(bmesh, subbmarker, 1)
            subbmesh_map = subbmesh.data().array("parent_cell_indices", basedim-1)
            
            logger.info('        slow loop')
            for sub_cell in cells(bsubmesh):
                sub_facet = Facet(submesh, bsubmesh_map[sub_cell.index()])
                sub_facet_vertices_set = set(vertex_parent_map[sub_facet.entities(0)])
                for cell in cells(subbmesh):
                    facet = Facet(mesh, bmesh_map[subbmesh_map[cell.index()]])
                    if set(facet.entities(0)) == sub_facet_vertices_set:
                        ff[sub_facet.index()] = facet_function[facet.index()]
                        break
            submesh_facet_functions.append(ff)
            logger.info('        copyied facet function to submesh')
        else:
            ff = None
        
        if debug:
            logger.info('        writing debugging output')
            tmp_mf = MeshFunction('double', submesh, basedim, 0)
            tmp_mf.set_values(cf.vector())
            sa_hdf5.write_dolfin_mesh(submesh, 'submesh_{:d}_restrictions'.format(count), cell_function = tmp_mf, facet_function = ff)
            sa_hdf5.write_dolfin_mesh_functions(submesh, 'submesh_{:d}_markers'.format(count), cell_functions = [subsubmesh_markers[-1]])
            uu = Function(submesh_scalar_VV)
            uu.vector()[submesh_scalar_nonzero] = coeff[submesh_scalar_dof_to_mesh_scalar_dof_map[submesh_scalar_nonzero]]
            sa_hdf5.write_dolfin_scalar_cg1('submesh_{:d}_pu'.format(count), [uu])
            vv = Function(submesh_VV)
            vv.vector()[submesh_nonzero] = debug_uu.vector()[submesh_dof_to_mesh_dof_map[submesh_nonzero]]
            vv.vector()[submesh_nonzero] *= submesh_nonzero_pu
            if vector:
                sa_hdf5.write_dolfin_vector_cg1('submesh_{:d}_vector'.format(count), [vv])
            else:
                sa_hdf5.write_dolfin_scalar_cg1('submesh_{:d}_scalar'.format(count), [vv])
            logger.info('        written debugging output')
        logger.info('    patch [{:d}/{:d}] end'.format(count+1, num))
        count += 1
    ret = dict()
    ret['meshes'] = oversampled_submeshes
    ret['spaces'] = oversampled_submeshes_VV
    ret['markers'] = subsubmesh_markers
    ret['local_to_global'] = local_to_global_data
    if cell_function is not None:
        ret['cell_functions'] = submesh_cell_functions
    if facet_function is not None:
        ret['facet_functions'] = submesh_facet_functions
    logger.info('make_subdomains, [{:d}] patches, mesh [{:d}] dof, end'.format(len(pu), mesh.num_vertices()));
    return ret

def local_to_global(VV, functions, local_to_global_data):
    ret = []
    for fs, (idx_map, idx, pus) in zip(functions, local_to_global_data):
        for ff in fs:
            uu = Function(VV)
            uu.vector()[idx_map[idx]] = ff.vector()[idx]*pus
            ret.append(uu)
    return ret

def cell_function_to_dg(ff, V0 = None):
    if V0 is None:
        V0 = FunctionSpace(ff.mesh(), 'DG', 0)
    v0 = Function(V0)
    v0.vector().set_local(ff.array())
    return v0
    
def create_oht_example():
    res = 2**6
    layers = 8
    layer_zz = 0.0074
    dims = np.array([6., 3., layers*layer_zz])
    basename = 'oht'

    box = Box(Point(*(-dims/2)), Point(*(dims/2)))
    hole = Cylinder(Point(0,0,-1), Point(0,0,1), 0.125, 0.125, int(res))
    domain = box-hole

    mesh = generate_mesh(domain, res)

    cell_coeff = MeshFunction('double', mesh, 3, 1)
    for cell in cells(mesh):
        if cell.midpoint().z() < layer_zz:
            cell_coeff[cell] = 10

    facet_function = MeshFunction('size_t', mesh, 2, 0)
    bc_dict = dict()
    left = AutoSubDomain(lambda xx, on: on and near(xx[0], -0.5*dims[0], eps = eps))
    bc_dict[1] = left
    right = AutoSubDomain(lambda xx, on: on and near(xx[0], 0.5*dims[0], eps = eps))
    bc_dict[2] = right
    front = AutoSubDomain(lambda xx, on: on and near(xx[1], -0.5*dims[1], eps = eps))
    bc_dict[3] = front
    back = AutoSubDomain(lambda xx, on: on and near(xx[1], 0.5*dims[1], eps = eps))
    bc_dict[4] = back
    bottom = AutoSubDomain(lambda xx, on: on and near(xx[2], -0.5*dims[2], eps = eps))
    bc_dict[5] = bottom
    top = AutoSubDomain(lambda xx, on: on and near(xx[2], 0.5*dims[2], eps = eps))
    bc_dict[6] = top
    border = AutoSubDomain(lambda xx, on: on)
    border.mark(facet_function, 100)
    for key in bc_dict:
        bc_dict[key].mark(facet_function, key)

    sa_hdf5.write_dolfin_mesh(mesh, '{:s}'.format(basename), cell_function = cell_coeff, facet_function = facet_function)
    return mesh, cell_coeff, facet_function, basename


if __name__ == '__main__':
    polydegree = 1
    create = True
    debug = True
    vector = True
    if vector:
        Space = VectorFunctionSpace
        write_functions = sa_hdf5.write_dolfin_vector_cg1
    else:
        Space = FunctionSpace
        write_functions = sa_hdf5.write_dolfin_scalar_cg1
    
    if create:
        mesh, cell_coeff, facets, basename = create_oht_example()
        
        pum_low = np.array([-4.5,-1.5,-1.5])
        pum_high = np.array([7.5,1.5,1.5])
        pum_res = 2
        pum_alpha = 0.25

        oversampling = [1.5]    
        pu, markers = construct_pu(
            mesh, pum_low, pum_high,
            pum_res = pum_res, pum_alpha = pum_alpha, oversampling = oversampling
        )
        
        logger.info('writing')
        sa_hdf5.write_dolfin_scalar_cg1('{:s}_pu'.format(basename), pu)
        for ii, oo in enumerate(oversampling):
            sa_hdf5.write_dolfin_mesh_functions(mesh, '{:s}_{:.2e}'.format(basename, oo), cell_functions = markers[ii])

        del mesh, cell_coeff, facets, pu, markers

    logger.info('reloading')
    read1 = sa_hdf5.read_dolfin_mesh('{:s}'.format(basename))
    mesh = read1['mesh']
    basedim = mesh.geometric_dimension()
    cell_function = cell_function_to_dg(read1['cell_function'])
    facet_function = read1['facet_function']
    del read1
    read2 = sa_hdf5.read_dolfin_mesh_functions('{:s}_{:.2e}'.format(basename, oversampling[-1]))
    markers = read2['cell_functions']
    del read2
    pu = sa_hdf5.read_dolfin_scalar_cg1('{:s}_pu'.format(basename)) 
    
    subdomain_data = make_subdomains(
        mesh, pu, markers,
        cell_function = cell_function, facet_function = facet_function,
        debug = debug, vector = vector
    )
    
    poly_exprs = sympy_get_monomials(basedim, polydegree, vector)
    sub_polys = get_interpolants(subdomain_data['spaces'], poly_exprs)
    if debug:
        for ii, ps in enumerate(sub_polys):
            write_functions('{:s}_patch_{:d}_polys'.format(basename, ii), ps)
    global_VV = Space(mesh, 'CG', 1)
    global_polys = local_to_global(global_VV, sub_polys, subdomain_data['local_to_global'])
    if debug:
        write_functions('{:s}_polys'.format(basename), global_polys)
    
    