from dolfin import *
import time

# whole domain
basedim = 3
res = 30
mesh = UnitCubeMesh(res, res, res)
print('{:d} dofs total'.format(mesh.num_vertices()))
eps = 1e-10

# cell coefficient
cell_function = MeshFunction('double', mesh, basedim, 1)
subdiag = AutoSubDomain(lambda xx, on: xx[1] < xx[0])
subdiag.mark(cell_function, 100)

# facet markers
top = AutoSubDomain(lambda xx, on: on and near(xx[1], 1, eps=eps))
right = AutoSubDomain(lambda xx, on: on and near(xx[0], 1, eps=eps))
facet_function = MeshFunction('size_t', mesh, basedim-1, 0)
top.mark(facet_function, 1)
right.mark(facet_function, 2)

File('cell_function.pvd') << cell_function
File('facet_function.pvd') << facet_function

# subdomain
print('subdomain')
tic = time.process_time()
sub_domain = AutoSubDomain(lambda xx, on: xx[0] > 0.5-eps and xx[1] > 0.5-eps)
sub_marker = MeshFunction('size_t', mesh, basedim, 0)
sub_domain.mark(sub_marker, 1)
sub_mesh = SubMesh(mesh, sub_marker, 1)
print('{:d} dofs submesh'.format(sub_mesh.num_vertices()))
toc = time.process_time()
print('    {:.2e}s'.format(toc-tic))

# subdomain cell function
print('subdomain cell function')
tic = time.process_time()
sub_mesh_cmap = sub_mesh.data().array('parent_cell_indices', basedim)
sub_cell_function = MeshFunction('double', sub_mesh, basedim, 0)
sub_cell_function.set_values(cell_function.array()[sub_mesh_cmap])
toc = time.process_time()
print('    {:.2e}s'.format(toc-tic))
File('sub_cell_function.pvd') << sub_cell_function

# subdomain facet function
print('subdomain facet function')
tic = time.process_time()
sub_facet_function = MeshFunction('size_t', sub_mesh, basedim-1, 0)
boundary_mesh = BoundaryMesh(mesh, 'exterior')
boundary_mesh_cmap = boundary_mesh.entity_map(basedim-1)
sub_boundary_marker = MeshFunction('size_t', boundary_mesh, basedim-1, 0)
sub_domain_min = sub_mesh.coordinates().min(0)
sub_domain_max = sub_mesh.coordinates().max(0)
sub_domain_bounding = AutoSubDomain(lambda xx, on: (sub_domain_min-eps <= xx).all() and (xx <= sub_domain_max+eps).all())
sub_domain_bounding.mark(sub_boundary_marker, 1)
sub_boundary_mesh = SubMesh(boundary_mesh, sub_boundary_marker, 1)
print('{:d} cells on boundary mesh restriction'.format(sub_boundary_mesh.num_cells()))
File('sub_boundary_mesh.pvd') << sub_boundary_mesh
sub_boundary_mesh_cmap = sub_boundary_mesh.data().array('parent_cell_indices', basedim-1)

boundary_sub_mesh = BoundaryMesh(sub_mesh, 'exterior')
print('{:d} cells on submesh boundary'.format(boundary_sub_mesh.num_cells()))
boundary_sub_mesh_cmap = boundary_sub_mesh.entity_map(basedim-1)

sub_mesh_vmap = sub_mesh.data().array('parent_vertex_indices', 0)
print('slow loop')
tuc = time.process_time()
for boundary_sub_mesh_cell in cells(boundary_sub_mesh):
    sub_mesh_facet = Facet(sub_mesh, boundary_sub_mesh_cmap[boundary_sub_mesh_cell.index()])
    sub_mesh_facet_vertex_set = set(sub_mesh_vmap[sub_mesh_facet.entities(0)])
    for sub_boundary_mesh_cell in cells(sub_boundary_mesh):
        mesh_facet = Facet(mesh, boundary_mesh_cmap[sub_boundary_mesh_cmap[sub_boundary_mesh_cell.index()]])
        if set(mesh_facet.entities(0)) == sub_mesh_facet_vertex_set:
            sub_facet_function[sub_mesh_facet.index()] = facet_function[mesh_facet.index()]
            break
toc = time.process_time()
print('    {:.2e}s total, {:.2e}s loop'.format(toc-tic, toc-tuc))
            
File('sub_facet_function.pvd') << sub_facet_function