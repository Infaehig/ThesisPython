import h5py

from dolfin import *
from dolfin_utils.meshconvert import meshconvert
from subprocess import call
import os, sys, getopt

import numpy as np
import numpy.random as rnd

import sa_utils
from sa_utils import comm

my_compression=None
#my_compression=my_compression

def to_szip(oldname, newname):
    old_file = h5py.File(oldname+'.h5', 'r')
    new_file = h5py.File(newname+'.h5', 'w')

    def copy_recursive(path):
        grp = old_file[path]
        for key in grp.keys():
            dt = grp[key]
            newpath = '{:s}/{:s}'.format(path, key)
            if isinstance(dt, h5py.Dataset):
                new_file.create_dataset(newpath, dt.shape, dt[:], compression = my_compression)
            else:
                new_file.create_group(newpath)
                copy_recursive(newpath)
    copy_recursive('/')
    old_file.close()
    new_file.close()
    del old_file, new_file


def write_dolfin_mesh(mesh, basename, cell_function = None, cell_coeff = None, facet_function = None, transform = None):
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
    h5_file.create_dataset('/cells', (num_cells, basedim+1), data = np.array(mesh.cells(), dtype = np.uintp), compression = my_compression)
    h5_file.create_dataset('/facets', (num_facets, basedim), data = np.array([facet.entities(0) for facet in facets(mesh)], dtype = np.uintp), compression = my_compression)
    if cell_function is not None:
        h5_file.create_dataset('/cell_function', (num_cells, 1), data = cell_function.array(), compression = my_compression)
    if cell_coeff is not None:
        h5_file.create_dataset('/cell_coeff', (num_cells, 1), data = cell_coeff.array(), compression = my_compression)
    if facet_function is not None:
        h5_file.create_dataset('/facet_function', (num_facets, 1), data = facet_function.array(), compression = my_compression)
    h5_file.close()
    del h5_file

    cells_string = r''

    if cell_function is not None:
        cells_string += r"""
            <Attribute Name = "cell_function" AttributeType = "Scalar" Center = "Cell">
                <DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/cell_function</DataItem>
            </Attribute>""".format(num_cells, real_basename)

    if cell_coeff is not None:
        cells_string += r"""
            <Attribute Name = "cell_coeff" AttributeType = "Scalar" Center = "Cell">
                <DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/cell_coeff</DataItem>
            </Attribute>""".format(num_cells, real_basename)

    cell_file = open(basename+'_cells.xdmf', 'w')
    cell_file.write(r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "cells" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>{:s}
        </Grid>
    </Domain>
</Xdmf>""".format(num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
                  num_cells, basedim+1, real_basename, 
                  "XYZ" if basedim == 3 else "XY", 
                  num_vertices, basedim, real_basename, 
                  cells_string))
    cell_file.close()
    del cell_file

    if facet_function is None:
        facets_string = r''
    else:
        facets_string = r"""
            <Attribute Name = "facet_function" AttributeType = "Scalar" Center = "Cell">
                <DataItem Format = "HDF" Dimensions = "{:d} 1">{:s}.h5:/facet_function</DataItem>
            </Attribute>""".format(num_facets, real_basename)

    facet_file = open(basename+'_facets.xdmf', 'w')
    facet_file.write(r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "facets" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/facets</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>{:s}
        </Grid>
    </Domain>
</Xdmf>""".format(num_facets, 'Triangle' if basedim == 3 else 'PolyLine" NodesPerElement = "2', 
                  num_facets, basedim, real_basename, 
                  "XYZ" if basedim == 3 else "XY", 
                  num_vertices, basedim, real_basename, 
                  facets_string))
    facet_file.close()
    del facet_file

    print('written ['+basename+'] mesh and markers')

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
    h5_cells = h5_file.create_dataset('/cells', (num_cells, basedim+1), data = np.array(mesh.cells(), dtype = np.uintp), compression = my_compression)
    h5_grp = h5_file.create_group('/scalar')
    num = len(scalar_functions)
    index_map = vertex_to_dof_map(scalar_functions[0].function_space())

    if scale_to_one:
        for ii, uu in enumerate(scalar_functions):
            data = uu.vector()[index_map]
            umin = np.min(data); umax = np.max(data)
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
    string = r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "cells" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>
        </Grid>""".format(
        num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
        num_cells, basedim+1, real_basename, 
        "XYZ" if basedim == 3 else "XY", 
        num_vertices, basedim, real_basename
    )
    string += """
        <Grid Name = "scalar" GridType = "Collection" CollectionType = "Temporal">
            <Time TimeType = "List">
                <DataItem Format = "XML" Dimensions = "{:d}">
                    """.format(num)
    for ii in range(num):
        string += " {:d}".format(ii)
    string += r"""
                </DataItem>
            </Time>"""
    for ii in range(num):
        string += r"""
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
            </Grid>""".format(ii, 
                              num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
                              num_cells, basedim+1, real_basename, 
                              "XYZ" if basedim == 3 else "XY", 
                              num_vertices, basedim, real_basename, 
                              num_vertices, real_basename, ii)
    string += r"""
        </Grid>
    </Domain>
</Xdmf>"""
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
    h5_cells = h5_file.create_dataset('/cells', (num_cells, basedim+1), data = np.array(mesh.cells(), dtype = np.uintp), compression = my_compression)
    h5_grp = h5_file.create_group('/vector')
    num = len(vector_functions)
    index_map = vertex_to_dof_map(vector_functions[0].function_space())
    if scale_to_one:
        for ii, uu in enumerate(vector_functions):
            data = uu.vector()[index_map]
            umin = np.min(data); umax = np.max(data)
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
    string = r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "cells" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>
        </Grid>""".format(
        num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
        num_cells, basedim+1, real_basename, 
        "XYZ" if basedim == 3 else "XY", 
        num_vertices, basedim, real_basename
    )
    string += """
        <Grid Name = "vector" GridType = "Collection" CollectionType = "Temporal">
            <Time TimeType = "List">
                <DataItem Format = "XML" Dimensions = "{:d}">
                    """.format(num)
    for ii in range(num):
        string += " {:d}".format(ii)
    string += r"""
                </DataItem>
            </Time>"""
    for ii in range(num):
        string += r"""
            <Grid Name = "grid_{:d}" GridType = "Uniform">
                <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                    <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
                </Topology>
                <Geometry GeometryType = "{:s}">
                    <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
                </Geometry>""".format(ii, 
                                      num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
                                      num_cells, basedim+1, real_basename, 
                                      "XYZ" if basedim == 3 else "XY", 
                                      num_vertices, basedim, real_basename)
        for jj in range(basedim):
            string += r"""
                <Attribute Name = "u_{:d}" AttributeType = "Scalar" Center = "Node">
                    <DataItem ItemType = "HyperSlab" Dimensions = "{:d}">
                        <DataItem Dimensions = "3 1" Format = "XML">
                            {:d} 
                            {:d}
                            {:d}
                        </DataItem>
                        <DataItem Format = "HDF" Dimensions = "{:d}">{:s}.h5:/vector/{:d}</DataItem>
                    </DataItem>
                </Attribute>""".format(jj, 
                                       num_vertices, 
                                       jj, 
                                       basedim, 
                                       num_vertices, 
                                       num_vertices*basedim, real_basename, ii)
        string += r"""
            </Grid>"""
    string += r"""
        </Grid>
    </Domain>
</Xdmf>"""
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
    h5_file.create_dataset('/cells', (num_cells, basedim+1), data = np.array(mesh.cells(), dtype = np.uintp), compression = my_compression)
    h5_file.create_dataset('/facets', (num_facets, basedim), data = np.array([facet.entities(0) for facet in facets(mesh)], dtype = np.uintp), compression = my_compression)
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
        print('    writing {:d} cell functions'.format(len(cell_functions)))
        cell_functions_file = open(basename+'_cell_functions.xdmf', 'w')
        cell_functions_file.write(r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "cells" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>
        </Grid>""".format(
            num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
            num_cells, basedim+1, real_basename, 
            "XYZ" if basedim == 3 else "XY", 
            num_vertices, basedim, real_basename
        ))
        num = len(cell_functions)
        cell_functions_file.write(r"""
        <Grid Name = "cell_functions" GridType = "Collection" CollectionType = "Temporal">
            <Time TimeType = "List">
                <DataItem Format = "XML" Dimensions = "{:d}">
                    """.format(num))                
        for ii in range(num):
            cell_functions_file.write(" {:d}".format(ii))
        cell_functions_file.write(r"""
                </DataItem>
            </Time>""")
        for ii, ff in enumerate(cell_functions):
            cell_functions_file.write(r"""
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
            </Grid>""".format(
                ii,
                num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
                num_cells, basedim+1, real_basename, 
                "XYZ" if basedim == 3 else "XY", 
                num_vertices, basedim, real_basename,
                num_cells, real_basename, ii
            ))
        cell_functions_file.write(r"""
        </Grid>
    </Domain>
</Xdmf>""")
        cell_functions_file.close()
        del cell_functions_file
        print('    written {:d} cell functions'.format(len(cell_functions)))

    if facet_functions is not None:
        print('    writing {:d} facet functions'.format(len(facet_functions)))
        facet_functions_file = open(basename+'_facet_functions.xdmf', 'w')
        facet_functions_file.write(r"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
    <Domain>
        <Grid Name = "cells" GridType = "Uniform">
            <Topology NumberOfElements = "{:d}" TopologyType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/cells</DataItem>
            </Topology>
            <Geometry GeometryType = "{:s}">
                <DataItem Format = "HDF" Dimensions = "{:d} {:d}">{:s}.h5:/vertices</DataItem>
            </Geometry>
        </Grid>""".format(
            num_cells, 'Tetrahedron' if basedim == 3 else 'Triangle', 
            num_cells, basedim+1, real_basename, 
            "XYZ" if basedim == 3 else "XY", 
            num_vertices, basedim, real_basename
        ))
        num = len(facet_functions)
        facet_functions_file.write(r"""
        <Grid Name = "facet_functions" GridType = "Collection" CollectionType = "Temporal">
            <Time TimeType = "List">
                <DataItem Format = "XML" Dimensions = "{:d}">
                    """.format(num))
        for ii in range(num):
            facet_functions_file.write(" {:d}".format(ii))
        facet_functions_file.write(r"""
                </DataItem>
            </Time>""")
        for ii, ff in enumerate(facet_functions):
            facet_functions_file.write("""
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
            </Grid>""".format(
                ii,
                num_facets, 'Triangle' if basedim == 3 else 'PolyLine" NodesPerElement = "2', 
                num_facets, basedim, real_basename, 
                "XYZ" if basedim == 3 else "XY", 
                num_vertices, basedim, real_basename,
                num_facets, real_basename, ii
            ))
        facet_functions_file.write(r"""
        </Grid>
    </Domain>
</Xdmf>""")
        facet_functions_file.close()
        del facet_functions_file
        print('    written {:d} cell functions'.format(len(facet_functions)))
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
            print('    read {:d} cell functions'.format(ii))

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
            print('    read {:d} facet functions'.format(ii))

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
