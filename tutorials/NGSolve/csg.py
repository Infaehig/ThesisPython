import os

import h5py
import netgen.csg as csg
import dolfin
import numpy
from dolfin_utils.meshconvert import meshconvert

basename = 'test'

cube = csg.OrthoBrick(csg.Pnt(-20, -5, -1), csg.Pnt(20, 5, 1))
hole = csg.Cylinder(csg.Pnt(0,0,-2), csg.Pnt(0,0,2), 1)

geo = csg.CSGeometry()
geo.Add(cube-hole)
mesh = geo.GenerateMesh(maxh=0.5)
print('Mesh created')
del geo
mesh.Export('tmp.msh', 'Gmsh2 Format')
print('Mesh saved as .msh')
del mesh
meshconvert.convert2xml('tmp.msh', 'tmp.xml')
os.remove('tmp.msh')
os.remove('tmp_facet_region.xml')
os.remove('tmp_physical_region.xml')
print('Mesh converted to .xml')

mesh = dolfin.Mesh('tmp.xml')
print('Mesh loaded as .xml')
os.remove('tmp.xml')

mesh.init()
basedim = mesh.geometry().dim()
num_vertices = mesh.num_vertices()
num_cells = mesh.num_cells()
num_facets = mesh.num_facets()

h5_file = h5py.File(f'{basename}.h5', 'w')
h5_file.create_dataset('/vertices', (num_vertices, basedim), data = mesh.coordinates())
h5_file.create_dataset('/cells', (num_cells, basedim+1), data = numpy.array(mesh.cells(), dtype = numpy.uintp))
h5_file.create_dataset('/facets', (num_facets, basedim), data = numpy.array([facet.entities(0) for facet in dolfin.facets(mesh)], dtype = numpy.uintp))
h5_file.close()
print('Mesh written as hdf5')
del h5_file
del mesh

cell_file = open(f'{basename}_cells.xdmf', 'w')
cell_file.write(f"""<?xml version = "1.0"?>
<Xdmf Version = "2.0" xmlns:xi = "http://www.w3.org/2001/XInclude">
<Domain>
	<Grid Name = "cells" GridType = "Uniform">
		<Topology NumberOfElements = "{num_cells}" TopologyType = "Tetrahedron">
			<DataItem Format = "HDF" Dimensions = "{num_cells} {basedim+1}">{basename}.h5:/cells</DataItem>
		</Topology>
		<Geometry GeometryType = "XYZ">
			<DataItem Format = "HDF" Dimensions = "{num_vertices} {basedim}">{basename}.h5:/vertices</DataItem>
		</Geometry>
	</Grid>
</Domain>
</Xdmf>""")
cell_file.close()
print('Mesh written as xdmf')
del cell_file

mesh = dolfin.Mesh()
xdmf = dolfin.XDMFFile(f'{basename}_cells.xdmf')
xdmf.read(mesh)
print('Read mesh again')