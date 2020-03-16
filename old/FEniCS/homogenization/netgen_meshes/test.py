import dolfin
from netgen.csg import *
from dolfin_utils.meshconvert import meshconvert

box=OrthoBrick(Pnt(-1,-1,-1),Pnt(1,1,1))
cone=Cone(Pnt(0,0,-1),Pnt(0,0,2),0,2)*Plane(Pnt(0,0,-1),Vec(0,0,-1))*Plane(Pnt(0,0,2),Vec(1,0,0))
geo=CSGeometry()
geo.Add(box*cone)
mesh=geo.GenerateMesh(maxh=0.2)
mesh.GenerateVolumeMesh()
mesh.Export('tmp.msh', 'Gmsh2 Format')
meshconvert.convert2xml('tmp.msh','tmp.xml')
bla=dolfin.Mesh('tmp.xml')
dolfin.File('tmp.pvd') << bla
