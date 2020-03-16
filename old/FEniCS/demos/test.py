from dolfin import *
from mshr import *

domain = Rectangle(Point(0,0), Point(2,1))-Circle(Point(1,0.5),0.4)
holes = Circle(Point(0.5,0.5),0.2)+Circle(Point(1.5,0.5),0.2)

domain.set_subdomain(1,holes)

mesh = generate_mesh(domain, 20)
plot(mesh)

marker = MeshFunction('size_t',mesh,2,mesh.domains())
plot(marker)

VV=FunctionSpace(mesh,'CG',1)

rh=Constant(-6.)
u0=Expression('1+x[0]*x[0]+x[1]*x[1]',degree=2)
boundary=AutoSubDomain(lambda xx, on_boundary: on_boundary)
bc=DirichletBC(VV,u0,boundary)

test=TestFunction(VV)
trial=TrialFunction(VV)

kk=inner(grad(test),grad(trial))*dx
ll=inner(test,rh)*dx
AA, bb = assemble_system(kk,ll,bc)

uu = Function(VV)
solve(AA, uu.vector(), bb,'mumps')

File('u.pvd') << uu

#lot(uu)
#interactive()
