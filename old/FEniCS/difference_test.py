from dolfin import *

import matplotlib.pyplot as plt


ff = Constant(-6.)
u0 = Expression('1+x[0]*x[0]+2*x[1]*x[1]')
def u0_boundary(xx, on_boundary):
    return on_boundary

uus = []
nns = []
fine_errors = []
real_errors = []
maxnn=8

for kk in xrange(1,maxnn+1):
    nn = 2**kk
    nns.append(nn)

    mesh = UnitSquareMesh(nn, nn)
    VV = FunctionSpace(mesh, 'Lagrange', 1)

    bc = DirichletBC(VV, u0, u0_boundary)

    uu = TrialFunction(VV)
    vv = TestFunction(VV)
    aa = inner(nabla_grad(uu), nabla_grad(vv))*dx
    LL = ff*vv*dx

    uu = Function(VV)
    solve(aa == LL, uu, bc)
    uus.append(uu)

for kk in xrange(1,maxnn):
    fine_errors.append(errornorm(uus[-1], uus[kk]))
    real_errors.append(errornorm(u0, uus[kk]))

print fine_errors
print real_errors

plt.figure()
plt.loglog(nns[:-1], fine_errors)
plt.title('Errors to fine solution')
plt.figure()
plt.loglog(nns[:-1], real_errors)
plt.title('Errors to real solution')
plt.show()
