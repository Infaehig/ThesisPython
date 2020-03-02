from dolfin import *
import numpy
import numpy.linalg as la
import matplotlib.pyplot as plt

def error(nn):

    mesh = UnitSquareMesh(nn, nn)
    VV = FunctionSpace(mesh, 'CG', 1)

    V0 = FunctionSpace(mesh, 'CG', 2)
    u0 = Function(V0)
    u0.interpolate(Expression('1+x[0]*x[0]+2*x[1]*x[1]'))

    fine_mesh = refine(refine(mesh))
    Vf = FunctionSpace(fine_mesh, 'CG', 1)

    boundary = AutoSubDomain(lambda xx, on_boundary: on_boundary)

    whole_domain = AutoSubDomain(lambda xx, on_boundary: True)
    holes = AutoSubDomain(lambda xx, on_boundary: True if (abs(xx[0]-0.4) <= 0.1 and abs(xx[1]-0.4) <= 0.1) or (abs(xx[0]-0.6) <= 0.1 and abs(xx[1]-0.6) <= 0.1) else False)
    subdomains = MeshFunction('size_t', mesh, 2)
    whole_domain.mark(subdomains, 0)
    holes.mark(subdomains, 1)
    V0 = FunctionSpace(mesh, 'DG', 0)
    kk = Function(V0)
    kk_values = [1, 100]
    helper = numpy.asarray(subdomains.array(), dtype=numpy.int32)
    kk.vector()[:] = numpy.choose(helper, kk_values)

    bc = DirichletBC(VV, u0, boundary)

    uu = TrialFunction(VV)
    vv = TestFunction(VV)
    ff = Constant(-6)
    aa = kk*inner(nabla_grad(uu), nabla_grad(vv))*dx
    LL = ff*vv*dx

    ur = Function(VV)
    solve(aa == LL, ur, bc)

    plot(ur, title=str(nn)+' regular solution', scale=5.)

    beta = Constant(10)
    hh = mesh.hmax()
    normal = FacetNormal(mesh)

    uu = TrialFunction(VV)
    vv = TestFunction(VV)

    aa = kk*inner(grad(uu), grad(vv))*dx - inner(dot(grad(uu), normal), vv)*ds - inner(uu, dot(grad(vv), normal))*ds + beta/hh*inner(uu, vv)*ds
    LL = inner(ff, vv)*dx - inner(u0, dot(grad(vv), normal))*ds + beta/hh*inner(u0, vv)*ds

    un = Function(VV)
    solve(aa == LL, un)

    plot(un, title=str(nn)+' nitsche solution', scale=5.)

    basis = []
    dim = VV.dim()
    for ii in xrange(dim):
        basis.append(Function(VV))
        basis[-1].vector()[ii] = 1
    stiffness = numpy.zeros((dim, dim))
    rhs = numpy.zeros(dim)
    for ii in xrange(dim):
        stiffness[ii,ii] = assemble(kk*inner(grad(basis[ii]), grad(basis[ii]))*dx - 2*inner(dot(grad(basis[ii]), normal), basis[ii])*ds + beta/hh*inner(basis[ii], basis[ii])*ds)
        for jj in xrange(ii+1, dim):
            stiffness[ii,jj] = assemble(kk*inner(grad(basis[ii]), grad(basis[jj]))*dx - inner(dot(grad(basis[ii]), normal), basis[jj])*ds - inner(basis[ii], dot(grad(basis[jj]), normal))*ds + beta/hh*inner(basis[ii], basis[jj])*ds)
            stiffness[jj,ii] = stiffness[ii,jj]
        rhs[ii] = assemble(inner(ff, basis[ii])*dx-inner(u0, dot(grad(basis[ii]), normal))*ds + beta/hh*inner(u0, basis[ii])*ds)
    coeff = la.solve(stiffness, rhs)
    um = Function(VV)
    um.vector()[:] = coeff
    plot(um, title=str(nn)+' nitsche manually assembled', scale=5.)

    AA = assemble(aa)
    bb = assemble(LL)

if __name__ == '__main__':
    for ii in xrange(2):
        error(4*2**ii)
    interactive()
