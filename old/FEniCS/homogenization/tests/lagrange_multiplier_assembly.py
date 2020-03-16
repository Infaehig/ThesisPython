from dolfin import *
from time import process_time as tic

def run_test(max_kk=10):
    for kk in range(max_kk):
        print(kk)
        mesh=UnitSquareMesh(2**kk,2**kk)

        VV=FunctionSpace(mesh,'CG',1)
        uu=TrialFunction(VV)
        vv=TestFunction(VV)
        t0 = tic()
        AA=assemble(inner(grad(uu),grad(vv))*dx)
        tt_regular = tic()-t0
        print('    {:3e}'.format(tt_regular))

        WV=FiniteElement('CG',mesh.ufl_cell(),1)
        WR=FiniteElement('R',mesh.ufl_cell(),0)
        WW=FunctionSpace(mesh,WV*WR)
        uu, pp=TrialFunctions(WW)
        vv, qq=TestFunctions(WW)

        t0 = tic()
        BB=assemble(inner(grad(uu),grad(vv))*dx)
        tt_mixed = tic()-t0
        print('    {:3e} = [{:3e}] x {:3e}'.format(tt_mixed,tt_mixed/tt_regular,tt_regular))

if __name__ == '__main__':
    run_test(10)
