from dolfin import *
from mshr import *
import numpy as np
import numpy.random as rnd
import time
import matplotlib.pyplot as plt
import os
import gc

min_res = 12
max_res = 14
corner_refine = 4
corner_close = 0.3
corner_subdomains = []
for ii in range(corner_refine):
    corner_subdomains.append(AutoSubDomain((lambda what: lambda xx, on: np.sqrt(xx@xx) < what)(corner_close)))
    corner_close *= 0.6

#krylov_methods = list(krylov_solver_methods().keys())
#krylov_methods.remove('default')
krylov_methods = ['cg']
krylov_preconditioners = list(krylov_solver_preconditioners().keys())
krylov_preconditioners.remove('default')
lu_methods = list(lu_solver_methods().keys())
lu_methods.remove('default')
lu_methods.remove('superlu_dist')

errors = {
    'umfpack': True,
    'petsc': True,
    ('bicgstab','icc'): True,
    ('bicgstab','ilu'): True,
    ('bicgstab','jacobi'): True,
    ('bicgstab','none'): True,
    ('bicgstab','sor'): True,
    ('cg','icc'): True,
    ('cg','ilu'): True,
    ('cg','jacobi'): True,
    ('cg','none'): True,
    ('cg','sor'): True,
    ('gmres','icc'): True,
    ('gmres','ilu'): True,
    ('gmres','jacobi'): True,
    ('gmres','none'): True,
    ('gmres','sor'): True,
    ('minres','none'): True,
    ('richardson','icc'): True,
    ('richardson','jacobi'): True,
    ('richardson','none'): True,
    ('tfqmr','jacobi'): True,
    ('tfqmr','none'): True,
    ('tfqmr','icc'): True,
    ('tfqmr','ilu'): True,
}

number = 20
nodes = []
radii = []
count = 0
while count < number:
    tmp = 2*rnd.rand(2)-1
    if (tmp > 0).all():
        continue
    rad = 4*rnd.rand()/number
    ok = True
    for ii in range(len(nodes)):
        dist = np.sqrt((tmp-nodes[ii])@(tmp-nodes[ii]))
        if dist < 1.2*(rad+radii[ii]):
            ok = False
            break
    if ok:
        nodes.append(tmp)
        radii.append(rad)
        count += 1
domain = Rectangle(Point(-1, -1), Point(1, 1))-Rectangle(Point(0, 0), Point(2, 2))
inside = Circle(Point(0, 0), 0.6, 200)
outside = domain-inside
inclusions = Circle(Point(*nodes[0]), radii[0], 100)
for ii in range(1, number):
    inclusions += Circle(Point(*nodes[ii]), radii[ii], 100)
domain.set_subdomain(1, inside*inclusions)
domain.set_subdomain(2, outside*inclusions)

boundary = AutoSubDomain(lambda xx, on: on)

dimensions = []
times = dict()
iterations = dict()
for key in lu_methods:
    times[key] = ([], [], [])
for key in krylov_methods:
    for pre in krylov_preconditioners:
        times[(key, pre)] = ([], [], [])
        iterations[(key, pre)] = ([], [], [])


write = lambda res: res == max_res

print("""
----------
New run starting {:s}
""".format(time.asctime()))
exceptions_file = open('exceptions.txt', 'a')
exceptions_file.write("""
----------
New run starting {:s}
""".format(time.asctime()))
exceptions_file.flush()
ff = open('solvers_with_errors.txt', 'a')
ff.write("""
----------
New run starting {:s}
""".format(time.asctime()))
ff.write('Old\n')
for key in errors:
    ff.write('{:s}\n'.format(str(key)))
ff.write('----------\n')
ff.flush()

for resolution in range(min_res, max_res+1):
    print('\nResolution [{:d}] start'.format(resolution))
    print('    meshing')
    mesh = generate_mesh(domain, 2**resolution)
    cells = MeshFunction('size_t', mesh, 2, mesh.domains())
    cells.rename('cell_function', 'label')
    for ii in range(corner_refine):
        mf = CellFunction('bool', mesh, False)
        corner_subdomains[ii].mark(mf, True)
        mesh = refine(mesh, mf)
        cells = adapt(cells, mesh)
    if write(resolution):
        os.makedirs('out', exist_ok = True)
        File('out/cells_{:02d}.pvd'.format(resolution)) << cells
    help_array = np.asarray(cells.array(), dtype = int)
    V0 = FunctionSpace(mesh, 'DG', 0)
    kappa = Function(V0, name = 'kappa')
    kappa.vector().set_local(np.choose(help_array, [1, 10, 1e4]))

    print('    assembly')
    VV = FunctionSpace(mesh, 'CG', 1)
    dimensions.append(VV.dim())
    test = TestFunction(VV)
    trial = TrialFunction(VV)
    aa = kappa*inner(grad(trial), grad(test))*dx(mesh)

    LLs = [inner(Constant(-6), test)*dx(mesh)]

    bcs = [DirichletBC(VV, Constant(0), boundary)]
    dofmap = vertex_to_dof_map(VV)
    facets = FacetFunction('size_t', mesh)
    boundary.mark(facets, 1)
    for facet in SubsetIterator(facets, 1):
        for vertex in vertices(facet):
            fun = Function(VV)
            fun.vector()[dofmap[vertex.index()]] = 1.
            bcs.append(DirichletBC(VV, fun, boundary))
            LLs.append(inner(Constant(0), test)*dx(mesh))
            break
        break
    del dofmap, facets

    for ii, bc in enumerate(bcs):
        print('Case [{:d}/2]'.format(ii+1))
        LL = LLs[ii]
        AA, bb = assemble_system(aa, LL, bc)
        print('    assembled')

        print('    Solvers:')
        for key in lu_methods:
            if key in errors:
                continue
            print('        [{:s}] starting'.format(key))
            uu = Function(VV, name = 'u')
            try:
                old = time.process_time()
                solve(AA, uu.vector(), bb, key)
                new = time.process_time()
            except Exception as ee:
                print('            [{:s}] solver error'.format(key))
                errors[key] = True
                ff.write('{:s}\n'.format(str(key)))
                ff.flush()
                times[key][ii].append(float('NaN'))
                exceptions_file.write('res [{:d}], bc [{:d}], {:s}, {:s}\n{:s}\n\n'.format(resolution, ii, key, str(type(ee)), str(ee)))
                exceptions_file.flush()
            else:
                times[key][ii].append(new-old)
                if write(resolution):
                    File('out/bc_{:d}/res_{:d}_{:s}.pvd'.format(ii, resolution, key)) << uu
                print('            [{:s}] solver finished in [{:.2e}] seconds'.format(key, new-old))

        for key in krylov_methods:
            print('        [{:s}] starting'.format(key))
            for pre in krylov_preconditioners:
                if (key, pre) in errors:
                    continue
                print('            [{:s}] preconditioner'.format(pre))
                uu = Function(VV, name = 'u')
                try:
                    old = time.process_time()
                    solver = KrylovSolver(key, pre)
                    it = solver.solve(AA, uu.vector(), bb)
                    new = time.process_time()
                    del solver
                except Exception as ee:
                    print('                [{:s}, {:s}] solver error'.format(key, pre))
                    errors[(key, pre)] = True
                    ff.write('{:s}\n'.format(str((key,pre))))
                    ff.flush()
                    times[(key, pre)][ii].append(float('NaN'))
                    iterations[(key, pre)][ii].append(float('NaN'))
                    exceptions_file.write('res [{:d}], bc [{:d}], {:s}, {:s}, {:s}\n{:s}\n\n'.format(resolution, ii, key, pre, str(type(ee)), str(ee)))
                    exceptions_file.flush()
                else:
                    times[(key, pre)][ii].append(new-old)
                    iterations[(key, pre)][ii].append(it)
                    if write(resolution):
                        File('out/bc_{:d}/res_{:d}_{:s}_{:s}.pvd'.format(ii, resolution, key, pre)) << uu
                    print('                [{:s}, {:s}] solver finished in [{:.2e}] seconds'.format(key, pre, new-old))
            print('        [{:s}] done'.format(key))
        del AA, bb, LL
        gc.collect()
    del mesh, V0, VV, test, trial, aa, bcs, LLs
    gc.collect()
    print('\nResolution [{:d}] done'.format(resolution))

exceptions_file.close()
ff.close()

rows = 4
for ii in range(2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = []
    count = 0
    for key in lu_methods:
        if key in errors:
            continue
        handles.append(*ax.loglog(dimensions, times[key][ii], 'o:', label = key))
        count += 1
    if count:
        ax.grid(True)
        ax.set_ylabel(r'solution time')
        ax.set_xlabel(r'$\operatorname{dim}V$')
        fig.tight_layout()
        fig.savefig('lu_times_{:d}.pdf'.format(ii))

        if not ii:
            figlegend = plt.figure(figsize = (np.ceil(count/rows)*2, rows), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil(count/rows)))
            figlegend.savefig('lu_legend.pdf'.format(ii), bbox_extra_artists = (lgd, ))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = []
    count = 0
    for key in krylov_methods:
        for pre in krylov_preconditioners:
            if (key, pre) in errors:
                continue
            handles.append(*ax.loglog(dimensions, times[(key, pre)][ii], 'o:', label = '{:s}, {:s}'.format(key, pre)))
            count += 1
    if count:
        ax.grid(True)
        ax.set_ylabel(r'solution time')
        ax.set_xlabel(r'$\operatorname{dim}V$')
        fig.tight_layout()
        fig.savefig('krylov_times_{:d}.pdf'.format(ii))

        if not ii:
            figlegend = plt.figure(figsize = (np.ceil(count/rows)*4, rows), frameon = False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc = 10, ncol = int(np.ceil(count/rows)))
            figlegend.savefig('krylov_legend.pdf'.format(ii), bbox_extra_artists = (lgd, ))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for key in krylov_methods:
            for pre in krylov_preconditioners:
                if (key, pre) in errors:
                    continue
                ax.loglog(dimensions, iterations[(key, pre)][ii], 'o:', label = '{:s}, {:s}'.format(key, pre))
        ax.legend(loc = 0)
        ax.grid(True)
        ax.set_ylabel(r'solution time')
        ax.set_xlabel(r'$\operatorname{dim}V$')
        fig.tight_layout()
        fig.savefig('krylov_iterations_{:d}.pdf'.format(ii))
