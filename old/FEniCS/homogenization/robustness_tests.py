from localizations_nitsche import *

plotcolors_list = ['b','b','g','g','r','r','y','y','c','c','m','m']
plotmarkers_list = ['bo:','b+:','go:','g+:','ro:','r+:','yo:','y+:','co:','c+:','mo:','m+:']
markernum = len(plotcolor_list)

def robustness_test(uus, mesh, mesh_regions, mesh_facets, extended_mesh, extended_mesh_regions, extended_mesh_facets, elasticity=False, matplotlib=False, contrast=1e4, prefix='test', box=np.array([0,0,1,1]), outdir='heat', outer_bc=False):
    comm = mpi_comm_world()
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    basedim = mesh.geometry().dim()
    low = box[:basedim]; high = box[basedim:]; lengths = high-low

    keys = list(uus.keys())
    dofmax = np.max([len(uus[key]) for key in keys])
    plotmarkers = dict()
    plotcolor = dict()
    for ii,key in enumerate(keys):
        plotmarkers[key] = plotmarkers_list[ii%markernum]
        plotcolors[key] = plotcolors_list[ii%markernum]

    print('\nSetup')
    # Global Test problem
    whole_boundary = AutoSubDomain(lambda xx, on_boundary: on_boundary)

    fine_mesh = mesh
    basedim = mesh.geometry().dim()
    if elasticity:
        fine_VV = VectorFunctionSpace(fine_mesh, 'CG', 1, basedim)
    else:
        fine_VV = FunctionSpace(fine_mesh, 'CG', 1)
    fine_dim = fine_VV.dim()
    fine_hh = fine_mesh.hmax()
    fine_normal = FacetNormal(fine_mesh)
    fine_one = Function(fine_VV)
    fine_one.vector()[:] = 1

    # On extended mesh
    extended_V0 = FunctionSpace(extended_mesh, 'DG', 0)
    if elasticity:
        extended_VV = VectorFunctionSpace(extended_mesh, 'CG', 1, basedim)
    else:
        extended_VV = FunctionSpace(extended_mesh, 'CG', 1)
    extended_dim = extended_VV.dim()

    print('    Forms')
    V0 = FunctionSpace(fine_mesh, 'DG', 0)
    help_array = np.asarray(mesh_regions.array(), dtype=np.int32)
    extended_help_array = np.asarray(extended_mesh_regions.array(), dtype=np.int32)
    kappa_values = np.array([1.,contrast,1.])
    if elasticity:
        poisson = 0.3
        kappa_values *= 7e11
        ll_values = (poisson*kappa_values)/((1.+poisson)*(1.-2.*poisson))
        mu_values = kappa_values/(2.*(1.+poisson))
        ll = Function(V0)
        ll.vector()[:] = np.choose(help_array, ll_values)
        extended_ll = Function(extended_V0)
        extended_ll.vector()[:] = np.choose(extended_help_array, ll_values)
        mu = Function(V0)
        mu.vector()[:] = np.choose(help_array, mu_values)
        extended_mu = Function(extended_V0)
        extended_mu.vector()[:] = np.choose(extended_help_array, mu_values)
        kappa = Function(V0)
        kappa.vector()[:] = np.choose(help_array, kappa_values)
        epsilon = lambda uu: (grad(uu)+nabla_grad(uu))*0.5
        sigma = lambda uu: 2.*mu*epsilon(uu)+ll*tr(epsilon(uu))*Identity(basedim)
        extended_sigma = lambda uu: 2*extended_mu*epsilon(uu)+extended_ll*tr(epsilon(uu))*Identity(basedim)
        extended_kk_form = lambda uu, vv: inner(extended_sigma(uu), epsilon(vv))*dx(extended_mesh)
        interior_kk_form = lambda uu, vv: inner(extended_sigma(uu), epsilon(vv))*(dx(subdomain_id=0, subdomain_data=extended_mesh_regions)+dx(subdomain_id=1, subdomain_data=extended_mesh_regions))
        zero = Constant([0]*basedim)

        VT = TensorFunctionSpace(mesh, 'DG', 0)
    else:
        kappa = Function(V0)
        kappa.vector()[:] = np.choose(help_array, kappa_values)
        extended_kappa = Function(extended_V0)
        extended_kappa.vector()[:] = np.choose(extended_help_array, kappa_values)
        epsilon = lambda uu: grad(uu)
        sigma = lambda uu: kappa*epsilon(uu)
        extended_sigma = lambda uu: extended_kappa*epsilon(uu)
        kk_form = lambda uu, vv: inner(sigma(uu), epsilon(vv))*dx
        extended_kk_form = lambda uu, vv: inner(extended_sigma(uu), epsilon(vv))*dx
        interior_kk_form = lambda uu, vv: inner(extended_sigma(uu), epsilon(vv))*(dx(subdomain_id=0, subdomain_data=extended_mesh_regions)+dx(subdomain_id=1, subdomain_data=extended_mesh_regions))
        zero = Constant(0) 

        VT = VectorFunctionSpace(mesh, 'DG', 0, basedim)

    l2_form = lambda uu, vv: inner(uu, vv)*dx
    interior_l2_form = lambda uu, vv: inner(uu, vv)*(dx(subdomain_id=0, subdomain_data=extended_mesh_regions)+dx(subdomain_id=1, subdomain_data=extended_mesh_regions))
    ref_l2_form = lambda uu, vv: inner(uu, vv)*dx(ref_mesh)

    # More forms
    extended_trial = TrialFunction(extended_VV)
    extended_test = TestFunction(extended_VV)
    
    extended_kk = extended_kk_form(extended_trial, extended_test)
    interior_kk = interior_kk_form(extended_trial, extended_test)
    extended_mm = l2_form(extended_trial, extended_test)
    interior_mm = interior_l2_form(extended_trial, extended_test)

    extended_zero = inner(zero,extended_test)*dx

    print('    fine mesh matrix assembly')
    extended_KK = PETScMatrix()
    assemble(extended_kk, tensor = extended_KK)
    interior_KK = PETScMatrix()
    assemble(interior_kk, tensor = interior_KK)
    extended_MM = PETScMatrix()
    assemble(extended_mm, tensor = extended_MM)
    interior_MM = PETScMatrix()
    assemble(interior_mm, tensor = interior_MM)

    def extended_inner(uu, vv):
        return uu.vector().inner(extended_MM*vv.vector())

    def extended_l2(uu):
        return sqrt(extended_inner(uu,uu))

    def fine_project(uu):
       #vv = project(uu, fine_VV, solver_type=solver_type)
        vv = interpolate(uu, fine_VV)
        vv.vector()[:] /= interior_l2(vv)
        vv.rename('u','label')
        return vv

    def fine_interpolate(uu):
        vv = interpolate(uu, fine_VV)
        vv.vector()[:] /= interior_l2(vv)
        vv.rename('u','label')
        return vv

    def extended_harmonic(uu):
        vv = Function(extended_VV, name='u')
        if outer_bc:
            bcs = [DirichletBC(extended_VV, uu, extended_mesh_facets, ii) for ii in range(1,5)]
            solve(extended_kk == extended_zero, vv, bcs, solver_parameters=params)
        else:
            solve(extended_kk == extended_zero, vv, DirichletBC(extended_VV, uu, whole_boundary), solver_parameters=params)
        vv.vector()[:] /= extended_l2(vv)
        return vv

    fine_trial = TrialFunction(fine_VV)
    fine_test = TestFunction(fine_VV)
    fine_kk = kk_form(fine_trial, fine_test)
    fine_l2 = l2_form(fine_trial, fine_test)
    fine_zero = inner(zero,fine_test)*dx
    
    fine_MM = PETScMatrix()
    assemble(fine_l2, tensor = fine_MM, form_compiler_parameters=mass_parameters)
    fine_KK = PETScMatrix()
    assemble(fine_kk, tensor = fine_KK)

    def fine_inner_sqrt(uu, vv):
        l2 = sqrt(uu.inner(fine_MM*vv))
        h1 = sqrt(uu.inner(fine_KK*vv))
        return l2, h1

    def fine_error(uu, vv):
        return fine_inner_sqrt(uu-vv,uu-vv)

    nitsche_kk = fine_kk - inner(dot(sigma(fine_trial), fine_normal), fine_test)*ds - inner(fine_trial, dot(sigma(fine_test), fine_normal))*ds + beta/fine_hh*inner(fine_trial, fine_test)*ds 
    nitsche_KK = assemble(nitsche_kk)

    print('Computing reference solutions')
    # Right hand sides
    if elasticity:
        rhs = [zero]#, Constant((0,-0.1))]
    else:
        rhs = [Constant(0)]

    # Boundary Condition
    if elasticity: 
        left_bc = Constant([-1e-4]+[0.]*(basedim-1))
        right_bc = Constant([1e-4]+[0.]*(basedim-1))
        top_bc = Constant([0.]+[1e-3]+[0.]*(basedim-2))
        bottom_bc = Constant([0.]+[-1e-3]+[0.]*(basedim-2))

        all_bcs = [[(left_bc, 1), (right_bc, 2)],\
                   [(right_bc, 1), (left_bc, 2)],
                   [(top_bc, 1), (bottom_bc, 2)]]
    else:
        bla = Expression('sin((2*pi*(x[0]-xx)/aa+(x[1]-yy)/bb))', xx=low[0],aa=lengths[0],yy=low[1],bb=lengths[1], degree=10)
        all_bcs = [[(Expression('1+pow((x[0]-xx)/aa,2)+2.*pow((x[1]-yy)/bb,2)',xx=low[0],aa=lengths[0],yy=low[1],bb=lengths[1], degree=2),None)],\
                   [(bla,3),(bla,4)]]     

    rhs_bcs = [(rh, bcs) for rh in rhs for bcs in all_bcs]

    num_pairs = len(rhs_bcs)
    fine_solutions = []
    nitsche_KKs = []
    nitsche_LLs = []
    fine_l2s = []
    fine_h1s = []
    for ii, (ff, bcs) in enumerate(rhs_bcs):
        print('\nReference solutions: pair ', ii)
        basename = prefix+'_'+str(ii)

        if len(bcs) == 1 and bcs[0][1] == None:
            bc = bcs[0][0]
            nitsche_KKs.append(nitsche_KK)
            nitsche_rhs = inner(ff, fine_test)*dx(fine_mesh) - inner(bc, dot(sigma(fine_test), fine_normal))*ds(fine_mesh) + beta/fine_hh*inner(bc, fine_test)*ds(fine_mesh)
            fine_bcs = [DirichletBC(fine_VV, bc, mesh_facets, kk) for kk in range(1,5)]
        else:
            tmp_ds = Measure('ds', domain=fine_mesh, subdomain_data=mesh_facets)
            tmp_kk = kk_form(fine_trial, fine_test)
            nitsche_rhs = l2_form(ff, fine_test)
            for (bc, domain) in bcs: 
                tmp_kk += (-inner(dot(sigma(fine_trial), fine_normal), fine_test) - inner(fine_trial, dot(sigma(fine_test), fine_normal)))*tmp_ds(domain) + beta/fine_hh*inner(fine_trial, fine_test)*tmp_ds(domain)
                nitsche_rhs += (-inner(bc, dot(sigma(fine_test), fine_normal)) + beta/fine_hh*inner(bc, fine_test))*tmp_ds(domain)
            tmp_KK = assemble(tmp_kk)
            nitsche_KKs.append(tmp_KK)
            fine_bcs = [DirichletBC(fine_VV, bc, mesh_facets, domain) for (bc,domain) in bcs]

        nitsche_LL = assemble(nitsche_rhs)

        nitsche_LLs.append(nitsche_LL)
 
        print('        Compute reference solution')
        # Finest solution
        fine_rh = inner(ff, fine_test)*dx
        fine_uu = Function(fine_VV, name='u')

        AA, bb = assemble_system(fine_kk, fine_rh, fine_bcs)
        solve(AA, fine_uu.vector(), bb, solver_type)
        
        fine_solutions.append(fine_uu)

        fine_l2, fine_h1 = fine_inner_sqrt(fine_uu.vector(),fine_uu.vector())

        fine_l2s.append(fine_l2)
        fine_h1s.append(fine_h1)

    print("\nReference solutions computed")
    def interior_inner(uu, vv):
        return uu.vector().inner(fine_MM*vv.vector())

    def interior_l2(uu):
        return sqrt(interior_inner(uu, uu))

    def extended_orthogonalize(uu, uus):
        uu.vector()[:] /= sqrt(uu.vector().inner(uu.vector()))
        for vv in uus:
            uu.vector().axpy(-uu.vector().inner(vv.vector()),vv.vector())
        ret = sqrt(uu.vector().inner(uu.vector()))
        uu.vector()[:] /= ret
        return ret

    def interior_orthogonalize(uu, uus):
        return extended_orthogonalize(uu, uus)

    print('Setting up supinf computations')
    print('    Null space')
    extended_null_basis = []
    if elasticity:
        extended_null_basis = [interpolate(Constant((1,0)),extended_VV),\
                               interpolate(Constant((0,1)),extended_VV),\
                               interpolate(Expression(('-x[1]','x[0]'),degree=1), extended_VV)]
        interior_null_basis = [interpolate(Constant((1,0)),fine_VV),\
                               interpolate(Constant((0,1)),fine_VV),\
                               interpolate(Expression(('-x[1]','x[0]'),degree=1), fine_VV)]
    else:
        extended_null_basis = [interpolate(Constant(1),extended_VV)]
        interior_null_basis = [interpolate(Constant(1),fine_VV)]
    null_dim = len(extended_null_basis)
    for ii in range(null_dim):
        extended_null_basis[ii].rename('u','label')
        extended_orthogonalize(extended_null_basis[ii],extended_null_basis[:ii])
        interior_null_basis[ii].rename('u','label')
        interior_orthogonalize(interior_null_basis[ii],interior_null_basis[:ii])
        interior_null_basis[ii].vector()[:] /= interior_l2(interior_null_basis[ii])

    num = np.min([fine_dim, int(1.3*dofmax)])
    print('    Harmonic polynomials')
    harmonic_factory = HarmonicFactory(extended_VV, extended_MM, elasticity, base=low, lengths=lengths)
    while harmonic_factory.num_products < num:
        harmonic_factory.initialize(harmonic_factory.degree+1)
    print('        initialized until degree ', harmonic_factory.degree)
    harmonics = harmonic_factory.products
    for ii in range(harmonic_factory.degree+1):
        harmonic_dim = harmonic_factory.degree_offsets[ii+1]
        if harmonic_dim >= dofmax:
            harmonic_degree = ii
            break

    print('        k-harmonic extensions of harmonic polynomials')
    old = sa_utils.get_time()
    extended_harmonic_polys = extended_laplace_polys[:harmonic_factory.degree_offsets[1]]
    new = sa_utils.get_time()
    harmonic_times_harmonic = [new-old]
    for deg in range(1,harmonic_factory.degree+1):
        for ii in range(harmonic_factory.degree_offsets[deg],harmonic_factory.degree_offsets[deg+1]):
            extended_harmonic_polys.append(extended_harmonic(extended_laplace_polys[ii]))
        harmonic_times_harmonic.append(harmonic_times_harmonic[-1]+new-old)

    print('    Building basis orthogonal to null space')

    extended_orthogonalized = []
    interior_orthogonalized = []
    idx = 0
    alldofs = [null_dim]
    count = null_dim
    degree = harmonic_factory.degree
    offsets  = harmonic_factory.degree_offsets
    functions = extended_harmonic_polys

    old = sa_utils.get_time()
    for ii in range(1,degree+1):
        for jj in range(offsets[ii],offsets[ii+1]):
            uu = functions[jj].copy()
            extended_orthogonalize(uu, extended_null_basis)
            ret = extended_orthogonalize(uu, extended_orthogonalized)
            if ret < myeps:
                print('            skip', jj)
            else:
                extended_orthogonalized.append(uu)
                interior_orthogonalized.append(fine_project(uu))
                count += 1
        if count > alldofs[-1]:
            alldofs.append(count)
    for uu in extended_orthogonalized:
        uu.vector()[:] /= extended_l2(uu)
    multiple_degrees = []

    extended_harmonic_dim = alldofs[-1]-null_dim
    for ii in range(len(alldofs)):
        newmax = alldofs[ii]
        if newmax > dofmax:
            break
    print('    total: ['+str(alldofs[-1])+'], shapes ['+str(newmax)+']')
    newdofs = alldofs[1:ii+1]

    double_dofs = []
    jj = 0
    for ii in range(len(newdofs)):
        while(alldofs[jj]-null_dim < 2*(newdofs[ii]-null_dim) and jj < len(alldofs)-1):
            jj += 1
        double_dofs.append(jj)
    assert(len(alldofs)-1 == double_dofs[-1])
    assert(extended_harmonic_dim > dofmax)
    assert(extended_harmonic_dim == len(extended_orthogonalized))

    dofs = dict()
    for key in keys:
        dim = len(uus[key])
        if dim > newdofs[-1]:
            dofs[key] = append(newdofs,dim)
        else:
            ii = len(newdofs)
            while (dofs[key][ii-1] >= dim):
                ii -= 1
            dofs[key] = append(newdofs,dim)

    print('    Eigensolutions from extended Polynomials')
    lasti = 0
    extended_KP_harmonic = []
    interior_KP_harmonic = []
    interior_MP_harmonic = []
    fine_KP_harmonic = []
    fine_MP_harmonic = []
    for deg in range(len(newdofs)):
        currenti = alldofs[double_dofs[deg]]-null_dim
        for ii in range(lasti,currenti):
            extended_KP_harmonic.append(extended_KK*extended_orthogonalized[ii].vector())
            interior_KP_harmonic.append(interior_KK*extended_orthogonalized[ii].vector())
            interior_MP_harmonic.append(interior_MM*extended_orthogonalized[ii].vector())
            fine_KP_harmonic.append(fine_KK*interior_orthogonalized[ii].vector())
            fine_MP_harmonic.append(fine_MM*interior_orthogonalized[ii].vector())
        lasti = currenti
    
    extended_KK_harmonic = np.zeros((extended_harmonic_dim,extended_harmonic_dim))
    interior_KK_harmonic = np.zeros((extended_harmonic_dim,extended_harmonic_dim))
    interior_MM_harmonic = np.zeros((extended_harmonic_dim,extended_harmonic_dim))

    lasti = 0
    for deg in range(len(newdofs)):
        currenti = alldofs[double_dofs[deg]]-null_dim
        for ii in range(lasti,currenti):
            extended_KK_harmonic[ii,ii] = extended_orthogonalized[ii].vector().inner(extended_KP_harmonic[ii])
            interior_KK_harmonic[ii,ii] = extended_orthogonalized[ii].vector().inner(interior_KP_harmonic[ii])
            interior_MM_harmonic[ii,ii] = extended_orthogonalized[ii].vector().inner(interior_MP_harmonic[ii])
            for jj in range(ii):
                extended_KK_harmonic[ii,jj] = extended_orthogonalized[ii].vector().inner(extended_KP_harmonic[jj])
                extended_KK_harmonic[jj,ii] = extended_KK_harmonic[ii,jj]
                interior_KK_harmonic[ii,jj] = extended_orthogonalized[ii].vector().inner(interior_KP_harmonic[jj])
                interior_KK_harmonic[jj,ii] = interior_KK_harmonic[ii,jj]
                interior_MM_harmonic[ii,jj] = extended_orthogonalized[ii].vector().inner(interior_MP_harmonic[jj])
                interior_MM_harmonic[jj,ii] = interior_MM_harmonic[ii,jj]
        lasti = currenti
    print('    Harmonic Polynomial matrices set up in: [', new-old, '] seconds')

    min_dof = np.min([dofs[key][0] for key in keys])
    max_dof = np.max([dofs[key][-1] for key in keys])

    ns = [] 
    n1 = []
    n2 = []
    for ii in range(max_dof-null_dim+1,extended_harmonic_dim,2):
        ns.append(ii)
        n1.append(la.eigvalsh(interior_KK_harmonic[:ii,:ii],extended_KK_harmonic[:ii,:ii]))
        n2.append(la.eigvalsh(interior_MM_harmonic[:ii,:ii],extended_KK_harmonic[:ii,:ii]))

    print('    H1 nwidths')
    eigvals, eigvalues = la.eigh(interior_KK_harmonic,extended_KK_harmonic)
    idx = np.argsort(eigvals)[::-1]
    idx = idx[:max_dof-null_dim+2]
    eigvals_max = eigvals[idx[0]]
    max_idx1 = np.max(np.where(eigvals[idx] > myeps*eigvals_max)[0])
    nwidths1 = np.append(np.array([float('NaN')]*null_dim),np.sqrt(eigvals[idx]))

    print('    L2 nwidths')
    times_eigensolver = []
    eigvals, eigvalues = la.eigh(interior_MM_harmonic,extended_KK_harmonic)
    idx = np.argsort(eigvals)[::-1]
    idx = idx[:max_dof-null_dim+2]
    eigvals_max = eigvals[idx[0]]
    max_idx2 = np.max(np.where(eigvals[idx] > myeps*eigvals_max)[0])
    nwidths2 = np.append(np.array([float('NaN')]*null_dim),np.sqrt(eigvals[idx]))

    solution_vectors = dict()
    solutions = [dict() for ii in range(num_pairs)]
    supinfsE = dict()
    supinfs0 = dict()
    for key in keys:
        for ii in range(num_pairs):
            solutions[ii][key] = []

    print('Computing stuff with coarse bases')
    def compute_stuff_for_key(key, supinfs0_q, supinfs1_q, solutions_q):
        string = '    '+key+'\n'
        solution_vectors[key] = [[] for ii in range(num_pairs)]
        supinfsE[key] = []
        supinfs0[key] = []

        for deg,coarse_dim in enumerate(dofs[key]):
            string += '        '+str(coarse_dim)+'\n'
            funs = uus[key][:coarse_dim]
 
            string += '            computing supinfs\n'
            HH = np.zeros((coarse_dim,coarse_dim))
            for ii in range(coarse_dim):
                vv = fine_KK*funs[ii].vector()
                HH[ii,ii] = funs[ii].vector().inner(vv)
                for jj in range(ii+1,coarse_dim):
                    HH[ii,jj] = funs[jj].vector().inner(vv)
                    HH[jj,ii] = HH[ii,jj]
            TT = np.zeros((coarse_dim,extended_harmonic_dim))
            for ii in range(coarse_dim):
                for jj in range(extended_harmonic_dim):
                    TT[ii,jj] = funs[ii].vector().inner(fine_KP_harmonic[jj])
            Mtilde = interior_KK_harmonic - TT.T.dot(la.solve(HH, TT))
            eigvals = la.eigvalsh(Mtilde, extended_KK_harmonic)
            supinfsE[key].append(sqrt(eigvals[-1]))

            HH = np.zeros((coarse_dim,coarse_dim))
            for ii in range(coarse_dim):
                vv = fine_MM*funs[ii].vector()
                HH[ii,ii] = funs[ii].vector().inner(vv)
                for jj in range(ii+1,coarse_dim):
                    HH[ii,jj] = funs[jj].vector().inner(vv)
                    HH[jj,ii] = HH[ii,jj]
            TT = np.zeros((coarse_dim,extended_harmonic_dim))
            for ii in range(coarse_dim):
                for jj in range(extended_harmonic_dim):
                    TT[ii,jj] = funs[ii].vector().inner(fine_MP_harmonic[jj])
            Mtilde = interior_MM_harmonic - TT.T.dot(la.solve(HH, TT))
            eigvals = la.eigvalsh(Mtilde, extended_KK_harmonic)
            supinfs0[key].append(sqrt(eigvals[-1]))

            string += '            Solving with coarse shape functions\n'
            string += '            '
            for kk in range(num_pairs):
                string += str(kk)+(',' if kk < num_pairs-1 else '.')

                stiffness = np.zeros((coarse_dim,coarse_dim))
                rhs = np.zeros(coarse_dim)
                KK = nitsche_KKs[kk]
                for ii in range(coarse_dim):
                    stiffness[ii,ii] = funs[ii].vector().inner(KK*funs[ii].vector())
                    for jj in range(ii+1,coarse_dim):
                        stiffness[ii,jj] = funs[ii].vector().inner(KK*funs[jj].vector())
                        stiffness[jj,ii] = stiffness[ii,jj]
                LL = nitsche_LLs[kk]

                for ii in range(coarse_dim):
                    rhs[ii] = funs[ii].vector().inner(LL)

                solution = Function(fine_VV, name='u')
                coefficients = la.solve(stiffness, rhs)
                for ii in range(coarse_dim):
                    solution.vector().axpy(coefficients[ii], funs[ii].vector())

                solution_vectors[key][kk].append(solution.vector().array())
            string += '\n        done\n'
        string += '    done\n'
        print(string)
        supinfs0_q.put(supinfs0)
        supinfsE_q.put(supinfsE)
        solutions_q.put(solution_vectors)
    compute_processes = []
    supinfs0_q = multiprocessing.Queue()
    supinfsE_q = multiprocessing.Queue()
    solutions_q = multiprocessing.Queue()
    for key in keys:
        proc = multiprocessing.Process(target=compute_stuff_for_key, args=(key,supinfs0_q,supinfsE_q,solutions_q))
        compute_processes.append(proc)
        proc.start()
    print('processes started')
    for ii in range(numkeys):
        supinfs0.update(supinfs0_q.get())
        supinfsE.update(supinfsE_q.get())
        solution_vectors.update(solutions_q.get())
    for proc in compute_processes:
        proc.join()
    print('Waiting for join')
    for key in keys:
        for kk in range(num_pairs):
            for ii in range(len(solution_vectors[key][kk])):
                solutions[kk][key].append(Function(fine_VV,name='u'))
                solutions[kk][key][-1].vector()[:] = solution_vectors[key][kk][ii]

    print('Computations finished, plotting and writeout')

    errors = [dict() for kk in range(num_pairs)]
    l2_rates = [dict() for kk in range(num_pairs)]
    h1_rates = [dict() for kk in range(num_pairs)]
    supinfsE_rates = dict()
    supinfs0_rates = dict()
    supinfsE_compute_rates = dict()
    supinfs0_compute_rates = dict()
    for key in keys:
        print('    '+key)
        basename = prefix+'_'+key
        supinfE = supinfsE[key]
        supinf0 = supinfs0[key]
        dof = dofs[key]

        if_eigen = key in keys_filtered

        denom = 1./(np.log(dof[-1])-np.log(dof[0]))
        suprr1 = (np.log(supinfE[-1])-np.log(supinfE[0]))*denom
        supinfsE_rates[key] = suprr1
        suprr2 = (np.log(supinf0[-1])-np.log(supinf0[0]))*denom
        supinfs0_rates[key] = suprr2
        time_direct_r = (np.log(time_direct[-1])-np.log(time_direct[0]))*denom

        matfile = open(outdir+'/'+basename+'_supinfs.csv', 'w')
        matfile.write('dof, supinfE, lambda1, supinfEr, supinf0, lambda2, supinf0r\n')
        matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}\n'.format(fine_dim, 0, 0, suprr1, 0, 0, suprr2))
        matfile.write('{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}\n'.format(dof[0], supinfE[0], supinfE[0]/nwidths1[dof[0]], 0, supinf0[0], supinf0[0]/nwidths2[dof[0]]))
        for ii in range(len(dof)):
            denom = 1./np.log(1.*dof[ii]/dof[ii-1])
            suprr1=ln(supinfE[ii]/supinfE[ii-1])*denom
            suprr2=ln(supinf0[ii]/supinf0[ii-1])*denom
            matfile.write("{:d}, {:.3e}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3f}\n".format(dof[ii], supinfE[ii], supinfE[ii]/nwidths1[dof[ii]], suprr1, supinf0[ii], supinf0[ii]/nwidths2[dof[ii]], suprr2))
        matfile.close()

        for kk in range(num_pairs):
            sol = solutions[kk][key]
            fine_uu = fine_solutions[kk]
            fine_l2 = fine_l2s[kk]
            fine_h1 = fine_h1s[kk]
            this_errors = []
            for ii in range(len(sol)):
                l2, h1 = fine_error(fine_uu.vector(), sol[ii].vector())
                this_errors.append([l2/fine_l2, h1/fine_h1])
            this_errors = np.array(this_errors)
            errors[kk][key] = this_errors
    
            l2rr = (np.log(this_errors[-1,0])-np.log(this_errors[0,0]))/(np.log(dof[-1])-np.log(dof[0]))
            l2_rates[kk][key] = l2rr
            h1rr = (np.log(this_errors[-1,1])-np.log(this_errors[0,1]))/(np.log(dof[-1])-np.log(dof[0]))
            h1_rates[kk][key] = h1rr

            matfile = open(outdir+'/'+prefix+'_'+str(kk)+'_'+key+'.csv', 'w')
            matfile.write('dof, l2, l2r, h1, h1r \n')
            matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n".format(fine_dim, 0, l2rr, 0, h1rr))
            matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n".format(dof[0], this_errors[0,0], 0, this_errors[0,1], 0))
            for ii in range(1, len(sol)):
                denom = 1./np.log(1.*dof[ii]/dof[ii-1])
                l2rr=ln(this_errors[ii,0]/this_errors[ii-1,0])*denom
                h1rr=ln(this_errors[ii,1]/this_errors[ii-1,1])*denom
                suprr1=ln(supinfE[ii]/supinfE[ii-1])*denom
                suprr2=ln(supinf0[ii]/supinf0[ii-1])*denom
                matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}\n".format(dof[ii], this_errors[ii,0], l2rr, this_errors[ii,1], h1rr))
            matfile.close()
        print('            done')
    print('        Computations done, starting plotting')

    xfac = np.exp(ploteps*np.log(max_dof/min_dof))
    minx = min_dof/xfac; maxx = max_dof*xfac

    if matplotlib:
        doflimits = np.array([min_dof,np.sqrt(min_dof*max_dof), max_dof], dtype=float)
        dofrange=np.arange(min_dof,max_dof+1,dtype=int)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths1[dofrange]);
        for ii in range(len(ns)):
            supinf = n1[ii]
            handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf); yfac = np.exp(ploteps*np.log(ymax/ymin))
            miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
        yfac=np.exp(ploteps*np.log(maxy/miny))
        miny0 = np.min([supinfsE[key][0] for key in keys]); maxy0 = np.max([supinfsE[key][0] for key in keys])
        minr = np.min([supinfsE_rates[key] for key in keys]); maxr = np.max([supinfsE_rates[key] for key in keys])
        minlog0 = miny0/(min_dof**minr*yfac); maxlog0 = (yfac*maxy0)/min_dof**maxr
        handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
        handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label='$c n^{'+str("%.2g"%maxr)+r'}$'))
        handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'k-', mfc='none', label='$d^E_n$'))
        ax.grid(True, which='major')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
        ax.set_ylabel(r'$\Psi^E(V_l)$')
        fig.savefig(outdir+'/'+prefix+'_supinfsE_robustness.pdf')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths1[dofrange]);
        for key in keys:
            supinf = supinfsE[key]
            rr = supinfsE_rates[key]
            handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf); yfac = np.exp(ploteps*np.log(ymax/ymin))
            miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
        yfac=np.exp(ploteps*np.log(maxy/miny))
        miny0 = np.min([supinfsE[key][0] for key in keys]); maxy0 = np.max([supinfsE[key][0] for key in keys])
        minr = np.min([supinfsE_rates[key] for key in keys]); maxr = np.max([supinfsE_rates[key] for key in keys])
        minlog0 = miny0/(min_dof**minr*yfac); maxlog0 = (yfac*maxy0)/min_dof**maxr
        handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
        handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label='$c n^{'+str("%.2g"%maxr)+r'}$'))
        handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'k-', mfc='none', label='$d^E_n$'))
        ax.grid(True, which='major')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
        ax.set_ylabel(r'$\Psi^E(V_l)$')
        fig.savefig(outdir+'/'+prefix+'_supinfsE.pdf')

        figlegend = plt.figure(figsize=(np.ceil((numkeys+3)/2.)*legendx*1.05,2*legendy),frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/2.)))
        figlegend.savefig(outdir+'/'+prefix+'_supinfsE_legend.pdf',bbox_extra_artists=(lgd,))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = 0.9; maxy = 2.;
        for key in keys:
            supinf = supinfsE[key]
            n1 = nwidths1[dofs[key]]
            handles.append(*ax.loglog(dofs[key], supinf/n1, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n1); ymax = np.max(supinf/n1); yfac = np.exp(ploteps*np.log(ymax/ymin))
            miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
        ax.grid(True, which='major')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
        ax.set_ylabel(r'$\Lambda^1(V_l)$')
        fig.savefig(outdir+'/'+prefix+'_lambda1.pdf')

        figlegend = plt.figure(figsize=(1.05*np.ceil(numkeys/2)*legendx,2*legendy),frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil(numkeys/2)))
        figlegend.savefig(outdir+'/'+prefix+'_legend.pdf',bbox_extra_artists=(lgd,))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = np.min(nwidths2[dofrange]); maxy = np.max(nwidths2[dofrange]);
        for key in keys:
            supinf = supinfs0[key]
            handles.append(*ax.loglog(dofs[key], supinf, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf); ymax = np.max(supinf); yfac = np.exp(ploteps*np.log(ymax/ymin))
            miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
        yfac=np.exp(ploteps*np.log(maxy/miny))
        miny0 = np.min([supinfs0[key][0] for key in keys]); maxy0 = np.max([supinfs0[key][0] for key in keys])
        minr = np.min([supinfs0_rates[key] for key in keys]); maxr = np.max([supinfs0_rates[key] for key in keys])
        minlog0 = miny0/(min_dof**minr*yfac); maxlog0 = (yfac*maxy0)/min_dof**maxr
        handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
        handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label='$c n^{'+str("%.2g"%maxr)+r'}$'))
        handles.append(*ax.loglog(dofrange, nwidths2[dofrange], 'k-', mfc='none', label='$d^0_n$'))
        ax.grid(True, which='major')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
        ax.set_ylabel(r'$\Psi^0(V_l)$')
        fig.savefig(outdir+'/'+prefix+'_supinfs0.pdf')

        figlegend = plt.figure(figsize=(np.ceil((numkeys+3)/2.)*legendx*1.05,2*legendy),frameon=False)
        labels = [hh.get_label() for hh in handles]
        ax = figlegend.add_subplot(111)
        ax.axis('off')
        lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/2.)))
        figlegend.savefig(outdir+'/'+prefix+'_supinfs0_legend.pdf',bbox_extra_artists=(lgd,))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        handles = []
        miny = .9; maxy = 2.;
        for key in keys:
            supinf = supinfs0[key]
            n2 = nwidths2[dofs[key]]
            handles.append(*ax.loglog(dofs[key], supinf/n2, plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            ymin = np.min(supinf/n2); ymax = np.max(supinf/n2); yfac = np.exp(ploteps*np.log(ymax/ymin))
            miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
        ax.grid(True, which='major')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        sa_utils.set_log_ticks(ax, minx, maxx, True)
        sa_utils.set_log_ticks(ax, miny, maxy)
        ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
        ax.set_ylabel(r'$\Lambda^2(V_l)$')
        fig.savefig(outdir+'/'+prefix+'_lambda2.pdf')

        plt.close('all')

        for key in keys:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'r-', mfc='none', label='$d^E_n$'))
            handles.append(*ax.loglog(dofrange, nwidths2[dofrange], 'g-', mfc='none', label='$d^0_n$'))
            miny = np.min([np.min(nwidths1[dofrange]),np.min(nwidths2[dofrange]),\
                           np.min(supinfsE[key]), np.min(supinfs0[key])])
            maxy = np.max([np.max(nwidths1[max_dof:max_dof+1]),np.max(nwidths2[max_dof:max_dof+1]),\
                           np.max(supinfsE[key]), np.max(supinfs0[key])])
            yfac = np.exp(ploteps*np.log(maxy/miny)); miny /= yfac; maxy *= yfac
            handles.append(*ax.loglog(dofs[key], supinfsE[key], 'ro', mec='r', mfc='none', label=r'$\Psi^E(V_l)$'))
            handles.append(*ax.loglog(dofs[key], (supinfsE[key][0]/dofs[key][0]**supinfsE_rates[key])*dofs[key]**supinfsE_rates[key], 'r:', label = r'$c n^{'+str("%.2g"%supinfsE_rates[key])+r'}$'))
            handles.append(*ax.loglog(dofs[key], supinfs0[key], 'g+', mec='g', mfc='none', label=r'$\Psi^0(V_l)$'))
            handles.append(*ax.loglog(dofs[key], (supinfs0[key][0]/dofs[key][0]**supinfs0_rates[key])*dofs[key]**supinfs0_rates[key], 'g--', label = r'$c n^{'+str("%.2g"%supinfs0_rates[key])+r'}$'))
            ax.grid(True, which='major')
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.set_title(names[key])
            ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
            ax.set_ylabel(r'$\Psi^i(V_l)$')
            fig.savefig(outdir+'/'+prefix+'_'+str(key)+'_supinfs.pdf')
            
            figlegend = plt.figure(figsize=(3.5*legendx,2*legendy),frameon=False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc=10, ncol=3)
            figlegend.savefig(outdir+'/'+prefix+'_'+str(key)+'_supinfs_legend.pdf',bbox_extra_artists=(lgd,))
 
            plt.close('all')

        for kk in range(num_pairs):
            fine_l2 = fine_l2s[kk]
            fine_h1 = fine_h1s[kk]
            mul = fine_h1/fine_l2

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths2[dofrange]*mul); maxy = np.max(nwidths2[dofrange]*mul);
            for key in keys:
                err = errors[kk][key]
                ymin = np.min(err[:,0]); ymax = np.max(err[:,0]); yfac = np.exp(ploteps*np.log(ymax/ymin))
                miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
                handles.append(*ax.loglog(dofs[key], err[:,0], plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            yfac = np.exp(ploteps*np.log(maxy/miny))
            miny0 = np.min([errors[kk][key][0,0] for key in keys]); maxy0 = np.max([errors[kk][key][0,0] for key in keys])
            minr = np.min([l2_rates[kk][key] for key in keys]); maxr = np.max([l2_rates[kk][key] for key in keys])
            minlog0 = miny0/(min_dof**minr*yfac); maxlog0 = (yfac*maxy0)/min_dof**maxr
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label=r'$c n^{'+str("%.2g"%maxr)+r'}$'))
            handles.append(*ax.loglog(dofrange, nwidths2[dofrange]*mul, 'k-', mfc='none', label=r'$d^0_n$'))
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.grid(True,which='major')
            ax.set_xlabel(r'$n = \operatorname{\dim} V_l$')
            ax.set_ylabel(r'$e_2$')
            fig.savefig(outdir+'/'+prefix+'_'+str(kk)+'_l2.pdf')

            figlegend = plt.figure(figsize=(1.05*np.ceil((numkeys+3)/2.)*legendx,2*legendy),frameon=False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/2.)))
            figlegend.savefig(outdir+'/'+prefix+'_'+str(kk)+'_l2_legend.pdf',bbox_extra_artists=(lgd,))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            handles = []
            miny = np.min(nwidths1[dofrange]); maxy = np.max(nwidths1[dofrange]);
            for key in keys:
                err = errors[kk][key]
                ymin = np.min(err[:,1]); ymax = np.max(err[:,1]); yfac = np.exp(ploteps*np.log(ymax/ymin))
                miny = np.min([ymin/yfac, miny]); maxy = np.max([ymax*yfac, maxy])
                handles.append(*ax.loglog(dofs[key], err[:,1], plotmarkers[key], mec=plotcolors[key], mfc='none', label=names[key]))
            yfac = np.exp(ploteps*np.log(maxy/miny))
            miny0 = np.min([errors[kk][key][0,1] for key in keys]); maxy0 = np.max([errors[kk][key][0,1] for key in keys])
            minr = np.min([h1_rates[kk][key] for key in keys]); maxr = np.max([h1_rates[kk][key] for key in keys])
            minlog0 = miny0/(min_dof**minr*yfac); maxlog0 = (yfac*maxy0)/min_dof**maxr
            handles.append(*ax.loglog(doflimits, minlog0*doflimits**minr, 'k:', label=r'$c n^{'+str("%.2g"%minr)+r'}$'))
            handles.append(*ax.loglog(doflimits, maxlog0*doflimits**maxr, 'k--', label='$c n^{'+str("%.2g"%maxr)+r'}$'))
            handles.append(*ax.loglog(dofrange, nwidths1[dofrange], 'k-', mfc='none', label=r'$d^E_n$'))
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            sa_utils.set_log_ticks(ax, minx, maxx, True)
            sa_utils.set_log_ticks(ax, miny, maxy)
            ax.grid(True, which='major')
            ax.set_xlabel(r'$n = \operatorname{dim} V_l$')
            ax.set_ylabel(r'$e_E$')
            fig.savefig(outdir+'/'+prefix+'_'+str(kk)+'_energy.pdf')

            figlegend = plt.figure(figsize=(1.05*np.ceil((numkeys+3)/2.)*legendx,2*legendy),frameon=False)
            labels = [hh.get_label() for hh in handles]
            ax = figlegend.add_subplot(111)
            ax.axis('off')
            lgd = ax.legend(handles, labels, loc=10, ncol=int(np.ceil((numkeys+3)/2.)))
            figlegend.savefig(outdir+'/'+prefix+'_'+str(kk)+'_energy_legend.pdf',bbox_extra_artists=(lgd,))

            plt.close('all')
    
if __name__ == '__main__':
    uus = read_shapes_from_xdmf('enrichments/bar/heat/paraview/bar_9_shapes_polynomial_harmonic')

    meshdir = 'fem/bar'
    prefix = 'bar_9'
    box = np.array([-1e-1,-1e-2,1e-1,1e-2])

    mesh, mesh_regions, mesh_facets = sa_hdf5.read_xdmf(meshdir+'/'+prefix)
    extended_mesh, extended_mesh_regions, extended_mesh_facets = sa_hdf5.read_xdmf(meshdir+'/'+prefix+'_extended')

    bla = dict()
    bla['polynomial_harmonic_9'] = uus

    robustness_test(uus, mesh, mesh_regions, mesh_facets, extended_mesh, extended_mesh_regions, extended_mesh_facets)
