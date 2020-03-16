from dolfin import *
#from shapes import *

import numpy as np
import matplotlib.pyplot as plt

import sa_utils
from sa_utils import comm, rank, size

import sa_hdf5
import create_patches

import os, time, gc, multiprocessing

#PETScOptions.set('mat_mumps_icntl_14', 100)

parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

direct_params = ('pastix', )
krylov_params = ('cg', 'hypre_amg')


#ortho: [[E_1, G_12, G_13], [mu_12, E_2, G_23], [mu_13, mu_23, E_3]]

def run_fenics_example(prefix, myrange, elasticity = False, meshes = None, domains = None, facets = None, 
                       mesh_suffices = None, contrasts = [1e4], patch_bc = None, ldomain = False, logger = None, 
                       bc_case = None, direct = False, krylov = True, values_array = None,
                       kappa_is_EE = True, kappa_is_poisson = False, poisson = 0.3, EE = 1e11,
                       debug = False, orthotropic = False, ortho_axis = [0, 0, 1],
                       ortho_params = [[1.5e7, 7.5e5, 7.5e5], [0.3, 1e6, 1e6/(2*(1+0.44))], [0.3, 0.44, 1e6]]):
    if elasticity and not orthotropic:
        assert(bool(kappa_is_EE) ^ bool(kappa_is_poisson))
    if orthotropic:
        elasticity = True
    if not krylov and not direct:
        return
    if meshes is None or len(meshes) < 1:
        logger.info('Loading meshes')
        meshes, domains, facets = create_patches.load_meshes_h5py(prefix, prefix, myrange)

    basedim = meshes[-1].geometry().dim() 
    all_bcs = sa_utils.get_bcs(basedim, elasticity, reference = True, ldomain = ldomain)
    num_bcs = len(all_bcs)

    num_bc_cases = num_bcs*len(contrasts)
    if bc_case and bc_case >= num_bc_cases:
        logger.info('bc case limit reached')
        return 1

    if mesh_suffices is None:
        mesh_suffices = [mesh.num_vertices() for mesh in meshes]

    # Define Dirichlet boundary conditions
    logger.info('Setup')
    # Create mesh and define function space
    fine_mesh = meshes[-1]
    fine_domain = domains[-1]
    fine_facet = facets[-1]
    help_array = np.asarray(fine_domain.array(), dtype = int)
    fine_V0 = FunctionSpace(fine_mesh, 'DG', 0)

    if elasticity or orthotropic:
        if basedim == 2:
            null_expressions = [Constant((1, 0)), 
                                Constant((0, 1)), 
                                Expression(('-x[1]', 'x[0]'), degree = 1)]
        elif basedim == 3:
            null_expressions = [Constant((1, 0, 0)), 
                                Constant((0, 1, 0)), 
                                Constant((0, 0, 1)), 
                                Expression(('0', '-x[2]', 'x[1]'), degree = 1), 
                                Expression(('x[2]', '0', '-x[0]'), degree = 1), 
                                Expression(('-x[1]', 'x[0]', '0'), degree = 1)]
        else:
            raise NameError('Dimension ['+str(basedim)+'] null space not available')

        if orthotropic:
            base_dir = prefix+'/orthotropic'
        else:
            base_dir = prefix+'/elasticity'
        
        write_functions = sa_hdf5.write_dolfin_vector_cg1

        zero = Constant([0]*basedim)
        ones = Constant([1]*basedim)
    else:
        null_expressions = [Constant(1)]
        base_dir = prefix+'/heat'

        write_functions = sa_hdf5.write_dolfin_scalar_cg1

        zero = Constant(0)
        ones = Constant(1)

    null_dim = len(null_expressions)

    sa_utils.makedirs_norace(base_dir)
    
    l2_form = lambda uu, vv, mesh: inner(uu, vv)*dx(mesh)
    h1_form = lambda uu, vv, mesh: inner(grad(uu), grad(vv))*dx(mesh)

    if values_array is None:
        values_array = []
        for contrast in contrasts:
            values_array.append([1., contrast])
    count = 0
    for ct, kappa_values in enumerate(values_array):
        logger.info('Case [{:d}/{:d}]'.format(ct+1, len(values_array)))
        cc = []
        tmp = []
        for ii in range(len(kappa_values)):
            cc.append(kappa_values[ii])
            tmp.append(kappa_values[ii])
            if ii % 2:
                cc += tmp
                tmp = []
        help_values = np.unique(help_array)
        fine_kappa = Function(fine_V0, name = 'kappa')
        tmp = np.zeros_like(help_array, dtype=float)
        for val in help_values:
            tmp[np.where(help_array == val)] = cc[val]
        fine_kappa.vector().set_local(tmp)
        contrast = np.max(tmp)/np.min(tmp)
        del tmp, cc
        if debug:
            mf = MeshFunction('double', fine_mesh, 3, 0)
            mf.array()[:] = fine_kappa.vector().array()
            File('{:s}/debug/{:s}_kappa_{:d}.pvd'.format(prefix, prefix, ct)) << mf
#       XDMFFile(fine_mesh.mpi_comm(), '{:s}/{:s}_kappa_{:.2e}.xdmf'.format(prefix, prefix, contrast) ).write(fine_kappa)

        contrast_dir = '{:s}/case{:d}_{:.2e}'.format(base_dir, ct, contrast)
        sa_utils.makedirs_norace(contrast_dir)
        coeff_dir = '{:s}/coeffs'.format(contrast_dir)
        sa_utils.makedirs_norace(coeff_dir)
        if orthotropic:
            epsilon = lambda uu : sym(grad(uu))
            
            ortho_cos = cos(fine_kappa)
            ortho_sin = sin(fine_kappa)
            RR = as_matrix([
                [ortho_cos + ortho_axis[0]*ortho_axis[0]*(1 - ortho_cos), ortho_axis[0]*ortho_axis[1]*(1 - ortho_cos) - ortho_axis[2]*ortho_sin, ortho_axis[0]*ortho_axis[2]*(1 - ortho_cos) + ortho_axis[1]*ortho_sin],
                [ortho_axis[0]*ortho_axis[1]*(1 - ortho_cos) + ortho_axis[2]*ortho_sin, ortho_cos + ortho_axis[1]*ortho_axis[1]*(1 - ortho_cos), ortho_axis[1]*ortho_axis[2]*(1 - ortho_cos) - ortho_axis[0]*ortho_sin],
                [ortho_axis[0]*ortho_axis[2]*(1 - ortho_cos) - ortho_axis[1]*ortho_sin, ortho_axis[1]*ortho_axis[2]*(1 - ortho_cos) + ortho_axis[0]*ortho_sin, ortho_cos + ortho_axis[2]*ortho_axis[2]*(1 - ortho_cos)]
            ])

            nu_xy = ortho_params[1][0]
            nu_xz = ortho_params[2][0]
            nu_yz = ortho_params[2][1]
            EE = [ortho_params[0][0], ortho_params[1][1], ortho_params[2][2]]
            nu_yx = nu_xy*EE[1]/EE[0]
            nu_zx = nu_xz*EE[2]/EE[0]
            nu_zy = nu_yz*EE[2]/EE[1]
            kk = 1 - nu_xy*nu_yx - nu_yz*nu_zy - nu_xz*nu_zx - 2*nu_xy*nu_yz*nu_zx
            DD_00 = EE[0]*(1 - nu_yz*nu_zy)/kk; DD_01 = EE[0]*(nu_yz*nu_zx + nu_yx)/kk; DD_02 = EE[0]*(nu_zy*nu_yx + nu_zx)/kk
            DD_10 = EE[1]*(nu_xz*nu_zy + nu_xy)/kk; DD_11 = EE[1]*(1 - nu_xz*nu_zx)/kk; DD_12 = EE[1]*(nu_zx*nu_xy + nu_zy)/kk
            DD_20 = EE[2]*(nu_xy*nu_yz + nu_xz)/kk; DD_21 = EE[2]*(nu_yx*nu_xz + nu_yz)/kk; DD_22 = EE[2]*(1 - nu_xy*nu_yx)/kk
            del EE, kk, nu_xy, nu_xz, nu_yz, nu_yx, nu_zx, nu_zy

            def sigma(uu):
                eps = RR*epsilon(uu)*RR.T
                ss_00 = DD_00*eps[0,0]+DD_01*eps[1,1]+DD_02*eps[2,2]
                ss_11 = DD_10*eps[0,0]+DD_11*eps[1,1]+DD_12*eps[2,2]
                ss_22 = DD_20*eps[0,0]+DD_21*eps[1,1]+DD_22*eps[2,2]
                ss_01 = 2*ortho_params[0][1]*eps[0,1]
                ss_02 = 2*ortho_params[0][2]*eps[0,2]
                ss_12 = 2*ortho_params[1][2]*eps[1,2]
                return RR.T*as_matrix([[ss_00, ss_01, ss_02], [ss_01, ss_11, ss_12], [ss_02, ss_12, ss_22]])*RR

        elif elasticity:
            if kappa_is_EE:
                EE = fine_kappa
            elif kappa_is_poisson:
                poisson = fine_kappa
            ll = (poisson*EE)/((1.+poisson)*(1.-2.*poisson))
            mu = EE/(2.*(1.+poisson))

            epsilon = lambda uu: sym(grad(uu))
            sigma = lambda uu: 2.*mu*epsilon(uu)+ll*tr(epsilon(uu))*Identity(basedim)
        else:
            epsilon = lambda uu: grad(uu)
            sigma = lambda uu: fine_kappa*epsilon(uu)
        kk_form = lambda uu, vv, mesh: inner(sigma(uu), epsilon(vv))*dx(mesh)
            
        logger.info('Computing solutions')
        for num_bc, (dirichlet, neumann, ff) in enumerate(all_bcs):
            count += 1
            if bc_case is not None and count != bc_case+1:
                continue
            logger.info('Contrast {:.2e}, case [{:d}/{:d}], dirichlet {:s}, neumann {:s}, load {:s}'.format(contrast, num_bc+1, num_bcs, str([domain for (bc, domain) in dirichlet]), 
                                                                                                            str([domain for (bc, domain) in neumann]), str(not ff is None)))
            uus_direct = []
            uus_krylov = []
            ns = []
            times_assemble_direct = []
            times_assemble_krylov = []
            times_solve_direct = []
            times_solve_krylov = []
            iterations = []

            def solve_for(nn, last):
                logger.info('    Resolution ['+str(nn)+'] start')
                mesh = meshes[nn]
                domain = domains[nn]
                facet = facets[nn]
                coarse_ds = Measure('ds', domain = mesh, subdomain_data = facet)


                if orthotropic:
                    basename = prefix+'_'+str(mesh_suffices[nn])+'_orthotropic_{:.2e}_'.format(contrast)+str(num_bc)
                elif elasticity:
                    basename = prefix+'_'+str(mesh_suffices[nn])+'_elasticity_{:.2e}_'.format(contrast)+str(num_bc)
                else:
                    basename = prefix+'_'+str(mesh_suffices[nn])+'_heat_{:.2e}_'.format(contrast)+str(num_bc)

                logger.info('        forms')
                if elasticity:
                    VV = VectorFunctionSpace(mesh, 'CG', 1, basedim)
                else:
                    VV = FunctionSpace(mesh, 'CG', 1)

                null_space = sa_utils.build_nullspace(VV, elasticity = elasticity)

                trial = TrialFunction(VV)
                test = TestFunction(VV)
                
                kk = kk_form(trial, test, mesh)
                if ff is None:
                    LL = l2_form(zero, test, mesh)
                else:
                    LL = l2_form(ff, test, mesh)

                coarse_bcs = [DirichletBC(VV, bc, facet, domain) for (bc, domain) in dirichlet]

                uu = Function(VV, name = 'u')

                ns.append(VV.dim())

                LL += sum(inner(bc, test)*coarse_ds(int(ii)) for (bc, ii) in neumann)

                logger.info('        assembly')
                # Solution
                AA, bb = assemble_system(kk, LL, coarse_bcs)
                old = sa_utils.get_time()
                AA, bb = assemble_system(kk, LL, coarse_bcs)
                if not dirichlet:
                    as_backend_type(AA).set_nullspace(null_space)
                    null_space.orthogonalize(bb)
                new = sa_utils.get_time()
                times_assemble_krylov.append(new-old)
                time_assemble_direct = times_assemble_krylov[-1]
                logger.info('        solve')
                if krylov:
                    logger.info('            krylov')
                    print('krylov solve starting')
                    try:
                        old = sa_utils.get_time()
                        it = sa_utils.krylov_solve(AA, uu.vector(), bb, *krylov_params)
                        new = sa_utils.get_time()
                    except:
                        it = -1
                        uus_krylov.append(None)
                        iterations.append(float('NaN'))
                        times_solve_krylov.append(float('NaN'))
                    else:
                        write_functions(coeff_dir+'/krylov_'+basename, [uu])
                        uus_krylov.append(uu)
                        iterations.append(it)
                        times_solve_krylov.append(new-old)
                    logger.info('            krylov [{:d}] iterations done in [{:s}+{:s} = {:s}]'.format(it, sa_utils.human_time(times_assemble_krylov[-1]), 
                                                                                              sa_utils.human_time(times_solve_krylov[-1]), 
                                                                                              sa_utils.human_time(times_assemble_krylov[-1]+times_solve_krylov[-1])))
                    print('krylov solve done')
                if direct:
                    logger.info('            direct')
                    print('direct solve starting')
                    if not dirichlet:
                        if elasticity:
                            WV = VectorElement('CG', mesh.ufl_cell(), 1, basedim)
                        else:
                            WV = FiniteElement('CG', mesh.ufl_cell(), 1)
                        WR = VectorElement('R', mesh.ufl_cell(), 0, null_dim)
                        WW = FunctionSpace(mesh, WV*WR)

                        trial, cw = TrialFunctions(WW)
                        test, dw = TestFunctions(WW)
                        kk = kk_form(trial, test, mesh)-\
                             sum(cw[ii]*inner(test, null_expressions[ii])*dx for ii in range(null_dim))-\
                             sum(dw[ii]*inner(trial, null_expressions[ii])*dx for ii in range(null_dim))

                        if ff is None:
                            LL = l2_form(zero, test, mesh)
                        else:
                            LL = l2_form(ff, test, mesh)

                        LL += sum(inner(bc, test)*coarse_ds(int(ii)) for (bc, ii) in neumann)
                        
                        AA, bb = assemble_system(kk, LL, None)
                        old = sa_utils.get_time()
                        AA, bb = assemble_system(kk, LL, None)
                        new = sa_utils.get_time()
                        time_assemble_direct = new-old
                        uu = Function(WW, name = 'u')
                    times_assemble_direct.append(time_assemble_direct)
                    old = sa_utils.get_time()
                    solve(AA, uu.vector(), bb, *direct_params)
                    new = sa_utils.get_time()
                    times_solve_direct.append(new-old)
                    logger.info('            direct solve done in [{:s}+{:s} = {:s}]'.format(sa_utils.human_time(times_assemble_direct[-1]), 
                                                                                       sa_utils.human_time(times_solve_direct[-1]), 
                                                                                       sa_utils.human_time(times_assemble_direct[-1]+times_solve_direct[-1])))
                    print('direct_solve_done')
                    if not dirichlet:
                        vv = uu
                        uu = Function(VV, name = 'u')
                        assign(uu, vv.sub(0))
                    uus_direct.append(uu)
                    write_functions(coeff_dir+'/direct_'+basename, [uu])

                logger.info('        writeout finished')
                
                del AA, bb, old, new, LL, kk
                del uu, test, trial, basename
                gc.collect()
                logger.info('    Resolution ['+str(nn)+'] end')

                if last:
                    trial = TrialFunction(VV)
                    test = TestFunction(VV)
                    L2 = PETScMatrix()
                    assemble(l2_form(trial, test, mesh), tensor = L2)
                    H1 = PETScMatrix()
                    assemble(h1_form(trial, test, mesh), tensor = H1)
                    KK = PETScMatrix()
                    assemble(kk_form(trial, test, mesh), tensor = KK)

                    return VV, L2, H1, KK
                else:
                    return None, None, None, None

            for kk in range(len(meshes)):
                logger.info('{:d}/{:d}'.format(kk+1, len(meshes)))
                VV, L2, H1, KK = solve_for(kk, kk == len(meshes)-1)
            fine_dim = VV.dim()

            all_ns = np.array(ns)
            ns = all_ns[:-1]
            logger.info('Solutions obtained')
            print('Solutions obtained')

            def ref_inner_sqrts(uu):
                return sqrt(uu.inner(L2*uu)), sqrt(uu.inner(H1*uu)), sqrt(uu.inner(KK*uu))

            def ref_norms(uu):
                return ref_inner_sqrts(uu.vector())

            if direct:
                fine_uu_direct = uus_direct[-1]
                fine_l2_direct, fine_h1_direct, fine_energy_direct = ref_norms(fine_uu_direct)

                times_assemble_direct = np.array(times_assemble_direct)
                times_solve_direct = np.array(times_solve_direct)
                times_total_direct = times_assemble_direct+times_solve_direct

                def ref_errors_direct(uu):
                    vv = interpolate(uu, VV)
                    diff = vv.vector()-fine_uu_direct.vector()
                    return ref_inner_sqrts(diff)

            if krylov:
                fine_uu_krylov = uus_krylov[-1]
                fine_l2_krylov, fine_h1_krylov, fine_energy_krylov = ref_norms(fine_uu_krylov)

                times_assemble_krylov = np.array(times_assemble_krylov)
                times_solve_krylov = np.array(times_solve_krylov)
                times_total_krylov = times_assemble_krylov+times_solve_krylov

                def ref_errors_krylov(uu):
                    vv = interpolate(uu, VV)
                    diff = vv.vector()-fine_uu_krylov.vector()
                    return ref_inner_sqrts(diff)

            if len(meshes) < 3:
                continue
            
            logger.info('Calculating errors')
            if direct:
                l2_errors_direct = []
                h1_errors_direct = []
                energy_errors_direct = []
                for kk in range(len(uus_direct)-1):
                    l2, h1, energy = ref_errors_direct(uus_direct[kk])
                    l2_errors_direct.append(l2/fine_l2_direct)
                    h1_errors_direct.append(h1/fine_h1_direct)
                    energy_errors_direct.append(energy/fine_energy_direct)
                
                l2_errors_direct = np.array(l2_errors_direct)
                h1_errors_direct = np.array(h1_errors_direct)
                del fine_uu_direct

            if krylov:
                l2_errors_krylov = []
                h1_errors_krylov = []
                energy_errors_krylov = []
                for kk in range(len(uus_krylov)-1):
                    l2, h1, energy = ref_errors_krylov(uus_krylov[kk])
                    l2_errors_krylov.append(l2/fine_l2_krylov)
                    h1_errors_krylov.append(h1/fine_h1_krylov)
                    energy_errors_krylov.append(energy/fine_energy_krylov)
                
                l2_errors_krylov = np.array(l2_errors_krylov)
                h1_errors_krylov = np.array(h1_errors_krylov)
                del fine_uu_krylov

            del VV, uus_direct, uus_krylov, L2, H1, KK
            gc.collect()

            logger.info('Obtaining rates, plotting etc')
            if rank:
                continue

            if orthotropic:
                basename = '{:s}_orthotropic_{:.2e}_{:d}'.format(prefix, contrast, num_bc)
            elif elasticity:
                basename = '{:s}_elasticity_{:.2e}_{:d}'.format(prefix, contrast, num_bc)
            else:
                basename = '{:s}_heat_{:.2e}_{:d}'.format(prefix, contrast, num_bc)

            if direct:
                maxerror = np.max(l2_errors_direct)
                minerror = 1e-12*maxerror
                lastidx = -1
                if l2_errors_direct[lastidx] < minerror:
                    lastidx = np.max(np.where(l2_errors_direct >= minerror)[0])
                global_l2rr = (np.log(l2_errors_direct[lastidx])-np.log(l2_errors_direct[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_h1rr = (np.log(h1_errors_direct[lastidx])-np.log(h1_errors_direct[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_energyrr = (np.log(energy_errors_direct[lastidx])-np.log(energy_errors_direct[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_assemblerr = (np.log(times_assemble_direct[-1])-np.log(times_assemble_direct[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_solverr = (np.log(times_solve_direct[-1])-np.log(times_solve_direct[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_totalrr = (np.log(times_total_direct[-1])-np.log(times_total_direct[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_l2_timerr = (np.log(l2_errors_direct[lastidx])-np.log(l2_errors_direct[0]))/(np.log(times_total_direct[lastidx])-np.log(times_total_direct[0]))
                global_h1_timerr = (np.log(h1_errors_direct[lastidx])-np.log(h1_errors_direct[0]))/(np.log(times_total_direct[lastidx])-np.log(times_total_direct[0]))

                matfile = open(contrast_dir+'/'+basename+"_direct.csv", 'w')
                matfile.write('dof, l2, l2r, h1, h1r, energy, energyr, assemble_direct, solve_direct, total_direct\n')
                matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(ns[0], l2_errors_direct[0], 0, h1_errors_direct[0], 0, energy_errors_direct[0], 0, times_assemble_direct[0], times_solve_direct[0], times_total_direct[0]))
                for ii in range(1, len(l2_errors_direct)):
                    l2rr = ln(l2_errors_direct[ii]/l2_errors_direct[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    h1rr = ln(h1_errors_direct[ii]/h1_errors_direct[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    energyrr = ln(energy_errors_direct[ii]/energy_errors_direct[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(ns[ii], l2_errors_direct[ii], l2rr, h1_errors_direct[ii], h1rr, energy_errors_direct[ii], energyrr, times_assemble_direct[ii], times_solve_direct[ii], times_total_direct[ii]))
                matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(fine_dim, 0, global_l2rr, 0, global_h1rr, 0, global_energyrr, times_assemble_direct[-1], times_solve_direct[-1], times_total_direct[-1]))
                matfile.close()

                minx = ns[0]; maxx = ns[-1]
                
                miny = np.min([np.min(energy_errors_direct), np.min(l2_errors_direct), np.min(h1_errors_direct)]); maxy = np.max([np.max(energy_errors_direct), np.max(l2_errors_direct), np.max(h1_errors_direct)])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(ns, l2_errors_direct, 'go', mec = 'g', mfc = 'none', label = r'$e_0$')
                ax.loglog(ns, (l2_errors_direct[0]/ns[0]**global_l2rr)*ns**global_l2rr, 'g-.', label = r'$c n^{'+str("%.2f"%global_l2rr)+r'}$')
                ax.loglog(ns, h1_errors_direct, 'bx', mec = 'b',  mfc = 'none', label = r'$e_1$')
                ax.loglog(ns, (h1_errors_direct[0]/ns[0]**global_h1rr)*ns**global_h1rr, 'b--', label = r'$c n^{'+str("%.2f"%global_h1rr)+r'}$')
                ax.loglog(ns, energy_errors_direct, 'r+', mec = 'r', mfc = 'none', label = r'$e_E$')
                ax.loglog(ns, (energy_errors_direct[0]/ns[0]**global_energyrr)*ns**global_energyrr, 'r:', label = r'$c n^{'+str("%.2f"%global_energyrr)+r'}$')
                ax.set_ylabel(r'$e_{\lbrace 0, 1, E\rbrace}$')
                ax.set_xlabel(r'$n = \#\operatorname{dof}$')
                sa_utils.set_log_ticks(ax, minx, maxx, True); sa_utils.set_log_ticks(ax, miny, maxy)
                fig.savefig(contrast_dir+'/'+basename+'_direct_errors.pdf')
                    
                figlegend = plt.figure(figsize = (4*sa_utils.legendx, 2*sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_direct_errors_legend.pdf', bbox_extra_artists = (lgd, ))
            
                mint = np.min([np.min(times_assemble_direct), np.min(times_solve_direct)]); maxt = np.max(times_total_direct)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(all_ns, times_assemble_direct, 'g+', mec = 'g', mfc = 'none', label = r'assembly, {:s}'.format(str(direct_params)))
                ax.loglog(all_ns, (times_assemble_direct[0]/all_ns[0]**global_assemblerr)*all_ns**global_assemblerr, 'g:', label = r'$c n^{'+str("%.2f"%global_assemblerr)+r'}$')
                ax.loglog(all_ns, times_solve_direct, 'gx', mec = 'g', mfc = 'none', label = r'solve, {:s}'.format(str(direct_params)))
                ax.loglog(all_ns, (times_solve_direct[0]/all_ns[0]**global_solverr)*all_ns**global_solverr, 'g-.', label = r'$c n^{'+str("%.2f"%global_solverr)+r'}$')
                ax.loglog(all_ns, times_total_direct, 'go', mec = 'g', mfc = 'none', label = r'total, {:s}'.format(str(direct_params)))
                ax.loglog(all_ns, (times_total_direct[0]/all_ns[0]**global_totalrr)*all_ns**global_totalrr, 'g-', label = r'$c n^{'+str("%.2f"%global_totalrr)+r'}$')
                ax.set_ylabel(r'$t$')
                ax.set_xlabel(r'$n = \#\operatorname{dof}$')
                sa_utils.set_log_ticks(ax, minx, maxx, True); sa_utils.set_log_ticks(ax, mint, maxt)
                fig.savefig(contrast_dir+'/'+basename+'_direct_times.pdf')

                figlegend = plt.figure(figsize = (4.2*sa_utils.legendx, 2*sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_direct_times_legend.pdf', bbox_extra_artists = (lgd, ))
             
                mint = np.min(times_total_direct); maxt = np.max(times_total_direct)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(times_total_direct[:-1], l2_errors_direct, 'go', mec = 'g', mfc = 'none', label = r'$e_0$')
                ax.loglog(times_total_direct[:-1], h1_errors_direct, 'bx', mec = 'b', mfc = 'none', label = r'$e_1$')
                ax.loglog(times_total_direct[:-1], energy_errors_direct, 'r+', mec = 'r', mfc = 'none', label = r'$e_E$')
                ax.set_ylabel(r'$e_{\lbrace 0, 1, E \rbrace}$')
                ax.set_xlabel(r'$t\;\operatorname{total}$')
                sa_utils.set_log_ticks(ax, mint, maxt, True); sa_utils.set_log_ticks(ax, miny, maxy)
                fig.savefig(contrast_dir+'/'+basename+'_direct_errors_vs_total_direct.pdf')

                figlegend = plt.figure(figsize = (3*sa_utils.legendx, sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_direct_errors_vs_total_legend.pdf', bbox_extra_artists = (lgd, ))

                plt.close('all')
                del fig, ax, figlegend, handles, labels, lgd, mint, maxt, minx, maxx, miny, maxy
                gc.collect()

            if krylov:
                maxerror = np.max(l2_errors_krylov)
                minerror = 1e-12*maxerror
                lastidx = -1
                if l2_errors_krylov[lastidx] < minerror:
                    lastidx = np.max(np.where(l2_errors_krylov >= minerror)[0])
                global_l2rr = (np.log(l2_errors_krylov[lastidx])-np.log(l2_errors_krylov[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_h1rr = (np.log(h1_errors_krylov[lastidx])-np.log(h1_errors_krylov[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_energyrr = (np.log(energy_errors_krylov[lastidx])-np.log(energy_errors_krylov[0]))/(np.log(ns[lastidx])-np.log(ns[0]))
                global_assemblerr = (np.log(times_assemble_krylov[-1])-np.log(times_assemble_krylov[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_solverr = (np.log(times_solve_krylov[-1])-np.log(times_solve_krylov[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_totalrr = (np.log(times_total_krylov[-1])-np.log(times_total_krylov[0]))/(np.log(all_ns[-1])-np.log(all_ns[0]))
                global_l2_timerr = (np.log(l2_errors_krylov[lastidx])-np.log(l2_errors_krylov[0]))/(np.log(times_total_krylov[lastidx])-np.log(times_total_krylov[0]))
                global_h1_timerr = (np.log(h1_errors_krylov[lastidx])-np.log(h1_errors_krylov[0]))/(np.log(times_total_krylov[lastidx])-np.log(times_total_krylov[0]))

                matfile = open(contrast_dir+'/'+basename+"_krylov.csv", 'w')
                matfile.write('dof, l2, l2r, h1, h1r, energy, energyr, assemble_krylov, solve_krylov, total_krylov\n')
                matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(ns[0], l2_errors_krylov[0], 0, h1_errors_krylov[0], 0, energy_errors_krylov[0], 0, times_assemble_krylov[0], times_solve_krylov[0], times_total_krylov[0]))
                for ii in range(1, len(l2_errors_krylov)):
                    l2rr = ln(l2_errors_krylov[ii]/l2_errors_krylov[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    h1rr = ln(h1_errors_krylov[ii]/h1_errors_krylov[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    energyrr = ln(energy_errors_krylov[ii]/energy_errors_krylov[ii-1])/ln(1.*ns[ii]/ns[ii-1])
                    matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(ns[ii], l2_errors_krylov[ii], l2rr, h1_errors_krylov[ii], h1rr, energy_errors_krylov[ii], energyrr, times_assemble_krylov[ii], times_solve_krylov[ii], times_total_krylov[ii]))
                matfile.write("{:d}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3f}, {:.3e}, {:.3e}, {:.3e}\n".format(fine_dim, 0, global_l2rr, 0, global_h1rr, 0, global_energyrr, times_assemble_krylov[-1], times_solve_krylov[-1], times_total_krylov[-1]))
                matfile.close()

                minx = ns[0]; maxx = ns[-1]
                
                miny = np.min([np.min(energy_errors_krylov), np.min(l2_errors_krylov), np.min(h1_errors_krylov)]); maxy = np.max([np.max(energy_errors_krylov), np.max(l2_errors_krylov), np.max(h1_errors_krylov)])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(ns, l2_errors_krylov, 'go', mec = 'g', mfc = 'none', label = r'$e_0$')
                ax.loglog(ns, (l2_errors_krylov[0]/ns[0]**global_l2rr)*ns**global_l2rr, 'g-.', label = r'$c n^{'+str("%.2f"%global_l2rr)+r'}$')
                ax.loglog(ns, h1_errors_krylov, 'bx', mec = 'b',  mfc = 'none', label = r'$e_1$')
                ax.loglog(ns, (h1_errors_krylov[0]/ns[0]**global_h1rr)*ns**global_h1rr, 'b--', label = r'$c n^{'+str("%.2f"%global_h1rr)+r'}$')
                ax.loglog(ns, energy_errors_krylov, 'r+', mec = 'r', mfc = 'none', label = r'$e_E$')
                ax.loglog(ns, (energy_errors_krylov[0]/ns[0]**global_energyrr)*ns**global_energyrr, 'r:', label = r'$c n^{'+str("%.2f"%global_energyrr)+r'}$')
                ax.set_ylabel(r'$e_{\lbrace 0, 1, E\rbrace}$')
                ax.set_xlabel(r'$n = \#\operatorname{dof}$')
                sa_utils.set_log_ticks(ax, minx, maxx, True); sa_utils.set_log_ticks(ax, miny, maxy)
                fig.savefig(contrast_dir+'/'+basename+'_krylov_errors.pdf')
                    
                figlegend = plt.figure(figsize = (4*sa_utils.legendx, 2*sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_krylov_errors_legend.pdf', bbox_extra_artists = (lgd, ))
            
                mint = np.min([np.min(times_assemble_krylov), np.min(times_solve_krylov)]); maxt = np.max(times_total_krylov)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(all_ns, times_assemble_krylov, 'g+', mec = 'g', mfc = 'none', label = r'assembly, {:s}'.format(str(krylov_params)))
                ax.loglog(all_ns, (times_assemble_krylov[0]/all_ns[0]**global_assemblerr)*all_ns**global_assemblerr, 'g:', label = r'$c n^{'+str("%.2f"%global_assemblerr)+r'}$')
                ax.loglog(all_ns, times_solve_krylov, 'gx', mec = 'g', mfc = 'none', label = r'solve, {:s}'.format(str(krylov_params)))
                ax.loglog(all_ns, (times_solve_krylov[0]/all_ns[0]**global_solverr)*all_ns**global_solverr, 'g-.', label = r'$c n^{'+str("%.2f"%global_solverr)+r'}$')
                ax.loglog(all_ns, times_total_krylov, 'go', mec = 'g', mfc = 'none', label = r'total, {:s}'.format(str(krylov_params)))
                ax.loglog(all_ns, (times_total_krylov[0]/all_ns[0]**global_totalrr)*all_ns**global_totalrr, 'g-', label = r'$c n^{'+str("%.2f"%global_totalrr)+r'}$')
                ax.set_ylabel(r'$t$')
                ax.set_xlabel(r'$n = \#\operatorname{dof}$')
                sa_utils.set_log_ticks(ax, minx, maxx, True); sa_utils.set_log_ticks(ax, mint, maxt)
                fig.savefig(contrast_dir+'/'+basename+'_krylov_times.pdf')

                figlegend = plt.figure(figsize = (4.2*sa_utils.legendx, 2*sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_krylov_times_legend.pdf', bbox_extra_artists = (lgd, ))
             
                mint = np.min(times_total_krylov); maxt = np.max(times_total_krylov)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(times_total_krylov[:-1], l2_errors_krylov, 'go', mec = 'g', mfc = 'none', label = r'$e_0$')
                ax.loglog(times_total_krylov[:-1], h1_errors_krylov, 'bx', mec = 'b', mfc = 'none', label = r'$e_1$')
                ax.loglog(times_total_krylov[:-1], energy_errors_krylov, 'r+', mec = 'r', mfc = 'none', label = r'$e_E$')
                ax.set_ylabel(r'$e_{\lbrace 0, 1, E \rbrace}$')
                ax.set_xlabel(r'$t\;\operatorname{total}$')
                sa_utils.set_log_ticks(ax, mint, maxt, True); sa_utils.set_log_ticks(ax, miny, maxy)
                fig.savefig(contrast_dir+'/'+basename+'_krylov_errors_vs_total_krylov.pdf')

                figlegend = plt.figure(figsize = (3*sa_utils.legendx, sa_utils.legendy), frameon = False)
                handles, labels = ax.get_legend_handles_labels()
                ax = figlegend.add_subplot(111)
                ax.axis('off')
                lgd = ax.legend(handles, labels, loc = 10, ncol = 3)
                figlegend.savefig(contrast_dir+'/'+basename+'_krylov_errors_vs_total_legend.pdf', bbox_extra_artists = (lgd, ))

                min_it = np.min(iterations); max_it = np.max(iterations)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.loglog(all_ns, iterations, 'ro:', mec = 'r', mfc = 'none', label = r'{:s} iterations'.format(str(krylov_params)))
                ax.set_ylabel(r'{:s} iterations'.format(str(krylov_params)))
                ax.set_xlabel(r'$n = \#\operatorname{dof}$')
                sa_utils.set_log_ticks(ax, all_ns[0], all_ns[-1], True); sa_utils.set_log_ticks(ax, min_it, max_it)
                fig.savefig(contrast_dir+'/'+basename+'_iterations.pdf')
            
                plt.close('all')
                del fig, ax, figlegend, handles, labels, lgd, mint, maxt, minx, maxx, miny, maxy
                gc.collect()

            gc.collect()
            logger.info('Case [{:d}/{:d}] done'.format(num_bc+1, num_bcs))
            print('Case [{:d}/{:d}] done'.format(num_bc+1, num_bcs))

    logger.info('DONE')
    print('DONE')
    return 0
