from puma import *

import os, os.path, time, gc, getopt, fcntl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy.linalg as la

def myprint(str):
    paunt_print('LOG: {:s}'.format(str))

def exclusive_write(ff, cnt):
    fcntl.lockf(ff, fcntl.LOCK_EX)
    ff.write(cnt)
    ff.flush()
    fcntl.lockf(ff, fcntl.LOCK_UN)

class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def human_time(tt):
    seconds = tt % 60
    if seconds > 0.01:
        ret = '{:.2f}s'.format(seconds)
    else:
        ret = '{:.2e}s'.format(seconds)
    tt = int((tt-seconds))//60
    if tt == 0:
        return ret
    minutes = tt % 60
    ret = '{:d}m:{:s}'.format(minutes, ret)
    tt = (tt-minutes)//60
    if tt == 0:
        return ret
    hours = tt % 24
    ret = '{:d}h:{:s}'.format(hours, ret)
    tt = (tt-hours)//24
    if tt == 0:
        return ret
    return '{:d}d:{:s}'.format(tt, ret)

def pointInRectangle(xx, yy, minx, maxx, miny, maxy):
    return (minx < xx) & (xx < maxx) & (miny < yy) & (yy < maxy)

def lineIntersectCircle(x0, y0, x1, y1, cx, cy, rr2):
    d0 = (x0-cx)*(x0-cx)+(y0-cy)*(y0-cy)
    d1 = (x1-cx)*(x1-cx)+(y1-cy)*(y1-cy)

    dx = x1-x0; dy = y1-y0
    ll = dx*dx+dy*dy
    dcx0 = cx-x0; dcy0 = cy-y0

    t0 = (dcy0-dy/dx*dcx0)/(-dx-dy/dx*dy); s0 = (dcx0-dy*t0)/dx
    t1 = dcx0/dy; s1 = dcy0/dy

    return (d0 < rr2) | (d1 < rr2) | ((dx == 0) & (0 < s1) & (s1 < 1) & (t1*t1*ll < rr2)) | ((0 < s0) & (s0 < 1) & (t0*t0*ll < rr2))

def boxIntersectNotInCircle(cx, cy, rr):
    patch_min_x = PatchStretchedDomain().min_corner()[0]
    patch_min_y = PatchStretchedDomain().min_corner()[1]
    patch_max_x = PatchStretchedDomain().max_corner()[0]
    patch_max_y = PatchStretchedDomain().max_corner()[1]

    rr2 = rr*rr

    dists2 = [(x0-cx)*(x0-cx)+(y0-cy)*(y0-cy) for x0 in [patch_min_x, patch_max_x] for y0 in [patch_min_y, patch_max_y]]

    return (((dists2[0] > rr2) | (dists2[1] > rr2) | (dists2[2] > rr2) | (dists2[3] > rr2)) &
            (pointInRectangle(cx, cy, patch_min_x, patch_max_x, patch_min_y, patch_max_y) |
             lineIntersectCircle(patch_min_x, patch_min_y, patch_max_x, patch_min_y, cx, cy, rr2) |
             lineIntersectCircle(patch_max_x, patch_min_y, patch_max_x, patch_max_y, cx, cy, rr2) |
             lineIntersectCircle(patch_max_x, patch_max_y, patch_min_x, patch_max_y, cx, cy, rr2) |
             lineIntersectCircle(patch_min_x, patch_max_y, patch_min_x, patch_min_y, cx, cy, rr2)))

def calculate_stuff(elasticity=False, *,
                    compute_errors=True, compute_unenriched_solutions=False, compute_geometrically_enriched_solutions=False,
                    compute_reference_uu=False, compute_enriched_solutions=True,
                    write_vtk=True, write_enrichments=True,
                    global_levels=np.array([1, 2, 3]), global_degrees=np.array([2]), global_alpha=1.25,
                    patch_levels=np.array([4, 6, 8]), patch_degrees=np.array([2]), patch_alpha=1.25, oversampling_factors=np.array([1.5, 2.0, 2.5]),
                    reference_level=10, reference_degree=2, reference_alpha=1.25,
                    enrichment_type='lipton', enrichments_tols=np.array([1e-1, 1e-2, 1e-3, 1e-4]),
                    contrast=1e4, nu=0.3, EE=1,
                    solver_tolerance=1e-10, solver_max_iterations=10000, orthogonalization_tolerance=1e-7,
                    with_inclusions=True, ldomain=True,
                    basedim=2):
    Logger.instance().enable_to_file()
    Logger.instance().enable_to_stdout(True)

    def solver_parameters(finest_level):
        return {
            'linear_solver' : {
                'type' : 'cg',
                'package' : 'petsc',
                'relative_tolerance': solver_tolerance,
                'absolute_tolerance': 0,
                'max_iterations': solver_max_iterations,
                'quiet': False,
                'preconditioner' : {
                    'type' : 'hypre.boomeramg'

                }
            }
        }

    myprint('Starting multiscale puma computations with the following options')
    myprint('    elasticity:                                {:s}'.format(str(elasticity)))
    myprint('    contrast:                                  {:.2e}'.format(contrast))
    if elasticity: 
        myprint('    nu:                                        {:.2e}'.format(nu))
        myprint('    E:                                         {:.2e}'.format(EE))
    myprint('    compute errors:                            {:s}'.format(str(compute_errors)))
    myprint('    compute enriched solutions:                {:s}'.format(str(compute_enriched_solutions)))
    myprint('    compute reference solution:                {:s}'.format(str(compute_reference_uu)))
    myprint('    compute unenriched solutions:              {:s}'.format(str(compute_unenriched_solutions)))
    myprint('    compute geometrically enriched solutions:  {:s}'.format(str(compute_geometrically_enriched_solutions)))
    myprint('    write vtk:                                 {:s}'.format(str(write_vtk)))
    myprint('    reference level:                           {:d}'.format(reference_level))
    myprint('    reference degree:                          {:d}'.format(reference_degree))
    myprint('    reference alpha:                           {:.2e}'.format(reference_alpha))
    myprint('    global levels:                             {:s}'.format(str(global_levels)))
    myprint('    global degrees:                            {:s}'.format(str(global_degrees)))
    myprint('    global alpha:                              {:.2e}'.format(global_alpha))
    myprint('    solver tolerance:                          {:.2e}'.format(solver_tolerance))
    myprint('    orthogonalization tolerance:               {:.2e}'.format(orthogonalization_tolerance))
    if compute_enriched_solutions:
        myprint('    write enrichments:                         {:s}'.format(str(write_enrichments)))
        myprint('    oversampling factors:                      {:s}'.format(str(oversampling_factors)))
        myprint('    patch levels:                              {:s}'.format(str(patch_levels)))
        myprint('    patch degrees:                             {:s}'.format(str(patch_degrees)))
        myprint('    patch alpha:                               {:.2e}'.format(patch_alpha))
        myprint('    enrichment type:                           {:s}'.format(enrichment_type))
        myprint('    enrichment accuracies:                     {:s}'.format(str(enrichments_tols)))

    use_local_solutions = True
    if enrichment_type == 'efendiev':
        use_local_solutions = False

    corner_dist = 0.2

    if with_inclusions:
        inclusion_centers = [Point(-0.375, -0.375), Point(0.5, -0.5)]
        inclusion_radii = [0.2, 0.1]
        assert(len(inclusion_centers) == len(inclusion_radii))
        num_inclusions = len(inclusion_centers)

        inclusions = ConstructiveSolidGeometry(Ball(inclusion_centers[0], inclusion_radii[0]), 64)
        for ii in range(num_inclusions):
            inclusions += ConstructiveSolidGeometry(Ball(inclusion_centers[ii], inclusion_radii[ii]), 64)
    else:
        num_inclusions = 0
        inclusions = None

    myprint('    num inclusions:                            {:d}'.format(num_inclusions))
    myprint('    ldomain                                    {:s}'.format(str(ldomain)))

    problem_type = ''
    xx = Position()

    if ldomain:
        problem_type += '_ldomain'
        def at_corner(area):
            return within_point_box_touches(Point(0,0), area)

        def near_corner(area):
            return distance_point_box(Point(0,0), area) < np.min([corner_dist, 0.5**(global_level-1)])

        def get_uniform_refinements(omega_i):
            return None

        def should_refine_adaptive(area, iteration):
            return (near_corner(area) & (iteration < 2)) | (at_corner(area) & (iteration < 4))

        pp = PolarTransformation()(xx)
        rr = pp[0]
        theta = pp[1]
        theta = conditional(theta <= 0, theta + 2*math.pi, theta)

        eta = rr**(2/3) * sin((2*theta - math.pi) / 3)
        
        detadr = 2/3 * rr**(-1./3) * sin((2*theta - math.pi) / 3)
        detadtheta = 2/3 * rr**(2/3) * cos((2*theta - math.pi) / 3)
        eta.set_gradient(grad(pp).T * (detadr, detadtheta))
        
        eta.set_polynomial_degree(conditional(at_corner(PatchStretchedDomain()), 4, 2))
        grad(eta).set_polynomial_degree((1,1))

    if elasticity:
        mul = basedim
        null_dim = 3
        null = (0,0)
        problem_type = 'elasticity'+problem_type
        TrialFunction = VectorTrialFunction
        TestFunction = VectorTestFunction
        Function = VectorFunction

        if num_inclusions:
            def get_kk(domain, csg):
                csg_matrix = ConstructiveSolidGeometrySubDomain2D(domain, csg-inclusions)
                csg_inclusions = ConstructiveSolidGeometrySubDomain2D(domain, csg*inclusions)
                return lambda uu, vv: LinearElasticMaterialModel(csg_matrix, E=EE, nu=nu).second_variation_of_strain_energy(uu, vv)+LinearElasticMaterialModel(csg_inclusions, E=EE*contrast, nu=nu).second_variation_of_strain_energy(uu, vv)
        else:
            def get_kk(domain, csg):
                csg_subdomain = ConstructiveSolidGeometrySubDomain2D(domain, csg)
                return lambda uu, vv: LinearElasticMaterialModel(csg_subdomain, E=EE, nu=nu).second_variation_of_strain_energy(uu, vv)
                
        global_dirichlet = [('Left',(-0.5,0)), ('Right',(0.5,0))]
        global_ff = (0,0)
    else:
        mul = 1
        null_dim = 1
        null = 0
        problem_type = 'heat'+problem_type
        TrialFunction = ScalarTrialFunction
        TestFunction = ScalarTestFunction
        Function = ScalarFunction

        if num_inclusions:
            def get_kk(domain, csg):
                csg_matrix = ConstructiveSolidGeometrySubDomain2D(domain, csg-inclusions)
                csg_inclusions = ConstructiveSolidGeometrySubDomain2D(domain, csg*inclusions)
                return lambda uu, vv: inner(grad(uu), grad(vv))*dx(csg_matrix)+contrast*inner(grad(uu), grad(vv))*dx(csg_inclusions)
        else:
            def get_kk(domain, csg):
                csg_subdomain = ConstructiveSolidGeometrySubDomain2D(domain, csg)
                return lambda uu, vv: inner(grad(uu), grad(vv))*dx(csg_subdomain)
        if not with_inclusions:
            if ldomain:
                polym = as_geco_expression((xx[0]**2+xx[0]*xx[1]-xx[1]**2 + 1))
                polym.set_gradient((2*xx[0]+xx[1], xx[0]-2*xx[1]))

                reference_uu = eta * polym
                reference_uu.set_gradient(eta*(grad(polym))+polym*(grad(eta)))

                global_ff = -2*dot(grad(eta), grad(polym))
            else:
                myprint('HERE')
                reference_uu = as_geco_expression((xx[0]*(xx[0]**4-6*xx[0]**2*xx[1]**2+xx[1]**4)))
                reference_uu.set_gradient((5*xx[0]**4 - 18*xx[0]**2*xx[1]**2 + xx[1]**4, 4*xx[0]*xx[1]*(-3*xx[0]**2+xx[1]**2))) 
                global_ff = -8*xx[0]*(xx[0]**2 - 3*xx[1]**2)
        else:
            reference_uu = 0
            global_ff = -1

        global_dirichlet = [('Left', reference_uu), ('Right', reference_uu),
                            ('Bottom', reference_uu), ('Top', reference_uu)]
        if ldomain:
            global_dirichlet += [('Corner_Right', reference_uu), ('Corner_Upper', reference_uu)]
    
    def dim_from_coeff(space):
        uu = Function(space)
        return uu.coefficient().local_scalar_size()

    ortho_eps = orthogonalization_tolerance*orthogonalization_tolerance

    number = 5
    number_fill = int(np.log(number)/np.log(10.))+1

    enrichments_tols_original = np.sort(enrichments_tols)[::-1]

    global_degrees = np.sort(global_degrees)
    global_degrees_len = len(global_degrees)

    def get_random_inclusions(number=10, radius=0.2, random_radius=0.1):
        ret = []
        radii = []
        percentages = np.arange(.1, 1.05, 0.1)*num_inclusions
        idx = 1
        while(len(ret) < num_inclusions):
            notok = True
            while(notok):
                new = rnd.rand(2)*2-1
                new_radius = (random_radius+rnd.rand()*(1.-random_radius))*shape_radius
                notok = False
                for oldx, oldr in zip(ret, radii):
                    diff = new-oldx
                    if sqrt(diff @ diff) < 1.1*(oldr+new_radius):
                        notok = True
                        break
            ret.append(new)
            radii.append(new_radius)
            if len(ret) > percentages[idx]:
                logger.info('    [{:d}/{:d} = {:.2f}%] done'.format(len(ret), num_inclusions, 100.*len(ret)/num_inclusions))
                idx += 1
        return [Point(*pt) for pt in ret], np.array(radii)

    global_tic = time.process_time()
    myprint('Start')

    box = ConstructiveSolidGeometry(Box(Point(-1, -1), Point(1, 1)))
    box.setCoDim1PhysicalGroupName(0, "Left");
    box.setCoDim1PhysicalGroupName(1, "Right");
    box.setCoDim1PhysicalGroupName(2, "Bottom");
    box.setCoDim1PhysicalGroupName(3, "Top");
    if ldomain:
        cut = ConstructiveSolidGeometry(Box(Point(0, 0), Point(1, 1)))
        cut.setCoDim1PhysicalGroupName(0, "Corner_Right");
        cut.setCoDim1PhysicalGroupName(2, "Corner_Upper");
        whole = box - cut
    else:
        whole = box
    global_omega = ConstructiveSolidGeometryDomain(whole)
    myprint('CSG domain set up')

    get_global_l2 = lambda uu, vv: inner(uu, vv)*dx
    get_global_h1 = lambda uu, vv: inner(grad(uu), grad(vv))*dx
    get_global_kk = get_kk(global_omega, whole)
    myprint('CSG subdomains and bilinear forms set up')

    problem_path = '{:s}_inclusions{:d}'.format(problem_type, num_inclusions)
    os.makedirs(problem_path, exist_ok=True)
    with cd(problem_path):
        if compute_errors:
            myprint('Preparing error computations')
            if not with_inclusions and not elasticity:
                reference_name = 'ref_analytical'
                def get_diff_handler(VV, *, local_spaces=None, uu_handler=None):
                    return uu_handler

                def reference_errors(uu, handler):
                    reference_VV = uu.V()
                    reference_l2 = Scalar()
                    reference_h1 = Scalar()
                    reference_energy = Scalar()
                    l2_error = Scalar()
                    h1_error = Scalar()
                    energy_error = Scalar()
                    diff = reference_uu-uu
                    assemble((get_global_l2(reference_uu, reference_uu), reference_l2, reference_VV),
                             (get_global_h1(reference_uu, reference_uu), reference_h1, reference_VV),
                             (get_global_kk(reference_uu, reference_uu), reference_energy, reference_VV),
                             (get_global_l2(diff, diff), l2_error, reference_VV),
                             (get_global_h1(diff, diff), h1_error, reference_VV),
                             (get_global_kk(diff, diff), energy_error, reference_VV),
                             integration_cell_handler=handler)
                    l2_rel = np.sqrt(l2_error.value()/reference_l2.value())
                    h1_rel = np.sqrt(h1_error.value()/reference_h1.value())
                    energy_rel = np.sqrt(energy_error.value()/reference_energy.value())
                    return l2_rel, h1_rel, energy_rel
                myprint('Error computation against analysical solution set up')
            else:
                load_reference_uu = False
                if not compute_reference_uu:
                    load_reference_uu = True
                
                reference_l2 = None
                reference_h1 = None
                reference_energy = None

                if load_reference_uu:
                    myprint('Trying to load reference solution')
                    name = 'geometrically_enriched_l{:d}_d{:d}'.format(reference_level, reference_degree)
                    path = 'geometrically_enriched_solutions/globallevel{:d}/globaldegree{:d}/{:s}.pnt'.format(reference_level, reference_degree, name)
                    try:
                        data = SimulationStateData()
                        data.read(path)
                        reference_uu = data.read_function_2d(name)
                    except Exception as ee:
                        myprint(str(ee))
                        compute_reference_uu = True
                        myprint('Could not load reference solution')
                    else:
                        myprint('    Computing norms')
                        reference_VV = reference_uu.V()
                        tic = time.process_time()
                        reference_integration_cell_handler = IntegrationCellHandlerCover(reference_VV.cover(), reference_VV.level())
                        reference_integration_cell_handler.generate_cached_cells([reference_VV])
                        toc = time.process_time()
                        myprint('    Constructed reference solution space integration cell handler in [{:s}]'.format(human_time(toc-tic)))

                        reference_l2 = Scalar()
                        reference_h1 = Scalar()
                        reference_energy = Scalar()
                        assemble((get_global_l2(reference_uu, reference_uu), reference_l2, reference_VV),
                                 (get_global_h1(reference_uu, reference_uu), reference_h1, reference_VV),
                                 (get_global_kk(reference_uu, reference_uu), reference_energy, reference_VV),
                                 integration_cell_handler=reference_integration_cell_handler)
                        reference_l2 = np.sqrt(reference_l2.value())
                        reference_h1 = np.sqrt(reference_h1.value())
                        reference_energy = np.sqrt(reference_energy.value())
                        myprint('    Computed norms')

                        del reference_integration_cell_handler
                        myprint('Loaded reference solution')
                    gc.collect()

                if compute_reference_uu:
                    myprint('Constructing reference solution space')
                    reference_VV = PumSpace(global_omega, level=reference_level, polynomial_degree=reference_degree, stretch_factor=reference_alpha)
                    reference_unenriched_dof = dim_from_coeff(reference_VV)
                    if ldomain:
                        global_level = reference_level
                        reference_VV.enrich(eta, EnrichmentApplicationType.additive, near_corner(PatchStretchedDomain()), get_uniform_refinements(PatchLocalDomain()), should_refine_adaptive(IntegrationDomainArea(), IterationCount()))

                    for ii in range(num_inclusions):
                        xx = Position()
                        TT = Translation(-inclusion_centers[ii])
                        pp = PolarTransformation()(TT(xx))
                        rr = pp[0]
                        theta = pp[1]

                        uu = rr-inclusion_radii[ii]
                        uu.set_gradient(grad(pp).T*(1, 0))
                        uu.set_polynomial_degree(3)
                        grad(uu).set_polynomial_degree((2,2))

                        reference_VV.enrich(conditional(rr-inclusion_radii[ii] > 0, uu, Constant(0)), EnrichmentApplicationType.multiplicative, boxIntersectNotInCircle(inclusion_centers[ii][0], inclusion_centers[ii][1], inclusion_radii[ii]))
                        del xx, TT, pp, rr, theta, uu
                    
                    reference_dof = dim_from_coeff(reference_VV)
                    myprint('Reference solution space of [{:d}] dof enriched to [{:d}] dof'.format(reference_unenriched_dof, reference_dof))
                    reference_bcs = []
                    for (part, bc) in global_dirichlet:
                        try:
                            tmp = PhysicalGroupSubDomain(global_omega, part)
                        except:
                            myprint('    Boundary part [{:s}] not found, skipping'.format(part))
                        else:
                            reference_bcs.append(DirichletBoundaryCondition(reference_VV, tmp, bc))

                    tic = time.process_time()
                    reference_integration_cell_handler = IntegrationCellHandlerCover(reference_VV.cover(), reference_VV.level())
                    reference_integration_cell_handler.generate_cached_cells([reference_VV])
                    toc = time.process_time()
                    myprint('Constructed reference solution space integration cell handler in [{:s}]'.format(human_time(toc-tic)))

                    myprint('Computing reference solution')
                    path = 'geometrically_enriched_solutions'
                    os.makedirs(path, exist_ok=True)
                    with cd(path):
                        myprint('    Stable transformation construction')
                        tic = time.process_time()
                        reference_test = TestFunction(reference_VV)
                        reference_trial = TrialFunction(reference_VV)

                        reference_kk = get_global_kk(reference_trial, reference_test)
                        reference_stable_kk = get_global_l2(reference_trial, reference_test) + get_global_h1(reference_trial, reference_test)
                        reference_stable_KK = GlobalDiagonalMatrix(reference_VV)
                        assemble((reference_stable_kk, reference_stable_KK), integration_cell_handler=reference_integration_cell_handler)
                        myprint('        matrix assembled')
                        reference_st = create_stable_transformation(reference_VV, reference_stable_KK, epsilon=orthogonalization_tolerance)
                        toc = time.process_time()
                        myprint('    Stable transformation constructed in [{:s}]'.format(human_time(toc-tic)))

                        reference_LL = inner(reference_test, global_ff)*dx

                        myprint('    Computing reference solution')
                        reference_uu = Function(reference_VV)
                        tic = time.process_time()
                        reference_solver = solve(reference_kk == reference_LL, reference_uu, reference_bcs, reference_st, integration_cell_handler=reference_integration_cell_handler, solver_parameters=solver_parameters(reference_level))
                        toc = time.process_time()
                        reference_iterations = reference_solver.linear_solver().last_iterations()
                        myprint('    Reference solution computed in [{:s}] with [{:d}] iterations'.format(human_time(toc-tic), reference_iterations))

                        path = 'globallevel{:d}/globaldegree{:d}'.format(reference_level, reference_degree)
                        os.makedirs(path, exist_ok=True)
                        with cd(path):
                            name = 'geometrically_enriched_l{:d}_d{:d}'.format(reference_level, reference_degree)
                            write_to_pnt('{:s}.pnt'.format(name), (reference_uu, name))
                            if write_vtk:
                                write_continuous_vtk(reference_VV, r'{:s}.vtk'.format(name), (reference_uu, name), integration_cell_handler=reference_integration_cell_handler)
                   
                        myprint('    Computing norms')
                        reference_l2 = Scalar()
                        reference_h1 = Scalar()
                        reference_energy = Scalar()
                        assemble((get_global_l2(reference_uu, reference_uu), reference_l2, reference_VV),
                                 (get_global_h1(reference_uu, reference_uu), reference_h1, reference_VV),
                                 (get_global_kk(reference_uu, reference_uu), reference_energy, reference_VV),
                                 integration_cell_handler=reference_integration_cell_handler)
                        reference_l2 = np.sqrt(reference_l2.value())
                        reference_h1 = np.sqrt(reference_h1.value())
                        reference_energy = np.sqrt(reference_energy.value())
                        myprint('    Computed norms')
                        with cd(path):
                            ff = open('{:s}.csv', 'w')
                            ff.write('level, degree, iterations, l2, h1, energy\n')
                            ff.write('{:d}, {:d}, {:d}, {:.3e}, {:.3e}, {:.3e}'.format(reference_level, reference_degree, reference_iterations, reference_l2, reference_h1, reference_energy))
                            ff.close()

                    del reference_bcs, reference_trial, reference_test, reference_integration_cell_handler, reference_st, reference_kk, reference_stable_kk, reference_stable_KK, reference_LL, reference_solver
                    gc.collect()
                    myprint('Computed reference solution with [{:d}] dof in {:s}'.format(reference_dof, human_time(toc-tic)))

                reference_name = 'ref_l{:d}_d{:d}'.format(reference_level, reference_degree)
                def get_diff_handler(VV, *, local_spaces=[], uu_handler=None):
                    handler = IntegrationCellHandlerCover(reference_VV.cover(), reference_VV.level())
                    handler.generate_cached_cells([reference_VV, VV]+local_spaces)
                    return handler

                def reference_errors(uu, handler):
                    l2_error = Scalar()
                    h1_error = Scalar()
                    energy_error = Scalar()
                    diff = reference_uu-uu
                    assemble((get_global_l2(diff, diff), l2_error, reference_VV),
                             (get_global_h1(diff, diff), h1_error, reference_VV),
                             (get_global_kk(diff, diff), energy_error, reference_VV),
                             integration_cell_handler=handler)
                    l2_rel = np.sqrt(l2_error.value()) / reference_l2
                    h1_rel = np.sqrt(h1_error.value()) / reference_h1
                    energy_rel = np.sqrt(energy_error.value()) / reference_energy
                    return l2_rel, h1_rel, energy_rel
                myprint('Error computation against analytical solution set up')

        if compute_unenriched_solutions:
            myprint('Computing unenriched solutions')
            
            path = 'unenriched_solutions'
            os.makedirs(path, exist_ok=True)
            with cd(path):
                if compute_errors:
                    unenriched_error_filename = '{:s}_unenriched_errors.csv'.format(reference_name)
                    if os.path.isfile(unenriched_error_filename):
                        unenriched_error_file = open(unenriched_error_filename, 'a')
                    else:
                        unenriched_error_file = open(unenriched_error_filename, 'w')
                        exclusive_write(unenriched_error_file, 'global_level, global_degree, dof, iterations, l2_rel, h1_rel, energy_rel\n')

                for global_level_ii, global_level in enumerate(global_levels):
                    myprint('[{:d}/{:d}], level [{:d}] start'.format(global_level_ii+1, len(global_levels), global_level))

                    for global_degree_ii, global_degree in enumerate(global_degrees):
                        myprint('    [{:d}/{:d}], level [{:d}], degree [{:d}] start'.format(global_degree_ii+1, len(global_degrees), global_level, global_degree))

                        global_VV = PumSpace(global_omega, level=global_level, polynomial_degree=global_degree, stretch_factor=global_alpha)
                        global_dof = dim_from_coeff(global_VV)

                        global_bcs = []
                        for (part, bc) in global_dirichlet:
                            try:
                                tmp = PhysicalGroupSubDomain(global_omega, part)
                            except:
                                myprint('        Boundary part [{:s}] not found, skipping'.format(part))
                            else:
                                global_bcs.append(DirichletBoundaryCondition(global_VV, tmp, bc))

                        myprint('        Global integration cell handler construction')
                        tic = time.process_time()
                        global_integration_cell_handler = IntegrationCellHandlerCover(global_VV.cover(), global_VV.level())
                        global_integration_cell_handler.generate_cached_cells([global_VV])
                        toc = time.process_time()
                        myprint('        Global integration cell handler constructed in [{:s}]'.format(human_time(toc-tic)))

                        myprint('        Stable transformation construction')
                        tic = time.process_time()
                        global_test = TestFunction(global_VV)
                        global_trial = TrialFunction(global_VV)

                        global_kk = get_global_kk(global_trial, global_test)
                        global_stable_kk = get_global_l2(global_trial, global_test) + get_global_h1(global_trial, global_test)
                        global_stable_KK = GlobalDiagonalMatrix(global_VV)
                        assemble((global_stable_kk, global_stable_KK), integration_cell_handler=global_integration_cell_handler)
                        global_st = create_stable_transformation(global_VV, global_stable_KK, epsilon=orthogonalization_tolerance)
                        toc = time.process_time()
                        myprint('        Stable transformation constructed in [{:s}]'.format(human_time(toc-tic)))

                        global_LL = inner(global_test, global_ff)*dx

                        myprint('        Computing unenriched solution')
                        global_uu = Function(global_VV)
                        tic = time.process_time()
                        global_solver = solve(global_kk == global_LL, global_uu, global_bcs, global_st, integration_cell_handler=global_integration_cell_handler, solver_parameters=solver_parameters(global_level))
                        toc = time.process_time()
                        global_iterations = global_solver.linear_solver().last_iterations()
                        myprint('        Unenriched solution computed in [{:s}] with [{:d}] iterations'.format(human_time(toc-tic), global_iterations))

                        path = 'globallevel{:d}/globaldegree{:d}'.format(global_level, global_degree)
                        os.makedirs(path, exist_ok=True)
                        with cd(path):
                            name = 'unenriched_l{:d}_d{:d}'.format(global_level, global_degree)
                            write_to_pnt('{:s}.pnt'.format(name), (global_uu, name))
                            if write_vtk:
                                write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (global_uu, name), integration_cell_handler=global_integration_cell_handler)

                        if compute_errors:
                            myprint('        Computing errors')
                            l2_rel, h1_rel, energy_rel = reference_errors(global_uu, get_diff_handler(global_VV, uu_handler=global_integration_cell_handler))
                            myprint('            dof:    {:d}'.format(global_dof))
                            myprint('            L2:     {:.2e}'.format(l2_rel))
                            myprint('            H1:     {:.2e}'.format(h1_rel))
                            myprint('            energy: {:.2e}'.format(energy_rel))
                            exclusive_write(unenriched_error_file, '{:d}, {:d}, {:d}, {:d}, {:.3e}, {:.3e}, {:.3e}\n'.format(global_level, global_degree, global_dof, global_iterations, l2_rel, h1_rel, energy_rel))
                        
                        del global_VV, global_bcs, global_trial, global_test, global_integration_cell_handler, global_st, global_kk, global_stable_kk, global_stable_KK, global_LL, global_uu, global_solver
                        gc.collect()
                        myprint('    [{:d}/{:d}], level [{:d}], degree [{:d}] done in [{:s}]'.format(global_degree_ii+1, len(global_degrees), global_level, global_degree, human_time(toc-global_tic)))
                    myprint('[{:d}/{:d}], level [{:d}] end'.format(global_level_ii+1, len(global_levels), global_level))

                if compute_errors:
                    unenriched_error_file.close()
            myprint('Computed unenriched solutions')

        if compute_geometrically_enriched_solutions:
            myprint('Computing geometrically enriched solutions')
            path = 'geometrically_enriched_solutions'
            os.makedirs(path, exist_ok=True)
            with cd(path):
                if compute_errors:
                    geometrically_enriched_error_filename = '{:s}_geometrically_enriched_errors.csv'.format(reference_name)
                    if os.path.isfile(geometrically_enriched_error_filename):
                        geometrically_enriched_error_file = open(geometrically_enriched_error_filename, 'a')
                    else:
                        geometrically_enriched_error_file = open(geometrically_enriched_error_filename, 'w')
                        exclusive_write(geometrically_enriched_error_file, 'global_level, global_degree, dof, iterations, l2_rel, h1_rel, energy_rel\n')

                for global_level_ii, global_level in enumerate(global_levels):
                    myprint('[{:d}/{:d}], level [{:d}] start'.format(global_level_ii+1, len(global_levels), global_level))

                    for global_degree_ii, global_degree in enumerate(global_degrees):
                        myprint('    [{:d}/{:d}], level [{:d}], degree [{:d}] start'.format(global_degree_ii+1, len(global_degrees), global_level, global_degree))

                        global_VV = PumSpace(global_omega, level=global_level, polynomial_degree=global_degree, stretch_factor=global_alpha)
                        if ldomain:
                            global_VV.enrich(eta, EnrichmentApplicationType.additive, near_corner(PatchStretchedDomain()), get_uniform_refinements(PatchLocalDomain()), should_refine_adaptive(IntegrationDomainArea(), IterationCount()))

                        for ii in range(num_inclusions):
                            xx = Position()
                            TT = Translation(-inclusion_centers[ii])
                            pp = PolarTransformation()(TT(xx))
                            rr = pp[0]
                            theta = pp[1]

                            uu = rr-inclusion_radii[ii]
                            uu.set_gradient(grad(pp).T*(1, 0))
                            uu.set_polynomial_degree(3)
                            grad(uu).set_polynomial_degree((2,2))

                            global_VV.enrich(conditional(rr-inclusion_radii[ii] > 0, uu, Constant(0)), EnrichmentApplicationType.multiplicative, boxIntersectNotInCircle(inclusion_centers[ii][0], inclusion_centers[ii][1], inclusion_radii[ii]))
                            del xx, TT, pp, rr, theta, uu

                        global_dof = dim_from_coeff(global_VV)
             
                        global_bcs = []
                        for (part, bc) in global_dirichlet:
                            try:
                                tmp = PhysicalGroupSubDomain(global_omega, part)
                            except:
                                myprint('        Boundary part [{:s}] not found, skipping')
                            else:
                                global_bcs.append(DirichletBoundaryCondition(global_VV, tmp, bc))

                        myprint('        Global integration cell handler construction')
                        tic = time.process_time()
                        global_integration_cell_handler = IntegrationCellHandlerCover(global_VV.cover(), global_VV.level())
                        global_integration_cell_handler.generate_cached_cells([global_VV])
                        toc = time.process_time()
                        myprint('        Global integration cell handler constructed in [{:s}]'.format(human_time(toc-tic)))

                        myprint('        Stable transformation construction')
                        tic = time.process_time()
                        global_test = TestFunction(global_VV)
                        global_trial = TrialFunction(global_VV)

                        global_kk = get_global_kk(global_trial, global_test)
                        global_stable_kk = get_global_l2(global_trial, global_test) + get_global_h1(global_trial, global_test)
                        global_stable_KK = GlobalDiagonalMatrix(global_VV)
                        assemble((global_stable_kk, global_stable_KK), integration_cell_handler=global_integration_cell_handler)
                        global_st = create_stable_transformation(global_VV, global_stable_KK, epsilon=orthogonalization_tolerance)
                        toc = time.process_time()
                        myprint('        Stable transformation constructed in [{:s}]'.format(human_time(toc-tic)))

                        global_LL = inner(global_test, global_ff)*dx

                        myprint('        Computing geometrically enriched solution')
                        global_uu = Function(global_VV)
                        tic = time.process_time()
                        global_solver = solve(global_kk == global_LL, global_uu, global_bcs, global_st, integration_cell_handler=global_integration_cell_handler, solver_parameters=solver_parameters(global_level))
                        toc = time.process_time()
                        global_iterations = global_solver.linear_solver().last_iterations()
                        myprint('        Geometrically enriched solution computed in [{:s}] with [{:d}] iterations'.format(human_time(toc-tic), global_iterations))

                        path = 'globallevel{:d}/globaldegree{:d}'.format(global_level, global_degree)
                        os.makedirs(path, exist_ok=True)
                        with cd(path):
                            name = 'geometrically_enriched_l{:d}_d{:d}'.format(global_level, global_degree)
                            write_to_pnt('{:s}.pnt'.format(name), (global_uu, name))
                            if write_vtk:
                                write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (global_uu, name), integration_cell_handler=global_integration_cell_handler)
                        
                        if compute_errors:
                            myprint('        Computing errors')
                            l2_rel, h1_rel, energy_rel = reference_errors(global_uu, get_diff_handler(global_VV, uu_handler=global_integration_cell_handler))
                            myprint('            dof:    {:d}'.format(global_dof))
                            myprint('            L2:     {:.2e}'.format(l2_rel))
                            myprint('            H1:     {:.2e}'.format(h1_rel))
                            myprint('            energy: {:.2e}'.format(energy_rel))
                            exclusive_write(geometrically_enriched_error_file, '{:d}, {:d}, {:d}, {:d}, {:.3e}, {:.3e}, {:.3e}\n'.format(global_level, global_degree, global_dof, global_iterations, l2_rel, h1_rel, energy_rel))
 
                        del global_VV, global_bcs, global_trial, global_test, global_integration_cell_handler, global_st, global_kk, global_stable_kk, global_stable_KK, global_LL, global_uu, global_solver
                        gc.collect()
                        myprint('    [{:d}/{:d}], level [{:d}], degree [{:d}] done in [{:s}]'.format(global_degree_ii+1, len(global_degrees), global_level, global_degree, human_time(toc-global_tic)))
                    myprint('[{:d}/{:d}], level [{:d}] start'.format(global_level_ii+1, len(global_levels), global_level))

            myprint('Computed geometrically enriched solutions')

        gc.collect()
        if compute_enriched_solutions:
            myprint('Compute optimally enriched solutions')
            if compute_errors:
                os.makedirs('enriched_solutions', exist_ok = True)
                enriched_error_filename = 'enriched_solutions/{:s}_enriched_errors.csv'.format(reference_name)
                if os.path.isfile(enriched_error_filename):
                    enriched_error_file = open(enriched_error_filename, 'a')
                else:
                    enriched_error_file = open(enriched_error_filename, 'w')
                    exclusive_write(enriched_error_file, 'enrichment_type, oversampling_factor, global_level, global_degree, patch_level, patch_degree, dof, iterations, l2_rel, h1_rel, energy_rel\n')
            
            for global_level_ii, global_level in enumerate(global_levels):
                myprint('[{:d}/{:d}], level [{:d}] start'.format(global_level_ii+1, len(global_levels), global_level))

                global_VV = PumSpace(global_omega, level=global_level, polynomial_degree=global_degrees[0], stretch_factor=global_alpha)
                global_dof = dim_from_coeff(global_VV)
                global_cover = global_VV.cover()
                global_num_patches = global_VV.local_count()
                global_num_fill = int(np.log(global_num_patches)/np.log(10.))+1
                
                for oversampling_factor_ii, oversampling_factor in enumerate(oversampling_factors):
                    if oversampling_factor > 2**global_level:
                        myprint('    [{:d}/{:d}], oversampling factor [{:.2e}] too large, skipping'.format(oversampling_factor_ii+1, len(oversampling_factors), oversampling_factor))
                        continue
                    else:
                        myprint('    [{:d}/{:d}], oversampling factor [{:.2e}] start'.format(oversampling_factor_ii+1, len(oversampling_factors), oversampling_factor))
                    for patch_level_ii, patch_level in enumerate(patch_levels):
                        myprint('        [{:d}/{:d}], patch level [{:d}] start'.format(patch_level_ii+1, len(patch_levels), patch_level))
                        for patch_degree_ii, patch_degree in enumerate(patch_degrees):
                            myprint('            [{:d}/{:d}], patch degree [{:d}] start'.format(patch_degree_ii+1, len(patch_degrees), patch_degree))

                            enrichments_tols = enrichments_tols_original.copy()
                            enrichments_tols_len = len(enrichments_tols)

                            myprint('            Computing enrichments')
                            local_spaces = []
                            local_enrichments = []
                            local_solutions = []
                            local_dof = 0
                            local_enriched_dof = 0

                            enrichment_numbers = []
                            patch_numbers = []
                            integration_cell_handlers = []
                            center_points = []
                            patches_ok = True
                            for global_ii in range(global_num_patches):
                                myprint('                patch [{:d}/{:d}] start'.format(global_ii+1, global_num_patches))
                                patch_name = 'patch{:s}'.format(str(global_ii).zfill(global_num_fill))
                                patch_path = 'oversamplingfactor{:.2e}/globallevel{:d}/patchlevel{:d}/patchdegree{:d}/{:s}'.format(oversampling_factor, global_level, patch_level, patch_degree, patch_name)
                                patch_tic = time.process_time()
                                patch = global_cover.patch_from_local_index(global_ii, global_level)
                                patch_stretched = patch.stretched_domain()
                                patch_min_x = patch_stretched.min_corner()[0]
                                patch_max_x = patch_stretched.max_corner()[0]
                                patch_min_y = patch_stretched.min_corner()[1]
                                patch_max_y = patch_stretched.max_corner()[1]

                                patch_center_x = (patch_max_x + patch_min_x)/2
                                patch_radius_x = (patch_max_x - patch_min_x)/2
                                patch_radius_x = patch_radius_x / global_alpha
                                patch_center_y = (patch_max_y + patch_min_y)/2
                                patch_radius_y = (patch_max_y - patch_min_y)/2
                                patch_radius_y = patch_radius_y / global_alpha

                                center_points.append(Point(patch_center_x, patch_center_y))

                                patch_min_x = patch_center_x - patch_radius_x
                                patch_max_x = patch_center_x + patch_radius_x
                                patch_min_y = patch_center_y - patch_radius_y
                                patch_max_y = patch_center_y + patch_radius_y
                                patch_box = Box(Point(patch_min_x, patch_min_y), Point(patch_max_x, patch_max_y))

                                oversampled_min_x = patch_center_x - patch_radius_x*oversampling_factor
                                oversampled_max_x = patch_center_x + patch_radius_x*oversampling_factor
                                oversampled_min_y = patch_center_y - patch_radius_y*oversampling_factor
                                oversampled_max_y = patch_center_y + patch_radius_y*oversampling_factor
                              
                                oversampled_patch_box = ConstructiveSolidGeometry(Box(Point(oversampled_min_x, oversampled_min_y), Point(oversampled_max_x, oversampled_max_y)))
                                oversampled_patch_box.setCoDim1PhysicalGroupName(0, "patch_left");
                                oversampled_patch_box.setCoDim1PhysicalGroupName(1, "patch_right");
                                oversampled_patch_box.setCoDim1PhysicalGroupName(2, "patch_bottom");
                                oversampled_patch_box.setCoDim1PhysicalGroupName(3, "patch_top");
                                oversampled_patch_omega = ConstructiveSolidGeometryDomain(whole*oversampled_patch_box)
                                oversampled_patch_boundary_parts = []
                                for part in ['patch_left', 'patch_right', 'patch_bottom', 'patch_top']:
                                    try:
                                        tmp = PhysicalGroupSubDomain(oversampled_patch_omega, part)
                                    except:
                                        pass
                                    else:
                                        oversampled_patch_boundary_parts.append(part)
                                oversampled_patch_boundary = PhysicalGroupSubDomain(oversampled_patch_omega, *oversampled_patch_boundary_parts)
                                get_oversampled_patch_kk = get_kk(oversampled_patch_omega, whole*oversampled_patch_box)

                                patch_box = ConstructiveSolidGeometry(Box(Point(patch_min_x, patch_min_y), Point(patch_max_x, patch_max_y)))
                                patch_subdomain = ConstructiveSolidGeometrySubDomain2D(oversampled_patch_omega, whole*patch_box)
                                get_patch_kk = get_kk(oversampled_patch_omega, whole*patch_box)

                                myprint('                    patch [{:d}/{:d}] csg done'.format(global_ii+1, global_num_patches))

                                oversampled_patch_VV = PumSpace(oversampled_patch_omega, level=patch_level, polynomial_degree=patch_degree, stretch_factor=patch_alpha)
                                patch_dof = dim_from_coeff(oversampled_patch_VV)
                                local_spaces.append(oversampled_patch_VV)
                                local_dof += patch_dof
                                
                                if write_vtk:
                                    path = 'patch_discrete/{:s}'.format(patch_path)
                                    os.makedirs(path, exist_ok=True)
                                    with cd(path):
                                        write_particle_vtk(oversampled_patch_VV.cover(), oversampled_patch_VV.level(), r'unenriched.vtk', (lambda patch: oversampled_patch_VV.local_basis_size(patch.global_index(oversampled_patch_VV.level())), int, "local_basis_size"))


                                if ldomain:
                                    oversampled_patch_VV.enrich(eta, EnrichmentApplicationType.additive, near_corner(PatchStretchedDomain()), get_uniform_refinements(PatchLocalDomain()), should_refine_adaptive(IntegrationDomainArea(), IterationCount()))

                                for ii in range(num_inclusions):
                                    xx = Position()
                                    TT = Translation(-inclusion_centers[ii])
                                    pp = PolarTransformation()(TT(xx))
                                    rr = pp[0]
                                    theta = pp[1]

                                    uu = rr-inclusion_radii[ii]
                                    uu.set_gradient(grad(pp).T*(1, 0))
                                    uu.set_polynomial_degree(3)
                                    grad(uu).set_polynomial_degree((2,2))

                                    oversampled_patch_VV.enrich(conditional(rr-inclusion_radii[ii] > 0, uu, Constant(0)), EnrichmentApplicationType.multiplicative, boxIntersectNotInCircle(inclusion_centers[ii][0], inclusion_centers[ii][1], inclusion_radii[ii]))
                                    del xx, TT, pp, rr, theta, uu

                                oversampled_patch_bcs = []
                                oversampled_patch_zero_bcs = []
                                for (part, bc) in global_dirichlet:
                                    try:
                                        tmp = PhysicalGroupSubDomain(oversampled_patch_omega, part)
                                    except:
                                        pass
                                    else:
                                        oversampled_patch_bcs.append(DirichletBoundaryCondition(oversampled_patch_VV, tmp, bc))
                                        oversampled_patch_zero_bcs.append(DirichletBoundaryCondition(oversampled_patch_VV, tmp, null))

                                if write_vtk:
                                    path = 'patch_discrete/{:s}'.format(patch_path)
                                    os.makedirs(path, exist_ok=True)
                                    with cd(path):
                                        write_particle_vtk(oversampled_patch_VV.cover(), oversampled_patch_VV.level(), r'enriched.vtk', (lambda patch: oversampled_patch_VV.local_basis_size(patch.global_index(oversampled_patch_VV.level())), int, "local_basis_size"))

                                patch_enriched_dof = dim_from_coeff(oversampled_patch_VV)
                                local_enriched_dof += patch_enriched_dof

                                myprint('                    local space setup [{:d}] dof, [{:d}] eriched dof done'.format(patch_dof, patch_enriched_dof))

                                oversampled_patch_integration_cell_handler = IntegrationCellHandlerCover(oversampled_patch_VV.cover(), oversampled_patch_VV.level())
                                oversampled_patch_integration_cell_handler.generate_cached_cells([oversampled_patch_VV])
                                integration_cell_handlers.append(oversampled_patch_integration_cell_handler)
                                
                                myprint('                    integration cell handler done')

                                oversampled_patch_test = TestFunction(oversampled_patch_VV)
                                oversampled_patch_trial = TrialFunction(oversampled_patch_VV)

                                oversampled_patch_kk = get_oversampled_patch_kk(oversampled_patch_trial, oversampled_patch_test)
                                oversampled_patch_KK = GlobalMatrix(oversampled_patch_VV)
                                oversampled_patch_mm = inner(oversampled_patch_trial, oversampled_patch_test)*dx
                                oversampled_patch_MM = GlobalMatrix(oversampled_patch_VV)
                                oversampled_patch_stable_kk = inner(grad(oversampled_patch_trial), grad(oversampled_patch_test))*dx + inner(oversampled_patch_trial, oversampled_patch_test)*dx 
                                oversampled_patch_stable_KK = GlobalDiagonalMatrix(oversampled_patch_VV)
                                assemble((oversampled_patch_kk, oversampled_patch_KK),
                                         (oversampled_patch_mm, oversampled_patch_MM),
                                         (oversampled_patch_stable_kk, oversampled_patch_stable_KK),
                                         integration_cell_handler=oversampled_patch_integration_cell_handler)
                                oversampled_patch_st = create_stable_transformation(oversampled_patch_VV, oversampled_patch_stable_KK, epsilon=orthogonalization_tolerance)
                               
                                myprint('                    stiffness and mass matrices assembled, stable transformation created')

                                patch_nullspace = []
                                if elasticity:
                                    uu = Function(oversampled_patch_VV)
                                    for ii in range(oversampled_patch_VV.local_count()):
                                        local_size = oversampled_patch_VV.local_basis_size(ii)
                                        local_vector = DenseVector(local_size*mul)
                                        local_vector[0] = 1
                                        uu.coefficient()[ii] = local_vector
                                    patch_nullspace.append(uu)
                                    uu = Function(oversampled_patch_VV)
                                    for ii in range(oversampled_patch_VV.local_count()):
                                        local_size = oversampled_patch_VV.local_basis_size(ii)
                                        local_vector = DenseVector(local_size*mul)
                                        local_vector[local_size] = 1
                                        uu.coefficient()[ii] = local_vector
                                    patch_nullspace.append(uu)
                                    uu = Function(oversampled_patch_VV)
                                    for ii in range(oversampled_patch_VV.local_count()):
                                        local_size = oversampled_patch_VV.local_basis_size(ii)
                                        local_vector = DenseVector(local_size*mul)
                                        local_vector[2] = -1
                                        local_vector[local_size+1] = 1
                                        uu.coefficient()[ii] = local_vector
                                    patch_nullspace.append(uu)
                                else:
                                    uu = Function(oversampled_patch_VV)
                                    for ii in range(oversampled_patch_VV.local_count()):
                                        local_size = oversampled_patch_VV.local_basis_size(ii)
                                        local_vector = DenseVector(local_size*mul)
                                        local_vector[0] = 1
                                        uu.coefficient()[ii] = local_vector
                                    patch_nullspace = [uu]
                                
                                for ii, uu in enumerate(patch_nullspace):
                                    tmp  = ParallelBlockVector(uu.coefficient())
                                    for jj in range(ii):
                                        multiply_assign(tmp, oversampled_patch_MM, patch_nullspace[jj].coefficient())
                                        tmp2 = ParallelBlockVector(patch_nullspace[jj].coefficient())
                                        multiply_assign(tmp2, inner_product(uu.coefficient(), tmp))
                                        subtract_assign(uu.coefficient(), uu.coefficient(), tmp2)
                                    multiply_assign(tmp, oversampled_patch_MM, uu.coefficient())
                                    multiply_assign(uu.coefficient(), 1./np.sqrt(inner_product(uu.coefficient(), tmp)))

                                def orthogonalize_null(uu):
                                    for vv in patch_nullspace:
                                        tmp = ParallelBlockVector(uu.coefficient())
                                        multiply_assign(tmp, oversampled_patch_MM, vv.coefficient())
                                        tmp2 = ParallelBlockVector(vv.coefficient())
                                        multiply_assign(tmp2, inner_product(uu.coefficient(), tmp))
                                        subtract_assign(uu.coefficient(), uu.coefficient(), tmp2)

                                myprint('                    null space set up')
                                
                                if use_local_solutions:
                                    oversampled_patch_LL = inner(oversampled_patch_test, global_ff)*dx
                                    oversampled_patch_uu = Function(oversampled_patch_VV)
                                    tic = time.process_time()
                                    oversampled_patch_solver = solve(oversampled_patch_kk == oversampled_patch_LL, oversampled_patch_uu, oversampled_patch_bcs, oversampled_patch_st, integration_cell_handler=oversampled_patch_integration_cell_handler, solver_parameters=solver_parameters(patch_level))
                                    if not len(oversampled_patch_zero_bcs):
                                        orthogonalize_null(oversampled_patch_uu)
                                    toc = time.process_time()
                                    oversampled_patch_iterations = oversampled_patch_solver.linear_solver().last_iterations()
                                    myprint('                    local particular solution computed in [{:s}] with [{:d}] iterations'.format(human_time(toc-tic), oversampled_patch_iterations))
                                    local_solutions.append(oversampled_patch_uu)
                                    if write_enrichments:
                                        path = 'local_solutions/{:s}'.format(patch_path)
                                        os.makedirs(path, exist_ok=True)
                                        with cd(path):
                                            name = 'local_solution_l{:d}_o{:.2e}_l{:d}_d{:d}'.format(global_level, oversampling_factor, patch_level, patch_degree)
                                            write_to_pnt('{:s}.pnt'.format(name), (oversampled_patch_uu, name))
                                            if write_vtk:
                                                write_continuous_vtk(oversampled_patch_VV, r'{:s}.vtk'.format(name), (oversampled_patch_uu, name), integration_cell_handler=oversampled_patch_integration_cell_handler)

                                patch_enrichments = []
                               
                                myprint('                    setup done, computing enrichments now')
                                if enrichment_type == 'lipton' or enrichment_type == 'lipton_efendiev':
                                    myprint('                        Lipton start')
                                    patch_kk = None
                                    if enrichment_type == 'lipton':
                                        patch_kk = get_patch_kk(oversampled_patch_trial, oversampled_patch_test)
                                    elif enrichment_type == 'lipton_efendiev':
                                        if num_inclusions:
                                            patch_matrix = ConstructiveSolidGeometrySubDomain2D(oversampled_patch_omega, (whole*patch_box)-inclusions)
                                            patch_inclusions = ConstructiveSolidGeometrySubDomain2D(oversampled_patch_omega, (whole*patch_box)*inclusions)
                                            patch_kk = inner(oversampled_patch_trial, oversampled_patch_test)*dx(patch_matrix) + contrast*inner(oversampled_patch_trial, oversampled_patch_test)*dx(patch_inclusions)
                                        else:
                                            patch_kk = inner(oversampled_patch_trial, oversampled_patch_test)*dx(patch_subdomain)

                                    patch_KK = GlobalMatrix(oversampled_patch_VV)
                                    assemble((patch_kk, patch_KK), integration_cell_handler=oversampled_patch_integration_cell_handler)

                                    boundary_length = Scalar()
                                    assemble((1*ds(oversampled_patch_boundary), boundary_length, oversampled_patch_VV), integration_cell_handler = oversampled_patch_integration_cell_handler)
                                    myprint('                            boundary length {:.2e}'.format(boundary_length.value()))

                                    boundary_hats = []
                                    boundary_cutoff = 1e-10
                                    total = 0
                                    total_boundary = 0
                                    for ii in range(oversampled_patch_VV.local_count()):
                                        local_size = oversampled_patch_VV.local_basis_size(ii) #*mul
                                        total += local_size*mul
                                
                                        # check if patch lies on local patch boundary
                                        local_vector = DenseVector(local_size*mul)
                                        local_vector[0] = 1
                                        local_one = Function(oversampled_patch_VV)
                                        local_one.coefficient()[ii] = local_vector
                                        
                                        local_boundary_length = Scalar()
                                        assemble((inner(local_one, local_one)*ds(oversampled_patch_boundary), local_boundary_length, oversampled_patch_VV), integration_cell_handler = oversampled_patch_integration_cell_handler)
                                        if local_boundary_length.value() < boundary_cutoff*boundary_length.value():
                                            continue
                                        
                                        total_boundary += mul

                                        for jj in range(mul):
                                            local_vector = DenseVector(local_size*mul)
                                            local_vector[jj*local_size] = 1
                                            uu = Function(oversampled_patch_VV)
                                            uu.coefficient()[ii] = local_vector
                                            boundary_hats.append(uu)

                                    boundary_number = len(boundary_hats)
                                    myprint('                            {:d}/{:d}/{:d} functions relevant for boundary'.format(boundary_number, total_boundary, total))
                                    
                                    harmonic_hats = []
                                    oversampled_patch_LL_null = inner(oversampled_patch_test, null)*dx
                                    iterations = 0
                                    min_iterations = solver_max_iterations+1 
                                    max_iterations = -1
                                    tic = time.process_time()
                                    for ii in range(boundary_number):
                                        bc = DirichletBoundaryCondition(oversampled_patch_VV, oversampled_patch_boundary, boundary_hats[ii])
                                        uu = Function(oversampled_patch_VV)
                                        solver = solve(oversampled_patch_kk == oversampled_patch_LL_null, uu, oversampled_patch_zero_bcs+[bc], oversampled_patch_st, integration_cell_handler=oversampled_patch_integration_cell_handler, solver_parameters=solver_parameters(patch_level))
                                        it = solver.linear_solver().last_iterations()
                                        iterations += it
                                        min_iterations = np.min([min_iterations, it])
                                        max_iterations = np.max([max_iterations, it])
                                        if not len(oversampled_patch_zero_bcs):
                                            orthogonalize_null(uu)
                                        harmonic_hats.append(uu)
                                    del boundary_hats
                                    toc = time.process_time()
                                    
                                    myprint('                            Harmonic extensions computed, average [{:s}] and [{:d}] iterations per item, max it [{:d}], min it [{:d}]'.format(human_time((toc-tic)/boundary_number), int(iterations/boundary_number), min_iterations, max_iterations)) 
                                    
                                    myprint('                            Filtering out zeros')
                                    def orthogonalize(uu, uus):
                                        tmp = ParallelBlockVector(uu.coefficient())
                                        multiply_assign(tmp, oversampled_patch_KK, uu.coefficient())
                                        ret = inner_product(uu.coefficient(), tmp)
                                        if ret < ortho_eps:
                                            return 0
                                        ret = np.sqrt(ret)
                                        multiply_assign(uu.coefficient(), 1./ret)
                                        if len(uus):
                                            for vv in uus:
                                                multiply_assign(tmp, oversampled_patch_KK, vv.coefficient())
                                                tmp2 = ParallelBlockVector(vv.coefficient())
                                                multiply_assign(tmp2, inner_product(uu.coefficient(), tmp))
                                                subtract_assign(uu.coefficient(), uu.coefficient(), tmp2)
                                            multiply_assign(tmp, oversampled_patch_KK, uu.coefficient())
                                            ret = inner_product(uu.coefficient(), tmp)
                                            if ret < ortho_eps:
                                                return 0
                                            ret = np.sqrt(ret)
                                            multiply_assign(uu.coefficient(), 1./ret)
                                        return ret

                                    filtered_harmonic_hats = []
                                    count = 0
                                    for ii, uu in enumerate(harmonic_hats):
                                        if orthogonalize(uu, filtered_harmonic_hats) > 0:
                                            filtered_harmonic_hats.append(uu)
                                        else:
                                            count += 1
                                    harmonic_hats = filtered_harmonic_hats
                                    old_boundary_number = boundary_number
                                    boundary_number = len(harmonic_hats)
                                    if not boundary_number:
                                        myprint('                patch [{:d}/{:d}] harmonic hats linearly dependent'.format(global_ii+1, global_num_patches))
                                        patches_ok = False
                                        break;

                                    myprint('                            [{:d}/{:d}] functions found to be linearly dependent in Gram-Schmidt. assemble stiffness matrices with total [{:d}] functions'.format(count, old_boundary_number, boundary_number))
                                    extended_KK_harmonic = np.zeros((boundary_number, boundary_number))
                                    interior_KK_harmonic = np.zeros((boundary_number, boundary_number))

                                    for ii in range(boundary_number):
                                        tmp_full = ParallelBlockVector(harmonic_hats[ii].coefficient())
                                        tmp_inner = ParallelBlockVector(harmonic_hats[ii].coefficient())
                                        multiply_assign(tmp_full, oversampled_patch_KK, harmonic_hats[ii].coefficient())
                                        multiply_assign(tmp_inner, patch_KK, harmonic_hats[ii].coefficient())
                                
                                        extended_KK_harmonic[ii, ii] = inner_product(harmonic_hats[ii].coefficient(), tmp_full)
                                        interior_KK_harmonic[ii, ii] = inner_product(harmonic_hats[ii].coefficient(), tmp_inner)

                                        for jj in range(ii):
                                            extended_KK_harmonic[ii, jj] = inner_product(harmonic_hats[jj].coefficient(), tmp_full)
                                            interior_KK_harmonic[ii, jj] = inner_product(harmonic_hats[jj].coefficient(), tmp_inner)
                                            extended_KK_harmonic[jj, ii] = extended_KK_harmonic[ii, jj]
                                            interior_KK_harmonic[jj, ii] = interior_KK_harmonic[ii, jj]

                                        del tmp_full, tmp_inner

                                    myprint('                            matrices assembled. computing eigenvalues and eigenvector')

                                    eigvals, eigvecs = la.eigh(interior_KK_harmonic, extended_KK_harmonic)
                                    idx = np.argsort(eigvals)[::-1]
                                    nwidths = np.sqrt(eigvals[idx])
                                    ev_harmonic = eigvecs[:, idx]
                                    myprint('                            eigenvalues and eigenvectors computed')
                                    patch_number = len(nwidths)
                                    patch_enrichment_numbers = np.ones(enrichments_tols_len, dtype=int)*patch_number
                                    patch_tol_jj = 0
                                    for ii in range(1,len(nwidths)):
                                        if nwidths[ii] <= enrichments_tols[patch_tol_jj]:
                                            patch_enrichment_numbers[patch_tol_jj] = ii
                                            patch_tol_jj += 1
                                            if patch_tol_jj == enrichments_tols_len:
                                                myprint('                            numbers for all accuracies found')
                                                break;
                                    if patch_tol_jj < enrichments_tols_len:
                                        myprint('                            numbers for all accuracies not found')
                                        if patch_enrichment_numbers[patch_tol_jj] < patch_number:
                                            myprint('                                last found dof {:d} with {:.2e} for accuracy {:.2e}'.format(patch_enrichment_numbers[patch_tol_jj-1], nwidths[patch_enrichment_numbers[patch_tol_jj-1]], enrichments_tols[patch_tol_jj-1]))
                                        myprint('                                last nwidth {:.2e}, accuracies {:s}'.format(nwidths[-1], str(enrichments_tols)))
                                        if patch_tol_jj < 1:
                                            myprint('                                found no dof for coarsest accuracy [{:.2e}]'.format(enrichments_tols[0]))
                                            patch_tol_jj = 1
                                        enrichments_tols = enrichments_tols[:patch_tol_jj]
                                        enrichments_tols_len = len(enrichments_tols)
                                        myprint('                                remaining accuracies {:s}'.format(str(enrichments_tols)))
                                        assert(enrichments_tols_len == patch_tol_jj)
                                    enrichment_numbers.append(patch_enrichment_numbers)
                                    patch_number = patch_enrichment_numbers[-1]
                                    myprint('                            dofs {:s} for accuracies {:s}'.format(str(patch_enrichment_numbers), str(np.array(enrichments_tols))))

                                    for ii in range(patch_number):
                                        uu = Function(oversampled_patch_VV)
                                        for jj in range(boundary_number):
                                            tmp = ParallelBlockVector(harmonic_hats[jj].coefficient())
                                            multiply_assign(tmp, ev_harmonic[jj, ii])
                                            add_assign(uu.coefficient(), tmp)
                                            del tmp
                                        patch_enrichments.append(uu)
                             
                                    del harmonic_hats
                                    myprint('                        Lipton end')

                                elif enrichment_type == 'efendiev':
                                    myprint('                        Efendiev start')
                                    oversampled_patch_nn = None
                                    if num_inclusions:
                                        oversampled_patch_matrix = ConstructiveSolidGeometrySubDomain2D(oversampled_patch_omega, (whole*oversampled_patch_box)-inclusions)
                                        oversampled_patch_inclusions = ConstructiveSolidGeometrySubDomain2D(oversampled_patch_omega, (whole*oversampled_patch_box)*inclusions)
                                        oversampled_patch_nn = inner(oversampled_patch_trial, oversampled_patch_test)*dx(oversampled_patch_matrix) + contrast*inner(oversampled_patch_trial, oversampled_patch_test)*dx(oversampled_patch_inclusions)
                                    else:
                                        oversampled_patch_nn = inner(oversampled_patch_trial, oversampled_patch_test)*dx(oversampled_patch_omega)

                                    oversampled_patch_NN = GlobalMatrix(oversampled_patch_VV)
                                    assemble((oversampled_patch_nn, oversampled_patch_NN), integration_cell_handler=oversampled_patch_integration_cell_handler)
                                    
                                    solver = PetscSolver()

                                    KK = ParallelSparseBlockMatrix(oversampled_patch_KK)
                                    oversampled_patch_st.to_transformed_entries(KK)
                                    NN = ParallelSparseBlockMatrix(oversampled_patch_NN)
                                    oversampled_patch_st.to_transformed_entries(NN)

                                    eigensolver = SlepcEigenvalueSolver()
                                    eigensolver.set_solver(solver)
                                    eigensolver.set_operators(KK, NN)
                                    eigensolver.set_tolerance(solver_tolerance)
                                    myprint('                            eigensolver constructed')
                                    if number+null_dim > patch_enriched_dof:
                                        number = patch_enriched_dof-null_dim
                                    myprint('                            eigensolver computing [{:d}] nwidth functions'.format(number))
                                    eigensolver.solve(number+null_dim, EigenvalueSolverSpectrum.smallest_magnitude)
                                    converged_number = eigensolver.last_converged_count()-null_dim
                                    myprint('                            eigensolver computed [{:d}] nwidth functions'.format(converged_number))
                                    patch_tol_jj = 0
                                    ii = 1
                                    patch_number = patch_enriched_dof-null_dim
                                    patch_enrichment_numbers = np.ones(enrichments_tols_len, dtype=int)*patch_number
                                    myprint('                            checking eigenvalues against accuracies')
                                    while True:
                                        nwidth = 1./np.sqrt(eigensolver.eigenvalue(ii+null_dim))
                                        if nwidth <= enrichments_tols[patch_tol_jj]:
                                            myprint('                            dof [{:d}], nwidth [{:.2e}] for accuracy [{:.2e}]'.format(ii, nwidth, enrichments_tols[patch_tol_jj]))
                                            patch_enrichment_numbers[patch_tol_jj] = ii
                                            patch_tol_jj += 1
                                            if patch_tol_jj == enrichments_tols_len:
                                                break;
                                        ii += 1
                                        if ii >= converged_number:
                                            number = int(1.5*converged_number+1)
                                            if number+null_dim > patch_enriched_dof:
                                                number = patch_enriched_dof-null_dim
                                            if ii >= number:
                                                break
                                            myprint('                            last computed nwidth [{:.2e}], next accuracy [{:.2e}]'.format(nwidth, enrichments_tols[patch_tol_jj]))
                                            myprint('                            eigensolver computing [{:d}] nwidth functions'.format(number))
                                            eigensolver.solve(number+null_dim, EigenvalueSolverSpectrum.smallest_magnitude)
                                            converged_number = eigensolver.last_converged_count()-null_dim
                                            myprint('                            eigensolver computed [{:d}] nwidth functions'.format(converged_number))
                                            if converged_number <= ii:
                                                break
                                    if patch_tol_jj < enrichments_tols_len:
                                        myprint('                            numbers for all accuracies not found')
                                        if patch_enrichment_numbers[patch_tol_jj-1] < patch_number:
                                            myprint('                                last found dof {:d} with nwidth {:.2e} for accuracy {:.2e}'.format(patch_enrichment_numbers[patch_tol_jj-1], 1./np.sqrt(eigensolver.eigenvalue(patch_enrichment_numbers[patch_tol_jj-1]+null_dim)), enrichments_tols[patch_tol_jj-1]))
                                        myprint('                                last nwidth {:.2e}, accuracies {:s}'.format(nwidth, str(enrichments_tols)))
                                        if patch_tol_jj < 1:
                                            myprint('                                found no dof for coarsest accuracy [{:.2e}]'.format(enrichments_tols[0]))
                                            patch_tol_jj = 1
                                        enrichments_tols = enrichments_tols[:patch_tol_jj]
                                        enrichments_tols_len = len(enrichments_tols)
                                        myprint('                                remaining accuracies {:s}'.format(str(enrichments_tols)))
                                        assert(enrichments_tols_len == patch_tol_jj)
                                    enrichment_numbers.append(patch_enrichment_numbers)
                                    patch_number = patch_enrichment_numbers[-1]

                                    myprint('                            {:d} eigenpairs computed'.format(eigensolver.last_converged_count()))
                                    values = []
                                    for ii in range(patch_number):
                                        uu = Function(oversampled_patch_VV)
                                        oversampled_patch_st.to_transformed_coefficients(uu.coefficient())    
                                        value = eigensolver.eigenpair(ii+null_dim, uu.coefficient())
                                        oversampled_patch_st.from_transformed_coefficients(uu.coefficient())
                                        values.append(value)
                                        patch_enrichments.append(uu)
                                    if patch_number+null_dim < converged_number:
                                        values.append(eigensolver.eigenvalue(patch_number))
                                    
                                    myprint('                            plotting eigenvalues')
                                    nwidths = 1./np.sqrt(np.array(values[1:]))

                                    myprint('                        Efendiev end')
                                else:
                                    myprint('                        UNIMPLEMENTED')

                                myprint('                            plotting eigenvalues/nwidths')

                                path = 'nwidths/{:s}'.format(patch_path)
                                os.makedirs(path, exist_ok=True)
                                with cd(path):
                                    title = '{:s}_o{:.2e}_l{:d}_l{:d}_d{:d}'.format(enrichment_type, oversampling_factor, global_level, patch_level, patch_degree)
                                    nwidth_file = open('{:s}.csv'.format(enrichment_type), 'w')
                                    nwidth_file.write('dof, nwidth\n')
                                    for ii, nwidth in enumerate(nwidths):
                                        nwidth_file.write('{:d}, {:.3e}\n'.format(ii+1, nwidth))
                                    nwidth_file.close()

                                    fig = plt.figure()
                                    ax = fig.add_subplot(111)
                                    ax.loglog(range(1, len(nwidths)), nwidths[1:], 'o:')
                                    ax.set_title('{:s} n-width loglog'.format(title))
                                    ax.grid(True)
                                    fig.tight_layout()
                                    fig.savefig('{:s}_loglog.pdf'.format(enrichment_type))

                                    fig = plt.figure()
                                    ax = fig.add_subplot(111)
                                    ax.semilogy(range(1, len(nwidths)), nwidths[1:], 'o:')
                                    ax.set_title('{:s} n-width semilogy'.format(title))
                                    ax.grid(True)
                                    fig.tight_layout()
                                    fig.savefig('{:s}_semilogy.pdf'.format(enrichment_type))
                                    
                                plt.close('all')

                                local_enrichments.append(patch_enrichments)
                                patch_numbers.append(patch_number)

                                gc.collect()

                            if not patches_ok:
                                myprint('            [{:d}/{:d}], patch degree [{:d}] ended prematurely'.format(patch_degree_ii+1, len(patch_degrees), patch_degree))
                                continue

                            if write_enrichments:
                                myprint('                Writing enrichments')
                                for global_ii in range(global_num_patches):
                                    patch_name = 'patch{:s}'.format(str(global_ii).zfill(global_num_fill))
                                    patch_path = 'oversamplingfactor{:.2e}/globallevel{:d}/patchlevel{:d}/patchdegree{:d}/{:s}'.format(oversampling_factor, global_level, patch_level, patch_degree, patch_name)
                                    path = 'enrichments/{:s}/{:s}'.format(enrichment_type, patch_path)
                                    os.makedirs(path, exist_ok=True)
                                    with cd(path):
                                        name = '{:s}_l{:d}_o{:.2e}_l{:d}_d{:}'.format(enrichment_type, global_level, oversampling_factor, patch_level, patch_degree)
                                        write_to_pnt('{:s}.pnt'.format(name), *[(local_enrichments[global_ii][ii], '{:s}_{:d}'.format(enrichment_type, ii)) for ii in range(patch_numbers[global_ii])])
                                        if write_vtk:
                                            write_continuous_vtk(oversampled_patch_VV, r'{:s}.vtk'.format(name), *[(local_enrichments[global_ii][ii], '{:s}_{:d}'.format(enrichment_type, ii)) for ii in range(patch_numbers[global_ii])], integration_cell_handler=integration_cell_handlers[global_ii])
                                myprint('                Wrote enrichments') 
                            
                            myprint('            Computing global enriched solutions')
                            diff_handler = None
                            for global_degree_ii, global_degree in enumerate(global_degrees):
                                myprint('                degree [{:d}], [{:d}/{:d}] start'.format(global_degree, global_degree_ii+1, len(global_degrees)))
                                path = 'enriched_solutions/{:s}/oversamplingfactor{:.2e}/globallevel{:d}/globaldegree{:d}/patchlevel{:d}/patchdegree{:d}'.format(enrichment_type, oversampling_factor, global_level, global_degree, patch_level, patch_degree)
                                os.makedirs(path, exist_ok=True)
                                with cd(path):
                                    dofs = []
                                    l2_errors = []
                                    h1_errors = []
                                    energy_errors = []

                                    for tol_ii, tol in enumerate(enrichments_tols):
                                        myprint('                    tol [{:.2e}], [{:d}/{:d}] start'.format(tol, tol_ii+1, enrichments_tols_len))
                                        name = 'tol{:.2e}'.format(tol)
                                        os.makedirs(name, exist_ok=True)
                                        with cd(name):
                                            global_VV = PumSpace(global_omega, level=global_level, polynomial_degree=global_degree, stretch_factor=global_alpha)

                                            myprint('                        Enriching patches')
                                            for global_ii in range(global_num_patches):
                                                num = enrichment_numbers[global_ii][tol_ii]
                                                global_VV.enrich(local_enrichments[global_ii][:num]+[local_solutions[global_ii]], EnrichmentApplicationType.additive, within_point_box(center_points[global_ii], PatchStretchedDomain()))
                                                myprint('                            patch [{:d}/{:d}] enriched with [{:d}] functions'.format(global_ii+1, global_num_patches, num))
                                            global_enriched_dof = dim_from_coeff(global_VV)
                                            myprint('                        Enriched patches')
                                            if diff_handler is None:
                                                diff_handler = get_diff_handler(global_VV, local_spaces=local_spaces)

                                            global_bcs = []
                                            for (part, bc) in global_dirichlet:
                                                try:
                                                    tmp = PhysicalGroupSubDomain(global_omega, part)
                                                except:
                                                    myprint('                            Boundary part [{:s}] not found, skipping')
                                                else:
                                                    global_bcs.append(DirichletBoundaryCondition(global_VV, tmp, bc))

                                            dofs.append(global_enriched_dof)
                                            myprint('                        Global space of [{:d}] dof enriched to [{:d}] dof using computations with total [{:d}] dof [{:d}] enriched dof'.format(global_dof, global_enriched_dof, local_dof, local_enriched_dof))

                                            myprint('                        Constructing global integration cell handler')
                                            tic = time.process_time()
                                            global_integration_cell_handler = IntegrationCellHandlerCover(global_VV.cover(), global_VV.level())
                                            global_integration_cell_handler.generate_cached_cells([global_VV]+local_spaces)
                                            toc = time.process_time()
                                            myprint('                        Constructed global integration cell handler in [{:s}]'.format(human_time(toc-tic)))

                                            myprint('                        Constructing stable transformation')
                                            tic = time.process_time()
                                            global_test = TestFunction(global_VV)
                                            global_trial = TrialFunction(global_VV)

                                            global_kk = get_global_kk(global_trial, global_test)
                                            global_stable_kk = get_global_l2(global_trial, global_test) + get_global_h1(global_trial, global_test)
                                            global_stable_KK = GlobalDiagonalMatrix(global_VV)
                                            assemble((global_stable_kk, global_stable_KK), integration_cell_handler=global_integration_cell_handler)
                                            global_st = create_stable_transformation(global_VV, global_stable_KK, epsilon=orthogonalization_tolerance)
                                            toc = time.process_time()
                                            myprint('                        Constructed stable transformation in [{:s}]'.format(human_time(toc-tic)))

                                            global_LL = inner(global_test, global_ff)*dx

                                            myprint('                        Global solve starting')
                                            global_uu = Function(global_VV)
                                            tic = time.process_time()
                                            global_solver = solve(global_kk == global_LL, global_uu, global_bcs, global_st, integration_cell_handler=global_integration_cell_handler, solver_parameters=solver_parameters(global_level))
                                            toc = time.process_time()
                                            global_iterations = global_solver.linear_solver().last_iterations()
                                            myprint('                        Global solve finished in [{:s}] with [{:d}] iterations'.format(human_time(toc-tic), global_iterations))

                                            name = 'solution_l{:d}_d{:d}_{:s}_l{:d}_d{:d}'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree)
                                            write_to_pnt('{:s}.pnt'.format(name), (global_uu, name))
                                            if write_vtk:
                                                write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (global_uu, name), integration_cell_handler=global_integration_cell_handler)

                                            local_name = '{:s}_error'.format(reference_name)
                                            os.makedirs(local_name, exist_ok = True)
                                            with cd(local_name):
                                                name = '{:s}_l{:d}_d{:d}_l{:d}_d{:d}'.format(local_name, global_level, global_degree, patch_level, patch_degree)
                                                write_to_pnt('{:s}.pnt'.format(name), (global_uu-reference_uu, name), integration_cell_handler=global_integration_cell_handler)
                                                if write_vtk:
                                                    write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (global_uu-reference_uu, name), integration_cell_handler=global_integration_cell_handler)
 
                                            myprint('                        Extracting local contributions')
                                            local_contribution = Function(global_VV)
                                            for ii in range(global_VV.local_count()):
                                                local_size = global_VV.local_basis_size(ii) #*mul
                                                local_vector = DenseVector(local_size*mul)
                                                for jj in range(mul):
                                                    local_vector[(jj+1)*local_size-1] = 1
                                                local_contribution.coefficient()[ii] = local_vector

                                            local_diff = Function(global_VV)
                                            subtract_assign(local_diff.coefficient(), global_uu.coefficient(), local_contribution.coefficient())

                                            local_name = 'local_solution'
                                            os.makedirs(local_name, exist_ok = True)
                                            with cd(local_name):
                                                name = '{:s}_l{:d}_d{:d}_l{:d}_d{:d}'.format(local_name, global_level, global_degree, patch_level, patch_degree)
                                                write_to_pnt('{:s}.pnt'.format(name), (local_contribution, name))
                                                if write_vtk:
                                                    write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (local_contribution, name), integration_cell_handler=global_integration_cell_handler)
                                     
                                            local_diff = Function(global_VV)
                                            subtract_assign(local_diff.coefficient(), global_uu.coefficient(), local_contribution.coefficient())

                                            local_name = 'diff_solution'
                                            os.makedirs(local_name, exist_ok = True)
                                            with cd(local_name):
                                                name0 = 'solution_l{:d}_d{:d}_{:s}_l{:d}_d{:d}'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree)
                                                name1 = '{:s}_l{:d}_d{:d}_l{:d}_d{:d}'.format(local_name, global_level, global_degree, patch_level, patch_degree)
                                                name = name0+'-'+name1
                                                write_to_pnt('{:s}.pnt'.format(name), (local_diff, name))
                                                if write_vtk:
                                                    write_continuous_vtk(global_VV, r'{:s}.vtk'.format(name), (local_diff, name), integration_cell_handler=global_integration_cell_handler)

                                            if compute_errors:
                                                myprint('                        Computing errors')
                                                l2_rel, h1_rel, energy_rel = reference_errors(global_uu, diff_handler)
                                                myprint('                            dof:    {:d}'.format(dofs[-1]))
                                                myprint('                            L2:     {:.2e}'.format(l2_rel))
                                                myprint('                            H1:     {:.2e}'.format(h1_rel))
                                                myprint('                            energy: {:.2e}'.format(energy_rel))
                                                l2_errors.append(l2_rel)
                                                h1_errors.append(h1_rel)
                                                energy_errors.append(energy_rel)
                                                exclusive_write(enriched_error_file, '{:s}, {:.3e}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:.3e}, {:.3e}, {:.3e}\n'.format(enrichment_type, oversampling_factor, global_level, global_degree, patch_level, patch_degree, global_enriched_dof, global_iterations, l2_rel, h1_rel, energy_rel))
                                            
                                            del global_VV, global_bcs, global_test, global_trial, global_kk, global_stable_kk, global_stable_KK, global_st, global_LL, global_solver, global_uu

                                        gc.collect()
                                        myprint('                    tol [{:.2e}], [{:d}/{:d}] end'.format(tol, tol_ii+1, enrichments_tols_len))

                                    if compute_errors:
                                        dofs = np.array(dofs); l2_errors = np.array(l2_errors); h1_errors = np.array(h1_errors); energy_errors = np.array(energy_errors)
                                        
                                        fig = plt.figure()
                                        ax = fig.add_subplot(111)
                                        ax.set_title('Errors for global level [{:d}], global degree [{:d}], {:s}, local level [{:d}], local degree [{:d}]'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree))
                                        ax.loglog(dofs, l2_errors, 'go:', label=r'$\|\cdot\|_2$')
                                        ax.loglog(dofs, h1_errors, 'bo-.', label=r'$|\cdot|_1$')
                                        ax.loglog(dofs, energy_errors, 'ro-', label=r'$|\cdot|_a$')
                                        ax.legend(loc=1)
                                        fig.savefig('enriched_errors_l{:d}_d{:d}_{:s}_l{:d}_d{:d}_loglog.pdf'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree))
                                        fig = plt.figure()
                                        ax = fig.add_subplot(111)
                                        ax.set_title('Errors for global level [{:d}], global degree [{:d}], {:s}, local level [{:d}], local degree [{:d}]'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree))
                                        ax.semilogy(dofs, l2_errors, 'go:', label=r'$\|\cdot\|_2$')
                                        ax.semilogy(dofs, h1_errors, 'bo-.', label=r'$|\cdot|_1$')
                                        ax.semilogy(dofs, energy_errors, 'ro-', label=r'$|\cdot|_a$')
                                        ax.legend(loc=1)
                                        fig.savefig('enriched_errors_l{:d}_d{:d}_{:s}_l{:d}_d{:d}_semilogy.pdf'.format(global_level, global_degree, enrichment_type, patch_level, patch_degree))
                                        plt.close('all')
                                myprint('                degree [{:d}], [{:d}/{:d}] end'.format(global_degree, global_degree_ii+1, len(global_degrees)))
                            myprint('            Computed global enriched solutions')
                            myprint('            [{:d}/{:d}], patch degree [{:d}] end'.format(patch_degree_ii+1, len(patch_degrees), patch_degree))
                        del diff_handler
                        gc.collect()
                        myprint('        [{:d}/{:d}], patch level [{:d}] end'.format(patch_level_ii+1, len(patch_levels), patch_level))
                    gc.collect()
                    myprint('    [{:d}/{:d}], oversampling factor [{:.2e}] end'.format(oversampling_factor_ii+1, len(oversampling_factors), oversampling_factor))
                gc.collect()
                myprint('[{:d}/{:d}], level [{:d}] end'.format(global_level_ii+1, len(global_levels), global_level))
            enriched_error_file.close()
            myprint('Computed optimally enriched solutions')

    toc = time.process_time()
    myprint('Done in [{:s}]'.format(human_time(toc-global_tic)))

if __name__ == '__main__':
    elasticity = False
    enrichment_type = 'lipton'
    reference_level = 10
    reference_degree = 2
    max_tol = 8
    compute_unenriched_solutions = False
    compute_geometrically_enriched_solutions = False
    compute_enriched_solutions = True 
    min_global_level = 1
    max_global_level = 3
    min_global_degree = 2
    max_global_degree = 2
    min_patch_level = 4
    max_patch_level = 8
    min_patch_degree = 2
    max_patch_degree = 2
    with_inclusions = True
    ldomain = True
    compute_reference = False
    basedim = 2

    def usage():
        print('''Usage
    python3 puma_lipton.py
    Option              Default             Info
    -h, --help                              help
    -e, --elasticity                        elasticity, otherwise scalar heat
    -u, --unenriched                        compute unenriched solutions
    -g, --geometric                         compute geometrically enriched solutions
    -n, --noenriched                        don't compute optimally enriched solutions
    -3                                      3 dimensional
    --computereference  1                   force computation of reference
    --withinclusions    1                   with inclusions
    --enrichmenttype    lipton              type of enrichment
    --mingloballevel    1                   minimal global level
    --maxgloballevel    3                   maximal global level
    --minglobaldegree   2                   minimal global polynomial dgree                
    --maxglobaldegree   2                   maximal global polynomial degree
    --minpatchlevel     1                   minimal patch level
    --maxpatchlevel     3                   maximal patch level
    --minpatchdegree    2                   minimal patch polynomial dgree                
    --maxpatchdegree    2                   maximal patch polynomial degree
    --referencelevel    10                  level of reference solution
    --referencedegree   2                   polynomial degree of reference solution
    --maxtol            4                   maximal nwidth accuracy to seek
    --ldomain           1                   L-domain?
    ''')

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'heugn3',
                                   ['help', 'elasticity', 'unenriched', 'geometric', 'noenriched',
                                    'mingloballevel=', 'maxgloballevel=',
                                    'minglobaldegree=', 'maxglobaldegree=',
                                    'minpatchlevel=', 'maxpatchlevel=',
                                    'minpatchdegree=', 'maxpatchdegree=',
                                    'referencelevel=', 'referencedegree=', 'enrichmenttype=', 'maxtol=',
                                    'withinclusions=', 'ldomain=', 'computereference='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'): 
            usage()
            sys.exit()
        elif opt in ('-e', '--elasticity'):
            elasticity = True
        elif opt in ('-u', '--unenriched'):
            compute_unenriched_solutions = True
        elif opt in ('-g', '--geometric'):
            compute_geometrically_enriched_solutions = True
        elif opt in ('-n', '--noenriched'):
            compute_enriched_solutions = False
        elif opt == '-3':
            basedim = 3
        elif opt == '--withinclusions':
            with_inclusions = int(arg) > 0
        elif opt == '--enrichmenttype':
            enrichment_type = arg
        elif opt == '--mingloballevel':
            min_global_level = int(arg)
        elif opt == '--maxgloballevel':
            max_global_level = int(arg)
        elif opt == '--minglobaldegree':
            min_global_degree = int(arg)
        elif opt == '--maxglobaldegree':
            max_global_degree = int(arg)
        elif opt == '--minpatchlevel':
            min_patch_level = int(arg)
        elif opt == '--maxpatchlevel':
            max_patch_level = int(arg)
        elif opt == '--minpatchdegree':
            min_patch_degree = int(arg)
        elif opt == '--maxpatchdegree':
            max_patch_degree = int(arg)
        elif opt == '--referencelevel':
            reference_level = int(arg)
        elif opt == '--referencedegree':
            reference_degree = int(arg)
        elif opt == '--maxtol':
            max_tol = int(arg)
        elif opt == '--ldomain':
            ldomain = int(arg) > 0
        elif opt == '--computereference':
            compute_reference = int(arg) > 0
        else:
            print('unhandled option')
            sys.exit(2)

    calculate_stuff(elasticity, enrichment_type=enrichment_type,
                    global_levels=np.arange(min_global_level, max_global_level+1, dtype=int),
                    global_degrees=np.arange(min_global_degree, max_global_degree+1, dtype=int),
                    patch_levels=np.arange(min_patch_level, max_patch_level+1, 2, dtype=int),
                    patch_degrees=np.arange(min_patch_degree, max_patch_degree+1, dtype=int),
                    compute_unenriched_solutions=compute_unenriched_solutions, compute_geometrically_enriched_solutions=compute_geometrically_enriched_solutions,
                    compute_enriched_solutions=compute_enriched_solutions,
                    reference_level=reference_level, reference_degree=reference_degree,
                    enrichments_tols=10.**(-np.arange(1, max_tol+1)),
                    with_inclusions=with_inclusions, ldomain=ldomain,
                    compute_reference_uu=compute_reference,
                    basedim=basedim)


