import os, sys, getopt, time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import functools

from puma import *
from paunt.utility.internal import _compile_feature

import numpy as np
import matplotlib.pyplot as plt

import sa_hdf5

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


def usage():
    print('''Usage
python3 nodeadlock.py
Option              Default             Info
-h --help                               help
-e --elasticity                         vector linear elasticity, else scalar heat
-f --finecg                             use cg for finest level, direct solver otherwise
-w --writediff                          write difference to reference
-o --old                                old paunt solver parameters
-u                  10                  uniform number of enrichments
--pumcoarsestlevel= 2                   coarse pum level, 2 for 3x3, no refine
--pumfinestlevel=   3                   fine pum level, 3 for 2-level solve
--pumdegree=        1                   degree of polynomials in local spaces
--basistype=        liptonE             type of enrichment used
--meshname=         square_ring         prefix of meshes enrichments to load
--meshresolution=   6                   mesh resolution of enrichments
--patchlevel=       0                   patch level of enrichments
--contrast=         1e4                 contrast of problem
--beta=             2.0                 oversampling factor of enrichments
--meshdir=          fem                 root path to look for meshes
--enrichmentsdir=   enrichments         root path to look for enrichment
--bccase=           0                   boundary condition case
--refresolution=    12                  resolution of reference''')

if __name__ == '__main__':
    pum_coarsest_level = 2
    pum_finest_level = 3
    pum_degree = 1
    basis_type = 'liptonE'
    mesh_name = "square_few_balls"
    mesh_resolution = 5
    patch_level = 0
    contrast = 1e4
    beta = 1.4
    finecg = False
    elasticity = False
    mesh_dir = '/data/wu/homogenization/fem_inside_refined'
    enrichments_dir = '/data/wu/homogenization/enrichments_inside_refined'
    bccase = 0
    ldomain = False
    ref_resolution = 12
    write_diff = False
    calc_errors = True
    alpha=1.25
    albert=True
    uniform=False

    try:
        opts, args = getopt.getopt(sys.argv[1:],'hefwou',
                                   ['help', 'elasticity', 'finecg', 'old', 'pumcoarsestlevel=', 'pumfinestlevel=',
                                    'pumdegree=', 'basistype=', 'meshname=', 'meshresolution=',
                                    'patchlevel=', 'contrast=', 'beta=', 'meshdir=', 'enrichmentsdir=', 'bccase=',
                                    'refresolution=', 'writediff'])
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
        elif opt in ('-f', '--finecg'):
            finecg = True
        elif opt in ('-w', '--writediff'):
            write_diff = True
        elif opt in ('-a', '--albert'):
            albert = False
        elif opt in ('-u', '--uniform'):
            uniform = True
        elif opt == '--pumcoarsestlevel':
            pum_coarsest_level = int(arg) if int(arg) > 0 else pum_coarsest_level
            if pum_coarsest_level > pum_finest_level:
                pum_finest_level = pum_coarsest_level
        elif opt == '--pumfinestlevel':
            pum_finest_level = int(arg) if int(arg) > 0 else pum_finest_level
            if pum_coarsest_level > pum_finest_level:
                pum_coarsest_level = pum_finest_level
        elif opt == '--pumdegree':
            pum_degree = int(arg) if int(arg) >= 0 else pum_degree
        elif opt == '--basistype':
            basis_type = arg
        elif opt == '--meshname':
            mesh_name = arg
            if mesh_name == 'ldomain':
                ldomain = True
        elif opt == '--meshresolution':
            mesh_resolution = int(arg) if int(arg) >= 0 else mesh_resolution
        elif opt == '--patchlevel':
            patch_level = int(arg) if int(arg) >= 0 else patch_level
        elif opt == '--contrast':
            contrast = float(arg) if float(arg) > 0 else contrast
        elif opt == '--beta':
            beta = float(arg) if float(arg) > 1 else beta
        elif opt == '--meshdir':
            mesh_dir = arg
        elif opt == '--enrichmentsdir':
            enrichments_dir = arg
        elif opt == '--bccase':
            bccase = int(arg) if int(arg) >= 0 else bccase
        elif opt == '--refresolution':
            ref_resolution = int(arg) if int(arg) > 0 else ref_resolution
        else:
            print('unhandled option')
            sys.exit(2)

    print("""python3 newer_run enrichments
pum coarsest level      {:d}
pum finest level        {:d}
pum degree              {:d}
basis type              {:s}
mesh name               {:s}
mesh resolution         {:d}
patch level             {:d}
contrast                {:.2e}
beta                    {:.2e}
finecg                  {:s}
elasticity              {:s}
mesh dir                {:s}
enrichments dir         {:s}
bc case                 {:d}
reference resolution    {:d}
write diff              {:s}
old syntax              {:s}""".format(pum_coarsest_level, pum_finest_level, pum_degree, 
                                       basis_type, mesh_name, mesh_resolution, patch_level, contrast,
                                       beta, str(finecg), str(elasticity), mesh_dir, enrichments_dir,
                                       bccase, ref_resolution, str(write_diff), str(albert)))

    if elasticity:
        problem_type = 'elasticity'
        data_name = 'vector'
    else:
        problem_type = 'heat'
        data_name = 'scalar'

    print("####    START")
    Logger.instance().enable_to_file()
    Logger.instance().enable_to_stdout(True)

    # CHANGED
    # The overall bounding box in hdf5_minimal seems to be bigger than before
    omega_min = Point(-1, -1)
    omega_max = Point( 1,  1)
    omega_box = Box(omega_min, omega_max)
    center = 0.5*(omega_min+omega_max)

    patch_boxes = np.genfromtxt('{:s}/{:s}/{:s}_patch_descriptors/{:d}.csv'.format(mesh_dir, mesh_name, mesh_name, patch_level), delimiter = ', ', names = True)
    patch_num = len(patch_boxes['idx'])
    patch_fill = int(np.log(patch_num)/np.log(10.))+1

    enrichments_count = (3*int(2**patch_level),3*int(2**patch_level))

    def to_patch_coordinates(ii):
        x = int(ii%enrichments_count[0])
        y = int(ii/enrichments_count[0])
        return (x,y)

    if 4 <= bccase < 6:
        if ldomain:
            def is_patch_at_dirichlet_boundary(xx):
                return (xx[1] == 0) or (xx[1] == enrichments_count[1]-1) or (xx[0] == int(enrichments_count[0]/2) and xx[1] >= int(enrichments_count[1]/2)) or (xx[1] == int(enrichments_count[1]/2) and xx[0] >= int(enrichments_count[0]/2))
        else:
            def is_patch_at_dirichlet_boundary(xx):
                return (xx[0] == 0) or (xx[0] == enrichments_count[0]-1) or (xx[1] == 0) or (xx[1] == enrichments_count[1]-1)

        def is_patch_at_neumann_boundary(xx):
            return False
    else:
        def is_patch_at_dirichlet_boundary(xx):
            return (xx[0] == 0) or (xx[0] == enrichments_count[0]-1)
        
        def is_patch_at_neumann_boundary(xx):
            return (xx[1] == 0) or (xx[1] == enrichments_count[1]-1)

    case_paths = None
    if ldomain:
        if beta < 2:
            if 0 <= bccase < 4:
                case_paths = [
                    ['D_N_______', '__N_______', '_DN_______'],
                    ['D_________', '______NN__', '_D_____N__'],
                    ['D__N______', '___N__N___', '__________']
                ]
            else:
                case_paths = [
                    ['D_D_______', '__D_______', '_DD_______'],
                    ['D_________', '__________', '_D________'],
                    ['D__D______', '___D______', '__________']
                ]
        else:
            if 0 <= bccase < 4:
                case_paths = [
                    ['D_N___NN__', '__N___NN__', '_DN___NN__'],
                    ['D_____NN__', '______NN__', '_D____NN__'],
                    ['D__N__NN__', '___N__NN__', '______NN__']
                ]
            else:
                case_paths = [
                    ['D_D___DD__', '__D___DD__', '_DD___DD__'],
                    ['D_____DD__', '______DD__', '_D____DD__'],
                    ['D__D__DD__', '___D__DD__', '__________']
                ]
    else:
        if 0 <= bccase < 4:
            case_paths = [
                ['D_N_______', '__N_______', '_DN_______'],
                ['D_________', '__________', '_D________'],
                ['D__N______', '___N______', '_D_N______']
            ]
        else:
            case_paths = [
                ['D_D_______', '__D_______', '_DD_______'],
                ['D_________', '__________', '_D________'],
                ['D__D______', '___D______', '_D_D______']
            ]
    nu = 0.3
    E = 1
    E_inc = E*contrast

    ##########################################
    # Define domain, subdomains, physical groups
    ##########################################
    paunt_print("generate geometry...")
    if ldomain:
        box = ConstructiveSolidGeometry(Box(Point(-1, -1), Point(1, 1)))
        box.setCoDim1PhysicalGroupName(0, "Left");
        box.setCoDim1PhysicalGroupName(1, "Right");
        box.setCoDim1PhysicalGroupName(2, "Bottom");
        box.setCoDim1PhysicalGroupName(3, "Top");
        cut = ConstructiveSolidGeometry(Box(Point(0, 0), Point(1, 1)))
        cut.setCoDim1PhysicalGroupName(0, "Corner_Right");
        cut.setCoDim1PhysicalGroupName(2, "Corner_Top");
        omega = ConstructiveSolidGeometryDomain(box - cut)
        del box, cut
    else:
        omega = RectangleDomain(omega_min, omega_max)
    omega.set_bounding_box(Box(omega_min, omega_min + (omega_max - omega_min) * 4 / 3))

    gamma_left = PhysicalGroupSubDomain(omega, "Left")
    gamma_right = PhysicalGroupSubDomain(omega, "Right")
    gamma_bottom = PhysicalGroupSubDomain(omega, "Bottom")
    gamma_top = PhysicalGroupSubDomain(omega, "Top")
    omega_inc = PhysicalGroupSubDomain(omega)
    omega_mat = PhysicalGroupSubDomain(omega)

    def on_gamma_left(xx):
       return xx[0] == omega_box.min_corner()[0]
    def on_gamma_right(xx):
       return xx[0] == omega_box.max_corner()[0]
    def on_gamma_bottom(xx):
        return xx[1] == omega_box.min_corner()[1]
    def on_gamma_top(xx):
        return xx[1] == omega_box.max_corner()[1]
    if ldomain:
        gamma_corner_right = PhysicalGroupSubDomain(omega, "Corner_Right")
        gamma_corner_top = PhysicalGroupSubDomain(omega, "Corner_Top")
        eps = 1e-12
        def on_gamma_corner_right(xx):
            return (abs(xx[0]-center[0]) < eps) & (xx[1] > center[1]-eps)
        def on_gamma_corner_top(xx):
            return (abs(xx[1]-center[1]) < eps) & (xx[0] > center[0]-eps)

    if uniform:
        accuracies = [10]
    else:
        accuracies = [1e-0, 1e-1, 1e-2, 1e-3]

    print('READ REFERENCE')
    if bccase == 4 and ldomain:
        u_ref = 
    else:
        ff = sa_hdf5.XDMFFile('{:s}/{:s}/{:s}_{:d}_cells.xdmf'.format(mesh_dir, mesh_name, mesh_name, ref_resolution))
        mesh = sa_hdf5.Mesh()
        ff.read(mesh)
        del ff
        if elasticity:
            u_ref = sa_hdf5.read_dolfin_vector_cg1('{:s}/{:s}/{:s}/case0_{:.2e}/coeffs/krylov_{:s}_{:d}_{:s}_{:.2e}_{:d}'.format(mesh_dir, mesh_name, problem_type, contrast, mesh_name, ref_resolution, problem_type, contrast, bccase), mesh=mesh)
        else:
            u_ref = sa_hdf5.read_dolfin_scalar_cg1('{:s}/{:s}/{:s}/case0_{:.2e}/coeffs/krylov_{:s}_{:d}_{:s}_{:.2e}_{:d}'.format(mesh_dir, mesh_name, problem_type, contrast, mesh_name, ref_resolution, problem_type, contrast, bccase), mesh=mesh)
        u_ref = u_ref[0]
        V_ref = u_ref.function_space()
        num_vertices_ref = V_ref.mesh().num_vertices()
        dim_ref = V_ref.dim()
        print('    Ref vertices: {:d}, dof: {:d}'.format(num_vertices_ref, dim_ref))
        coords_ref = mesh.coordinates()
        map_ref = sa_hdf5.vertex_to_dof_map(V_ref)

    ##########################################
    # Define material model
    ##########################################
    material0 = LinearElasticMaterialModel(omega_mat, E=E, nu=nu)
    material1 = LinearElasticMaterialModel(omega_inc, E=E_inc, nu=nu)
    material = MultiMaterialModel([material0, material1])

    full_mesh_dir = '{:s}/{:s}/{:s}_{:d}_patches/{:d}'.format(mesh_dir, mesh_name, mesh_name, mesh_resolution, patch_level)
    full_enrichments_dir = '{:s}/{:s}/{:s}/contrast_{:.2e}/patchlevel_{:d}/{{:s}}/res_{:d}/beta_{:.2e}'.format(enrichments_dir, mesh_name, problem_type, contrast, patch_level, mesh_resolution, beta)

    enrichment_meshes = []
    enrichment_disjoint_box_mesh_pairs = []
    enrichment_support_boxes = []
    enrichment_function_spaces = []

    for ii in range(0, patch_num):
        patch_name = r'patch_'+str(ii).zfill(patch_fill)
        base_name = r'{:s}_{:d}_{:d}_{:s}'.format(mesh_name, mesh_resolution, patch_level, patch_name)

        patch_path = full_enrichments_dir.format(patch_name)

        patch_coordinates = to_patch_coordinates(ii)
        (x, y) = patch_coordinates
        paunt_print("x = %f ; y = %f" % (x, y))

        # CHANGED
        paunt_print("read mesh '{:s}/{:s}_cells.xdmf'".format(full_mesh_dir, patch_name))
        mesh = read_mesh_2d('{:s}/{:s}_cells.xdmf'.format(full_mesh_dir, patch_name))
        V_FE = FemSpace(mesh)
        enrichment_function_spaces.append(V_FE)

        paunt_print("mark mesh facets...")
        xx = Position()
        marker = MeshVertexMarker(mesh, on_gamma_left(xx))
        paunt_print("left marked: %d" % marker.count_marked())
        mesh.add_facet_regions(marker, gamma_left)
        marker = MeshVertexMarker(mesh, on_gamma_right(xx))
        paunt_print("right marked: %d" % marker.count_marked())
        mesh.add_facet_regions(marker, gamma_right)
        marker = MeshVertexMarker(mesh, on_gamma_bottom(xx))
        paunt_print("bottom marked: %d" % marker.count_marked())
        mesh.add_facet_regions(marker, gamma_bottom)
        marker = MeshVertexMarker(mesh, on_gamma_top(xx))
        paunt_print("top marked: %d" % marker.count_marked())
        mesh.add_facet_regions(marker, gamma_top)
        if ldomain:
            marker = MeshVertexMarker(mesh, on_gamma_corner_right(xx))
            paunt_print("corner right marked: %d" % marker.count_marked())
            mesh.add_facet_regions(marker, gamma_corner_right)
            marker = MeshVertexMarker(mesh, on_gamma_corner_top(xx))
            paunt_print("corner top marked: %d" % marker.count_marked())
            mesh.add_facet_regions(marker, gamma_corner_top)

        paunt_print("mark inclusions...")
        marker = read_mesh_cell_marker_xdmf(mesh, '{:s}/{:s}_cells.xdmf'.format(full_mesh_dir, patch_name), "cell_function", 0)
       #paunt_print("marked: %d" % marker.count_marked())
        mesh.add_cell_regions(marker, omega_mat)
        
        paunt_print("mark matrix...")
        marker = read_mesh_cell_marker_xdmf(mesh, '{:s}/{:s}_cells.xdmf'.format(full_mesh_dir, patch_name), "cell_function", 1)
       #paunt_print("marked: %d" % marker.count_marked())
        mesh.add_cell_regions(marker, omega_inc)

        enrichment_meshes.append(mesh)
        enrichment_support_boxes.append(mesh.bounding_box())

        disjoint_box = Box(omega_min + Point((omega_max[0] - omega_min[0])/enrichments_count[0]*x, (omega_max[1] - omega_min[1])/enrichments_count[1]*y), omega_min + Point((omega_max[0] - omega_min[0])/enrichments_count[0]*(x+1), (omega_max[1] - omega_min[1])/enrichments_count[1]*(y+1)))
        paunt_print("box %d = (%e, %e) -> (%e, %e)" % (ii, disjoint_box.min_corner()[0], disjoint_box.min_corner()[1], disjoint_box.max_corner()[0], disjoint_box.max_corner()[1]))

        paunt_print("bbox %d = (%e, %e) -> (%e, %e)" % (ii, mesh.bounding_box().min_corner()[0], mesh.bounding_box().min_corner()[1], mesh.bounding_box().max_corner()[0], mesh.bounding_box().max_corner()[1]))

        enrichment_disjoint_box_mesh_pairs.append((disjoint_box, mesh))

    ##########################################
    # Generate discretization space
    ##########################################
    pum_resolution = max(4, 2 * pum_degree + 1)
    V = MultilevelPumSpace(omega, level=pum_coarsest_level, polynomial_degree=pum_degree, stretch_factor=alpha, use_patch_centers_for_domain_tests=ldomain)

    paunt_print("Refining Pum Space...")
    Vs = []
    for level in range(pum_coarsest_level, pum_finest_level+1):
        V.provide_level(level)
        V_k = V.subspace(level)

        Vs.append(V_k)
        Vs.append(V_k.function_space())

        paunt_print("dof(V_%d) = %d" % (level, V_k.global_dof(elasticity)))

    V_finest = V.subspace(V.finest_level())

    for level in range(pum_coarsest_level, pum_finest_level+1):
        V_k = V.subspace(level)
        write_continuous_vtk(V_k, 'patches_lvl_{:d}'.format(level))

    paunt_print("generating cells...")
    tic = time.process_time()
    integration_cell_handler = IntegrationCellHandlerMultiMeshCover(enrichment_disjoint_box_mesh_pairs, V_finest.function_space().cover(), V_finest.level())
    integration_cell_handler.generate_cached_cells(Vs)
    toc = time.process_time()
    paunt_print('integration cells assembled in [{:s}]'.format(human_time(toc-tic)))

    l2_errors = np.empty((len(accuracies), pum_finest_level-pum_coarsest_level+1), dtype=float)
    h1_errors = np.empty((len(accuracies), pum_finest_level-pum_coarsest_level+1), dtype=float)
    dofs = np.empty((len(accuracies), pum_finest_level-pum_coarsest_level+1), dtype=int)

    for ii_acc, accuracy in enumerate(accuracies):
#       patch_accuracy = accuracy
        patch_accuracy = (accuracy*(alpha-1.))/(2*alpha*np.sqrt(3*2**(pum_coarsest_level-2)))
    #   patch_accuracy = (accuracy*(alpha-1.))/(2*alpha)#*np.sqrt(3*2**(pum_finest_level-2)))
#       paunt_print('Modified accuracy per patch [{:.2e}] -> [{:.2e}]'.format(accuracy, patch_accuracy))

        ##########################################
        # read enrichments
        ##########################################
        paunt_print('reading enrichments')

        full_mesh_dir = '{:s}/{:s}/{:s}_{:d}_patches/{:d}'.format(mesh_dir, mesh_name, mesh_name, mesh_resolution, patch_level)
        full_enrichments_dir = '{:s}/{:s}/{:s}/contrast_{:.2e}/patchlevel_{:d}/{{:s}}/res_{:d}/beta_{:.2e}'.format(enrichments_dir, mesh_name, problem_type, contrast, patch_level, mesh_resolution, beta)

        enrichments_per_patch_count = np.zeros(patch_num, dtype=int)
        enrichments = []
        for ii in range(0, patch_num):
            patch_name = r'patch_'+str(ii).zfill(patch_fill)
            base_name = r'{:s}_{:d}_{:d}_{:s}'.format(mesh_name, mesh_resolution, patch_level, patch_name)

            patch_path = full_enrichments_dir.format(patch_name)

            patch_coordinates = to_patch_coordinates(ii)
            (x, y) = patch_coordinates
            paunt_print("x = %f ; y = %f" % (x, y))

            enrichment_paths = []
            enrichment_names = []   
            if bccase in [1,3]:
                enrichment_paths.append(('{:s}/coeffs/{:s}_0_f.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))          
            elif bccase == 5:
                enrichment_paths.append(('{:s}/coeffs/{:s}_2_f.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))          

            if is_patch_at_dirichlet_boundary(patch_coordinates):
                if 0 <= bccase < 2:
                    pass
                elif 2 <= bccase < 4:
                    enrichment_paths.append(('{:s}/coeffs/{:s}_1_g.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))
                elif bccase == 4:
                    enrichment_paths.append(('{:s}/coeffs/{:s}_2_g.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))
                enrichment_names.append('{:d}_dirichlet'.format(ii))
            else:
                if 6 <= bccase < 8:
                    enrichment_paths.append(('{:s}/coeffs/{:s}_0_h.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))          
                elif 8 <= bccase < 10:
                    enrichment_paths.append(('{:s}/coeffs/{:s}_1_h.h5'.format(patch_path, base_name), '{:s}/0'.format(data_name)))

            case_path = '{:s}/{:s}'.format(patch_path, case_paths[y][x])

            supinfs = np.genfromtxt('{:s}/{:s}_{:s}_supinfs.csv'.format(case_path, base_name, basis_type), delimiter = ', ', names = True)
            for idx in range(1, len(supinfs['dof'])):
                if supinfs['supinfE'][idx] < patch_accuracy:
                    enrichments_per_patch_count[ii] = supinfs['dof'][idx]
                    paunt_print('patch {:d}, type {:s}: {:.2e} < {:.2e} for global {:.2e} accuracy predicted with {:d} enrichments'.format(ii, basis_type, supinfs['supinfE'][idx], patch_accuracy, accuracy, enrichments_per_patch_count[ii]))
                    break;
            if not enrichments_per_patch_count[ii]:
                enrichments_per_patch_count[ii] = supinfs['dof'][-1]
                paunt_print('patch {:d}, type {:s}: {:.2e} > {:.2e} for global {:.2e} accuracy predicted with {:d} enrichments'.format(ii, basis_type, supinfs['supinfE'][-1], patch_accuracy, accuracy, enrichments_per_patch_count[ii]))
            if uniform:
                paunt_print('overriding number of enrichments')
                enrichments_per_patch_count[ii] = 10

            for jj in range(enrichments_per_patch_count[ii]):
                enrichment_paths.append(('{:s}/coeffs/{:s}_shapes_{:s}.h5'.format(case_path, base_name, basis_type), '{:s}/{:d}'.format(data_name, jj)))
                enrichment_names.append('{:d}_{:s}_{:d}'.format(ii, basis_type, jj))

            for path in enrichment_paths:
                print(path)

            enrichment = EnrichmentSpaceFem2D(enrichment_function_spaces[ii], enrichment_paths, enrichment_names, elasticity)
            enrichments.append(enrichment)

        print('number enrichments:')
        for ii, epp in enumerate(enrichments_per_patch_count):
            print('    patch {:d}: {:d}'.format(ii, epp))

        for level in range(pum_coarsest_level, pum_finest_level+1):
            V_k = V.subspace(level).function_space()
            V_k.clear_enrichments()
            
        for i in range(0, patch_num):

            # patch_min_x = Max(PatchStretchedDomain().min_corner()[0], omega_box.min_corner()[0])
            # patch_min_y = Max(PatchStretchedDomain().min_corner()[1], omega_box.min_corner()[1])

            # patch_max_x = Min(PatchStretchedDomain().max_corner()[0], omega_box.max_corner()[0])
            # patch_max_y = Min(PatchStretchedDomain().max_corner()[1], omega_box.max_corner()[1])

            # MAKE STRETCHED_DOMAIN ACTUALLY LOCAL_DOMAIN
            patch_min_x = PatchStretchedDomain().min_corner()[0]
            patch_min_y = PatchStretchedDomain().min_corner()[1]
            patch_max_x = PatchStretchedDomain().max_corner()[0]
            patch_max_y = PatchStretchedDomain().max_corner()[1]

            patch_center_x = (patch_max_x + patch_min_x) / 2
            patch_radius_x = (patch_max_x - patch_min_x) / 2
            patch_radius_x = patch_radius_x / alpha
            patch_center_y = (patch_max_y + patch_min_y) / 2
            patch_radius_y = (patch_max_y - patch_min_y) / 2
            patch_radius_y = patch_radius_y / alpha

            patch_min_x = patch_center_x - patch_radius_x
            patch_max_x = patch_center_x + patch_radius_x
            patch_min_y = patch_center_y - patch_radius_y
            patch_max_y = patch_center_y + patch_radius_y

            patch_min_x = Max(patch_min_x, omega_box.min_corner()[0])
            patch_min_y = Max(patch_min_y, omega_box.min_corner()[1])
            patch_max_x = Min(patch_max_x, omega_box.max_corner()[0])
            patch_max_y = Min(patch_max_y, omega_box.max_corner()[1])

            # paunt_print("bbbox %d = (%e, %e) -> (%e, %e)" % (i, patch_min_x, patch_min_y, patch_max_x, patch_max_y))

            #++++++++++++++++++++++++++
            # switched shoud_enrich functions
            #++++++++++++++++++++++++++
            #should_enrich = within_box_box_touches(MyBox((patch_min_x, patch_min_y), (patch_max_x, patch_max_y)), enrichment_support_boxes[i])
            #should_enrich = within_box_box_touches(MyBox((patch_min_x, patch_min_y), (patch_max_x, patch_max_y)), enrichment_disjoint_box_mesh_pairs[i][0])
            #should_enrich = within_point_box((patch_center_x, patch_center_y), enrichment_support_boxes[i])
            if ldomain:
                should_enrich = within_point_box((patch_center_x, patch_center_y), enrichment_disjoint_box_mesh_pairs[i][0])&(~within_point_box((patch_center_x, patch_center_y), Box(center, omega_max)))
            else:
                should_enrich = within_point_box((patch_center_x, patch_center_y), enrichment_disjoint_box_mesh_pairs[i][0])
            compiled_feature = _compile_feature(should_enrich, V_finest.domain_dimension(), V_finest.parallel_partition().communicator())
            V.enrich(enrichments[i], compiled_feature)

        for level in range(pum_coarsest_level, pum_finest_level+1):
            V_k = V.subspace(level)
            paunt_print("ENRICHED dof(V_%d) = %d" % (level, V_k.global_dof(elasticity)))
            write_continuous_vtk(V_k, 'patches_lvl_{:d}'.format(level))

        ##########################################
        # Define stable trafo
        ##########################################

        paunt_print("generating stable transformation...")
        st_M = GlobalDiagonalMatrix(V_finest)
        if elasticity:
            u = VectorTrialFunction(V, with_phi=False)
            v = VectorTestFunction(V, with_phi=False)
        else:
            u = ScalarTrialFunction(V, with_phi=False)
            v = ScalarTestFunction(V, with_phi=False)
        # assemble((material.second_variation_of_strain_energy(u, v, None) + inner(u, v)*dx, st_M), integration_cell_handler=integration_cell_handler)
        # assemble((dot(E * grad(u), grad(v))*dx(omega_inc) + dot(E_inc * grad(u), grad(v))*dx(omega_mat) + u*v*dx, st_M), integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        # assemble((E*u*v*dx(omega_inc) + E_inc*u*v*dx(omega_mat), st_M), integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        # assemble((dot(grad(u), grad(v))*dx + u*v*dx, st_M), integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        tic = time.process_time()
        assemble((inner(grad(u), grad(v))*dx + inner(u,v)*dx, st_M), integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        toc = time.process_time()
        paunt_print('stable transfomration assembled in [{:s}]'.format(human_time(toc-tic)))
        st = create_stable_transformation(V_finest, st_M, epsilon=1e-10)

        # paunt_print("dof(V) = %d ; dof(stable V) = %d" % (V.global_dof(True), V.global_dof(True, st)))
        paunt_print("dof(V_finest) = %d ; dof(stable V_finest) = %d" % (V_finest.global_dof(elasticity), V_finest.global_dof(elasticity, st)))

        ##########################################
        # Define Dirichlet boundary conditions
        ##########################################
        # g_left = (-1e-2, 0)
        # g_right = (1e-2, 0)
        yy = Position()
        cc = 0.5
        if 0 <= bccase < 2:
            if elasticity:
                bcs = [DirichletBoundaryCondition(V, gamma_left, (-cc, 0)), DirichletBoundaryCondition(V, gamma_right, (cc, 0))]
            else:
                bcs = [DirichletBoundaryCondition(V, gamma_left, -cc), DirichletBoundaryCondition(V, gamma_right, cc)]
        elif 2 <= bccase < 4:
            if elasticity:
                bcs = [DirichletBoundaryCondition(V, gamma_left, (cc*(1.-yy[1]*yy[1]), 0)), DirichletBoundaryCondition(V, gamma_right, (cc*(yy[1]*yy[1]-1.), 0))]
            else:
                bcs = [DirichletBoundaryCondition(V, gamma_left, cc*(1.-yy[1]*yy[1])), DirichletBoundaryCondition(V, gamma_right, cc*(yy[1]*yy[1]-1.))]
        elif 4 <= bccase < 6:
            if bccase == 4:
                if elasticity:
                    exprs = (cc*yy[0]*(1-yy[1]), cc*yy[1]*(1-yy[0]))
                else:
                    exprs = 1.+0.25*(yy[0]+1.)*(yy[0]+1.)+0.5*(yy[1]+1.)*(yy[1]+1.)
            else:
                if elasticity:
                    exprs = (0., 0.)
                else:
                   exprs = 0.
            bcs = [DirichletBoundaryCondition(V, gamma_left, exprs),
                   DirichletBoundaryCondition(V, gamma_right, exprs),
                   DirichletBoundaryCondition(V, gamma_bottom, exprs),
                   DirichletBoundaryCondition(V, gamma_top, exprs)]
            if ldomain:
                bcs.append(DirichletBoundaryCondition(V, gamma_corner_right, exprs))
                bcs.append(DirichletBoundaryCondition(V, gamma_corner_top, exprs))
        elif 6 <= bccase < 10:
            bcs = []


        ##########################################
        # Define loading conditions
        ##########################################
        if elasticity:
            f = (0., 0.)
        else:
            f = 0
        if bccase in [1, 3, 7, 9]:
            if elasticity:
                f = (0, cc) 
            else:
                f = cc 
        elif bccase in [4, 5]:
            if elasticity:
                f = (cc*(material0.lmbda()+material0.mu()), cc*(material0.lmbda()+material0.mu()))
            else:
                f = -3/2.

        ##########################################
        # Define variational problem
        ##########################################
        if elasticity:
            deltau = VectorTrialFunction(V)
            v = VectorTestFunction(V)
            a = material.second_variation_of_strain_energy(deltau, v, u)
            L = dot(f, v)*dx 
            u = VectorFunction(V)
        else:
            deltau = ScalarTrialFunction(V)
            v = ScalarTestFunction(V)
            a = dot(E * grad(deltau), grad(v))*dx(omega_mat) + dot(E_inc * grad(deltau), grad(v))*dx(omega_inc)
            L = f*v*dx 
            u = ScalarFunction(V_finest)

        ##########################################
        # solve variational problem
        ##########################################
        paunt_print("solving problem...")
        if pum_coarsest_level == pum_finest_level:
            if finecg:
                solver_parameters = {'linear_solver' : {  'type' : 'cg.hypre',
                                                          'package' : 'petsc', # for direct then type lu 'package' : 'petsc.mumps',
                                                        # 'preconditioner' : {'type' : '' }
                                                        'quiet' : False,
                                                        'max_iterations' : 25000,
                                                        # 'tolerance' : 1e-8,
                                                        # 'tolerance_is_absolute' : False,
                                                        # 'convergence_test_factor' : 0,
                                                        # 'convergence_test_iterations' : 0,
                                                        # 'preconditioner' : {'type' : 'bgaussseidel',
                                                        #                     'symmetric' : True,
                                                        #                     },
                                                         }
                                     }
            else:
                if albert:
                    solver_parameters = {'linear_solver' : { 'type' : 'preonly' } }
                else:
                    solver_parameters = {'linear_solver' : { 'type' : 'lu',
                                                             'package' : 'petsc.pastix' } }
        else:
            if albert:
                solver_parameters = {
                    'linear_solver' : {
                        'type' : 'cg',
                        'quiet' : False,
                        'max_iterations' : 500,
                        'tolerance' : 1e-10,
                        'tolerance_is_absolute' : True,
                        'convergence_test_factor' : 0,
                        'convergence_test_iterations' : 0,
                        'preconditioner' : {
                            'type' : 'multilevel',
                            'coarsest_level' : pum_coarsest_level,
                            'finest_level' : pum_finest_level,
                            'smoother' : { 'type' : 'bgaussseidel', 'symmetric' : False },
                            'symmetric' : True,
                            'pre_smoothing_steps' : 10,
                            'post_smoothing_steps' : 10,
                            'coarse_level_solver' : { 'type' : 'preonly' }
                        }
                    },
                    'solution_initialization' : 'prolongate_last'
                }
            else:
                solver_parameters = {
                    'linear_solver' : {
                        'type' : 'cg',
                        'quiet' : False,
                        'max_iterations' : 500,
                        'tolerance' : 1e-10,
                        'tolerance_is_absolute' : True,
                        'convergence_test_factor' : 0,
                        'convergence_test_iterations' : 0,
                        'preconditioner' : {
                            'type' : 'multilevel',
                            'coarsest_level' : pum_coarsest_level,
                            'finest_level' : pum_finest_level,
                            'smoother' : { 'type' : 'bgaussseidel', 'symmetric' : False },
                            'symmetric' : True,
                            'pre_smoothing_steps' : 10,
                            'post_smoothing_steps' : 10,
                            'coarse_level_solver' : {
                                'type' : 'lu',
                                'package' : 'petsc.pastix'
                            }
                        }
                    },
                    'solution_initialization' : 'prolongate_last'
                }
        tic = time.process_time()
        solver = solve(a == L, u, bcs, st, solver_parameters=solver_parameters, integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        toc = time.process_time()
        # solve(a == L, u, bcs, st, integration_cell_handler=integration_cell_handler, fixed_resolution=pum_resolution)
        paunt_print('System solved in [{:s}]'.format(human_time(toc-tic)))
        print("#### OUTPUT ")

        paunt_print("writing output...")
        if pum_coarsest_level < pum_finest_level:
            for (level, u) in solver.u_per_level().items():
                V_k = V.subspace(level)
                st_k = solver.st_per_level()[level]

                if level == pum_finest_level:
                    def local_basis_size(patch):
                        return V_k.local_basis_size(patch.global_index(V_k.level()))

                    def local_stable_basis_size(patch):
                        if st is not None:
                            transformed_size = st_k.transformed_local_size(patch.global_index(V_k.level()))
                            if transformed_size is not None:
                                return transformed_size
                            else:
                                return local_basis_size(patch)
                        else:
                            return local_basis_size(patch)
                    write_particle_vtk(V_k.function_space().cover(), V_k.level(), r'particles_acc{:.2e}_ml{:d}-{:d}_l{:d}_bc{:d}.vtu'.format(accuracy, pum_coarsest_level, pum_finest_level, V_k.level(), bccase), (local_basis_size, int, "local basis size"), (local_stable_basis_size, int, "local stable basis size"))
                    write_continuous_vtk(V.subspace(level), r'solution_acc{:.2e}_ml{:d}-{:d}_l{:d}_bc{:d}.vtk'.format(accuracy, pum_coarsest_level, pum_finest_level,level, bccase), (u, "u"), integration_cell_handler=integration_cell_handler)
        else:
            def local_basis_size(patch):
                return V_finest.local_basis_size(patch.global_index(V_finest.level()))

            def local_stable_basis_size(patch):
                if st is not None:
                    transformed_size = st.transformed_local_size(patch.global_index(V_finest.level()))
                    if transformed_size is not None:
                        return transformed_size
                    else:
                        return local_basis_size(patch)
                else:
                    return local_basis_size(patch)

            write_particle_vtk(V_finest.function_space().cover(), V_finest.level(), r'particles_acc{:.2e}_l{:d}_bc{:d}.vtu'.format(accuracy, V_finest.level(), bccase), (local_basis_size, int, "local basis size"), (local_stable_basis_size, int, "local stable basis size"))
            write_continuous_vtk(V_finest, r'solution_acc{:.2e}_l{:d}_bc{:d}.vtk'.format(accuracy, pum_finest_level, bccase), (u, "u"), integration_cell_handler=integration_cell_handler)

        print('#### ERROR COMPUTATION')
        if bccase == 4 and ldomain:
            if pum_coarsest_level < pum_finest_level:
                for (level, u) in solver.u_per_level().items():
                    V_k = V.subspace(level)
                    st_k = solver.st_per_level()[level]
                    dofs[ii_acc, level-pum_coarsest_level] = V_k.global_dof(elasticity, st)

                    u_interp = sa_hdf5.Function(V_ref, label='u_interp')
                    u_vec = np.empty(dim_ref)
                    tic = time.process_time()
                    if elasticity:
                        for ii in range(num_vertices_ref):
                            ret = u.evaluate(Vector2(*coords_ref[ii]))
                            u_vec[map_ref[2*ii]] = ret[0]
                            u_vec[map_ref[2*ii+1]] = ret[1]
                    else:
                        for ii in range(dim_ref):
                            u_vec[map_ref[ii]] = u.evaluate(Vector2(*coords_ref[ii]))
                    toc = time.process_time()
                    print(human_time(toc-tic))
                    u_interp.vector().set_local(u_vec)
                    ff = sa_hdf5.XDMFFile('u_interp_acc{:.2e}_l{:d}({:d}-{:d}).xdmf'.format(accuracy, level, pum_coarsest_level, pum_finest_level))
                    ff.write(u_interp)
                 
                    diff = sa_hdf5.Function(V_ref, label='u_ref-u_interp')
                    diff.assign(u_ref-u_interp)
                    if write_diff:
                        ff = sa_hdf5.XDMFFile('u_ref-u_interp_acc{:.2e}_l{:d}({:d}-{:d}).xdmf'.format(accuracy, level, pum_coarsest_level, pum_finest_level))
                        ff.write(diff)

                    l2err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(diff, diff)*sa_hdf5.dx))
                    h1err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(sa_hdf5.grad(diff), sa_hdf5.grad(diff))*sa_hdf5.dx))
                    l2_errors[ii_acc, level-pum_coarsest_level] = l2err
                    h1_errors[ii_acc, level-pum_coarsest_level] = h1err
                    print('acc: {:.2e}, lvl: {:d}/{:d}-{:d}, dofs: {:d}, L2: {:.2e}, H1: {:.2e}'.format(accuracy, level, pum_coarsest_level, pum_finest_level, dofs[ii_acc, level-pum_coarsest_level], l2err, h1err))
            else:
                dofs[ii_acc, pum_finest_level-pum_coarsest_level] = V_finest.global_dof(elasticity, st)

                u_interp = sa_hdf5.Function(V_ref, label='u_interp')
                u_vec = np.empty(dim_ref)
                tic = time.process_time()
                if elasticity:
                    for ii in range(num_vertices_ref):
                        ret = u.evaluate(Vector2(*coords_ref[ii]))
                        u_vec[map_ref[2*ii]] = ret[0]
                        u_vec[map_ref[2*ii+1]] = ret[1]
                else:
                    for ii in range(dim_ref):
                        u_vec[map_ref[ii]] = u.evaluate(Vector2(*coords_ref[ii]))
                u_interp.vector().set_local(u_vec)
                ff = sa_hdf5.XDMFFile('u_interp_acc{:.2e}.xdmf'.format(accuracy))
                ff.write(u_interp)
             
                diff = sa_hdf5.Function(V_ref, label='u_ref-u_interp')
                diff.assign(u_ref-u_interp)
                if write_diff:
                    ff = sa_hdf5.XDMFFile('u_ref-u_interp_acc{:.2e}.xdmf'.format(accuracy))
                    ff.write(diff)

                l2err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(diff, diff)*sa_hdf5.dx))
                h1err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(sa_hdf5.grad(diff), sa_hdf5.grad(diff))*sa_hdf5.dx))
                l2_errors[ii_acc, pum_finest_level-pum_coarsest_level] = l2err
                h1_errors[ii_acc, pum_finest_level-pum_coarsest_level] = h1err
                print('acc: {:.2e}, lvl: {:d}, dofs: {:d}, L2: {:.2e}, H1: {:.2e}'.format(accuracy, pum_finest_level, dofs[ii_acc, pum_finest_level-pum_coarsest_level], l2err, h1err))


        else:
            if pum_coarsest_level < pum_finest_level:
                for (level, u) in solver.u_per_level().items():
                    V_k = V.subspace(level)
                    st_k = solver.st_per_level()[level]
                    dofs[ii_acc, level-pum_coarsest_level] = V_k.global_dof(elasticity, st)

                    u_interp = sa_hdf5.Function(V_ref, label='u_interp')
                    u_vec = np.empty(dim_ref)
                    tic = time.process_time()
                    if elasticity:
                        for ii in range(num_vertices_ref):
                            ret = u.evaluate(Vector2(*coords_ref[ii]))
                            u_vec[map_ref[2*ii]] = ret[0]
                            u_vec[map_ref[2*ii+1]] = ret[1]
                    else:
                        for ii in range(dim_ref):
                            u_vec[map_ref[ii]] = u.evaluate(Vector2(*coords_ref[ii]))
                    toc = time.process_time()
                    print(human_time(toc-tic))
                    u_interp.vector().set_local(u_vec)
                    ff = sa_hdf5.XDMFFile('u_interp_acc{:.2e}_l{:d}({:d}-{:d}).xdmf'.format(accuracy, level, pum_coarsest_level, pum_finest_level))
                    ff.write(u_interp)
                 
                    diff = sa_hdf5.Function(V_ref, label='u_ref-u_interp')
                    diff.assign(u_ref-u_interp)
                    if write_diff:
                        ff = sa_hdf5.XDMFFile('u_ref-u_interp_acc{:.2e}_l{:d}({:d}-{:d}).xdmf'.format(accuracy, level, pum_coarsest_level, pum_finest_level))
                        ff.write(diff)

                    l2err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(diff, diff)*sa_hdf5.dx))
                    h1err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(sa_hdf5.grad(diff), sa_hdf5.grad(diff))*sa_hdf5.dx))
                    l2_errors[ii_acc, level-pum_coarsest_level] = l2err
                    h1_errors[ii_acc, level-pum_coarsest_level] = h1err
                    print('acc: {:.2e}, lvl: {:d}/{:d}-{:d}, dofs: {:d}, L2: {:.2e}, H1: {:.2e}'.format(accuracy, level, pum_coarsest_level, pum_finest_level, dofs[ii_acc, level-pum_coarsest_level], l2err, h1err))
            else:
                dofs[ii_acc, pum_finest_level-pum_coarsest_level] = V_finest.global_dof(elasticity, st)

                u_interp = sa_hdf5.Function(V_ref, label='u_interp')
                u_vec = np.empty(dim_ref)
                tic = time.process_time()
                if elasticity:
                    for ii in range(num_vertices_ref):
                        ret = u.evaluate(Vector2(*coords_ref[ii]))
                        u_vec[map_ref[2*ii]] = ret[0]
                        u_vec[map_ref[2*ii+1]] = ret[1]
                else:
                    for ii in range(dim_ref):
                        u_vec[map_ref[ii]] = u.evaluate(Vector2(*coords_ref[ii]))
                u_interp.vector().set_local(u_vec)
                ff = sa_hdf5.XDMFFile('u_interp_acc{:.2e}.xdmf'.format(accuracy))
                ff.write(u_interp)
             
                diff = sa_hdf5.Function(V_ref, label='u_ref-u_interp')
                diff.assign(u_ref-u_interp)
                if write_diff:
                    ff = sa_hdf5.XDMFFile('u_ref-u_interp_acc{:.2e}.xdmf'.format(accuracy))
                    ff.write(diff)

                l2err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(diff, diff)*sa_hdf5.dx))
                h1err = np.sqrt(sa_hdf5.assemble(sa_hdf5.inner(sa_hdf5.grad(diff), sa_hdf5.grad(diff))*sa_hdf5.dx))
                l2_errors[ii_acc, pum_finest_level-pum_coarsest_level] = l2err
                h1_errors[ii_acc, pum_finest_level-pum_coarsest_level] = h1err
                print('acc: {:.2e}, lvl: {:d}, dofs: {:d}, L2: {:.2e}, H1: {:.2e}'.format(accuracy, pum_finest_level, dofs[ii_acc, pum_finest_level-pum_coarsest_level], l2err, h1err))

        print('ACC {:d}, {:.2e} done\n'.format(ii_acc, accuracy))
  
    l2min = np.min(l2_errors); l2max = np.max(l2_errors)
    h1min = np.min(h1_errors); h1max = np.max(h1_errors)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('dofs')
    ax.set_ylabel('approximate l2 error')
    for ii_acc, acc in enumerate(accuracies):
        ax.loglog(dofs[ii_acc], l2_errors[ii_acc], 'o:', label='{:.2e}'.format(acc))
    sa_hdf5.sa_utils.set_log_ticks(ax, l2min, l2max)
    ax.legend(loc=0)
    fig.savefig('l2_errors_loglog.pdf')

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('dofs')
    ax.set_ylabel('approximate h1 error')
    for ii_acc, acc in enumerate(accuracies):
        ax.loglog(dofs[ii_acc], h1_errors[ii_acc], 'o:', label='{:.2e}'.format(acc))
    sa_hdf5.sa_utils.set_log_ticks(ax, h1min, h1max)
    ax.legend(loc=0)
    fig.savefig('h1_errors_loglog.pdf')

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('dofs')
    ax.set_ylabel('approximate l2 error')
    for ii_acc, acc in enumerate(accuracies):
        ax.semilogy(dofs[ii_acc], l2_errors[ii_acc], 'o:', label='{:.2e}'.format(acc))
    sa_hdf5.sa_utils.set_log_ticks(ax, l2min, l2max)
    ax.legend(loc=0)
    fig.savefig('l2_errors_semilogy.pdf')

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel('dofs')
    ax.set_ylabel('approximate h1 error')
    for ii_acc, acc in enumerate(accuracies):
        ax.semilogy(dofs[ii_acc], h1_errors[ii_acc], 'o:', label='{:.2e}'.format(acc))
    sa_hdf5.sa_utils.set_log_ticks(ax, h1min, h1max)
    ax.legend(loc=0)
    fig.savefig('h1_errors_semilogy.pdf')


    print("####    END")
