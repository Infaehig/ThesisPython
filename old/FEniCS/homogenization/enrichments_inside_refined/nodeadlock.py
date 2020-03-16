import os, sys, getopt
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import localizations_nitsche

def usage():
    print('''Usage
python3 nodeadlock.py
Option          Default                 Info
-h --help                               help
-d --debug                              debug mode
-l --ldomain                            ldomain, different dirichlet bc
--prefix=       square                  sets prefix and meshdir to ../fem_inside_refined/prefix
--meshdir=      ../fem_inside/prefix    meshdir
--patchlevel=   0                       patch refinement level starting with 3x3
--level=        5                       patch mesh level
--beta=         2.0                     oversampling factor
--contrast=     1.0                     contrast of inclusions, 1=no inclusions
--enrichments   2                       which enrichments to compute
                                        1   all
                                        2   liptonE, subspaceE, efendiev_ext, harmonic_polys
                                        3   liptonE, efendiev_ext, harmonic_polys
--batchlogging  0                       log batch multiprocessing
--batchprocs    None                    limit number processes for batch multiprocessing
--maxkeys=      None                    limit number processes for key multiprocessing
--dofmax=       100                     How many enrichments to compute in scalar case
--patches=      None                    which patch to compute
                                        None all
--noheat                                no heat
--noelasticity                          no elasticity
--holedirichlet                         consider Dirichlet bc at hole
--krylov=      1                        krylov solver instead of direct solver
--krylovneumann= 1                      krylov solver for neumann bc
--krylovharmonic= 1                     krylov solver for hat function harmonic stuff
--lagrangeneumannharmonic=              neumann_extended_harmonic function with lagrange multipliers
--normalizenullorthogonalize=           normalize_null with orthogonalization with respect to kernel
--orthogonalizehats= 0                   orthogonalize harmonic extensions of hat functions
--interiororthmatrix=0                  orthogonalize with interior stiffness
--outdirsuffix=                         outdir suffix''')

if __name__ == '__main__':
    prefix = 'square'
    meshdir = '../fem_inside_refined/'+prefix
    patchlevel = 0
    level = 5
    beta = 2.0
    contrast = 1.0
    enrichments = 2
    batchlogging = False
    batchprocs = 1
    maxkeys = 1
    maxprocs = 1
    patches = None
    patches_string = 'all'
    dofmax = 100
    debug = False
    heat = True
    elasticity = True
    ldomain = False
    holedirichlet = False
    krylov = False
    krylovharmonic = False
    krylovneumann = False
    lagrangeneumannharmonic = True
    normalizenullorthogonalize = False
    outdirsuffix = ''
    orthogonalizehats = True
    interiororthmatrix = False

    try:
        opts, args = getopt.getopt(sys.argv[1:],'hdl',['help','debug','noheat','noelasticity','ldomain','holedirichlet','meshdir=','prefix=',
                                                       'patchlevel=','level=','beta=','contrast=','enrichments=','patches=','batchlogging=','batchprocs=','maxkeys=','dofmax=',
                                                       'krylov=', 'krylovharmonic=', 'krylovneumann=','lagrangeneumannharmonic=','normalizenullorthogonalize=','outdirsuffix=',
                                                       'orthogonalizehats=', 'interiororthmatrix='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h','--help'): 
            usage()
            sys.exit()
        elif opt in ('-d','--debug'):
            debug = True
        elif opt in ('-l','--ldomain'):
            ldomain = True
        elif opt == '--noheat':
            heat = False
        elif opt == '--noelasticity':
            elasticity = False
        elif opt == '--holedirichlet':
            holedirichlet = True
        elif opt == '--krylov':
            krylov = int(arg) > 0
        elif opt == '--krylovharmonic':
            krylovharmonic = int(arg) > 0
        elif opt == '--krylovneumann':
            krylovneumann = int(arg) > 0
        elif opt == '--prefix':
            prefix = arg
            meshdir = '../fem_inside_refined/'+prefix
        elif opt == '--meshdir':
            meshdir = arg
        elif opt == '--patchlevel':
            patchlevel = int(arg)
        elif opt == '--level':
            level = int(arg)
        elif opt == '--beta':
            beta = float(arg)
        elif opt == '--contrast':
            contrast = float(arg)
        elif opt == '--enrichments':
            enrichments = int(arg)
        elif opt == '--patches':
            patches = [int(ii) for ii in arg.split(',')]
            patches_string = arg
        elif opt == '--batchlogging':
            batchlogging=(int(arg) > 0)
        elif opt == '--batchprocs':
            batchprocs = int(arg) if int(arg) > 0 else batchprocs
        elif opt == '--maxkeys':
            maxkeys = int(arg) if int(arg) > 0 else maxkeys
        elif opt == '--maxprocs':
            maxprocs = int(arg) if int(arg) > 0 else maxprocs
        elif opt == '--dofmax':
            dofmax = int(arg) if int(arg) > 0 else dofmax
        elif opt == '--lagrangeneumannharmonic':
            lagrangeneumannharmonic = (int(arg) > 0)
        elif opt == '--normalizenullorthogonalize':
            normalizenullorthogonalize = (int(arg) > 0)
        elif opt == '--outdirsuffix':
            outdirsuffix = arg
        elif opt == '--orthogonalizehats':
            orthogonalizehats = int(arg) > 0
        elif opt == '--interiororthmatrix':
            interiororthmatrix = int(arg) > 0
        else:
            print('unhandled option [{:s}]'.format(opt))
            sys.exit(2)

    print("""nodeadlock enrichments
prefix          {:s}
meshdir         {:s}
patchlevel      {:d}
level           {:d}
beta            {:.2e}
contrast        {:.2e}
enrichments     {:d}
patches         {:s}
batchlogging    {:s}
batchprocs      {:d}
maxkeys         {:d}
dofmax          {:d}
ldomain         {:s}
heat            {:s}
elasticity      {:s}""".format(prefix,meshdir,patchlevel,level,beta,contrast,enrichments,'all' if patches is None else str(patches),str(batchlogging),batchprocs,maxkeys,dofmax,str(ldomain),str(heat),str(elasticity)))

    outdir = prefix+outdirsuffix
    if debug:
        outdir = 'debug_{:s}'.format(outdir)
    with sa_utils.LogWrapper('{:s}_logs/{:s}_patchlevel{:d}_level{:d}_dofmax{:d}_contrast{:.2e}_minimal{:d}_beta{:.2e}_patch{:s}'.format(
        outdir, prefix, patchlevel, level, dofmax, contrast, enrichments, beta, patches_string)
    ) as logger:
        localizations_nitsche.patch_script(meshdir, prefix, level, patchlevel, outdir, write_shapes=True,
                                           logger=logger, max_procs=maxprocs, contrast=contrast, beta=beta,
                                           minimal=enrichments, patches=patches, batch_logging=batchlogging,
                                           batch_procs=batchprocs, max_keys=maxkeys, dofmax=dofmax, debug=debug,
                                           heat=heat, elasticity=elasticity, ldomain=ldomain,
                                           hole_only_neumann = not holedirichlet,
                                           krylov = krylov, krylov_harmonic = krylovharmonic, krylov_neumann = krylovneumann,
                                           lagrange_neumann_harmonic = lagrangeneumannharmonic, normalize_null_orthogonalize = normalizenullorthogonalize,
                                           orthogonalize_hats = orthogonalizehats, interior_orth_matrix = interiororthmatrix)
