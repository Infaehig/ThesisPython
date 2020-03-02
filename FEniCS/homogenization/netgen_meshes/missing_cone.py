import os, sys, getopt
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import netgen_csg
import fenics_example

if __name__ == '__main__':
    create = False
    calc = True

    opts, args = getopt.getopt(sys.argv[1:],'hcC')
    for opt, arg in opts:
        if opt == '-h': 
            print('-h help')
            print('-c create meshes')
            print('python run.py -h -c')  
            sys.exit()
        elif opt == '-c':
            create = True
        elif opt == '-C':
            create = True
            calc = False

    aa = 1
    max_resolution = 2.**(-4)
    max_refines = 2
    myrange = sa_utils.np.arange(max_refines) 
    prefix = 'missing_cone'
    logger = sa_utils.LogWrapper(prefix+'/'+prefix)

    if create:
        netgen_csg.create_patches(aa=aa, patch_num=3, alpha=1.25, beta=2.0, max_resolution=max_resolution, max_refines=max_refines, num=0, create_inclusions=False, prefix=prefix, logger=logger, lshape='cone')

    if calc:
        contrasts = [1.]
        fenics_example.run_fenics_example(prefix=prefix, myrange=myrange, elasticity=False, mesh_suffices=myrange, logger=logger, contrasts=contrasts)
        fenics_example.run_fenics_example(prefix=prefix, myrange=myrange, elasticity=True, mesh_suffices=myrange, logger=logger, contrasts=contrasts)
