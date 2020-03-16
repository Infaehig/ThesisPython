import os, sys, getopt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import fenics_example
import create_patches
import numpy

def usage():
    print('''Usage
python3 fem_split.py
Option          Default                 Info
-h --help                               help
-e --elasticity                         elasticity, otherwise heat
-l --domain                             ldomain with boundaries 7-9
--prefix=       square                  sets prefix
--rangelow=     5                       lower bound for resolution range
--rangehigh=    12                      upper bound for resolution range
--bccase        4                       which boundary condition case
--contrast      1                       contrast''')

if __name__ == '__main__':
    prefix = 'ldomain'
    rangelow = 4
    rangehigh = 8
    bccase = 4
    elasticity = False
    ldomain = False
    contrast = 1.

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hel', ['help', 'elasticity=', 'ldomain=', 'prefix=', 'rangelow=', 'rangehigh=', 'bccase=', 'contrast='])
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
        elif opt in ('-l', '--ldomain'):
            ldomain = True
        elif opt == '--prefix':
            prefix = arg
        elif opt == '--rangelow':
            rangelow = int(arg) if int(arg) -1 else rangelow
        elif opt == '--rangehigh':
            rangehigh = int(arg) if int(arg) -1 else rangehigh
        elif opt == '--bccase':
            bccase = int(arg) if int(arg) > -1 else bccase
        elif opt == '--contrast':
            contrast = float(arg) if float(arg) > 0 else contrast
    print("""fem split script
elasticity      {:s}
ldomain          {:s}
prefix          {:s}
rangelow        {:d}
rangehigh       {:d}
bccase          {:d}
contrast        {:.2e}""".format(str(elasticity), str(ldomain), prefix, rangelow, rangehigh, bccase, contrast))
    ret = 0
    myrange = numpy.arange(rangelow, rangehigh, dtype=int)
    meshes, domains, facets = create_patches.load_meshes_h5py(prefix, prefix, myrange)
    with sa_utils.LogWrapper('{:s}/logs/{:s}_range{:d}-{:d}_{:s}_bc{:d}'.format(prefix, prefix, rangelow, rangehigh, 'elasticity' if elasticity else 'heat', bccase)) as logger:
        ret = fenics_example.run_fenics_example(prefix=prefix, myrange=myrange, meshes=meshes, domains=domains, facets=facets, elasticity=elasticity, mesh_suffices=myrange, logger=logger, contrasts=[contrast], ldomain=ldomain, bc_case=bccase)
    sys.exit(ret)

