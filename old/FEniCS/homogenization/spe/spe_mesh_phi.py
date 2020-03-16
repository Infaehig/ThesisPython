import os, sys, getopt
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import spe
import sa_hdf5

if __name__ == '__main__':

    prefix='nowells'
    with sa_utils.LogWrapper(prefix+'/'+prefix) as logger:
        mesh, facets = spe.create_mesh(logger=logger)
        phi = spe.create_phi(mesh,logger=logger)
        sa_hdf5.write_dolfin_mesh(mesh,prefix+'/'+prefix,cell_coeff=phi,facet_function=facets)

    prefix='wells'
    with sa_utils.LogWrapper(prefix+'/'+prefix) as logger:
        mesh, facets = spe.create_mesh(wells=True,logger=logger)
        phi = spe.create_phi(mesh,wells=True,logger=logger)
        sa_hdf5.write_dolfin_mesh(mesh,prefix+'/'+prefix,cell_coeff=phi,facet_function=facets)
