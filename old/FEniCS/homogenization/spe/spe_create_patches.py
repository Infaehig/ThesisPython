import os, sys, getopt
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import spe
import sa_hdf5

if __name__ == '__main__':
    patch_h = 400

    prefix='nowells'
    with sa_utils.LogWrapper(prefix+'/'+prefix) as logger:
        ret = sa_hdf5.read_dolfin_mesh(prefix+'/'+prefix)
        spe.create_patches(ret['mesh'], ret['cell_coeff'], patch_h=patch_h, prefix=prefix, logger=logger)
        logger.info('{:s} finished'.format(prefix))
        del ret
    del prefix

    prefix='wells'
    with sa_utils.LogWrapper(prefix+'/'+prefix) as logger:
        ret = sa_hdf5.read_dolfin_mesh(prefix+'/'+prefix)
        spe.create_patches(ret['mesh'], ret['cell_coeff'], patch_h=patch_h, prefix=prefix, logger=logger)
        logger.info('{:s} finished'.format(prefix))
        del ret
    del prefix
