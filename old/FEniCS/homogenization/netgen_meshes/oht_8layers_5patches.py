import os, sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import netgen_csg
import numpy as np

if __name__ == '__main__':
    prefix = 'oht_8layers_5patches'
    logger = sa_utils.LogWrapper(prefix+'/'+prefix)

    netgen_csg.create_patches(box = np.array([0., 0., 0., 6, 1.5, 0.0592]), hole_radius = 0.125, layers = 8, max_resolution = 0.2, max_refines = 6,
                              num = 0, create_inclusions = False, prefix = prefix, logger = logger, alpha = 1.25, beta = 2.0, patch_nums = np.array([5, 1, 1]))
