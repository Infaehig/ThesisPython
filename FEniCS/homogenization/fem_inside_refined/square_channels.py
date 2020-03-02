import os, sys, getopt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
from fenics_example import *
from create_patches import *

if __name__ == '__main__':
    box = np.array([-1., -1., 1., 1.])
    inc_box = np.array([-1.1, -1.1, 1.1, 1.1]) 
    myrange = np.arange(7, 13, dtype=int)
    prefix = 'square_channels'
    partition_res_mod = -3
    logger = sa_utils.LogWrapper(prefix+'/'+prefix)

    create_patches(box=box, prefix=prefix, num=0, hole=True, 
                   myrange=myrange, betas=[2.0, 5.0], 
                   partition_level=1, partition_x=3, partition_res_mod=partition_res_mod, logger=logger, 
                   patches_only=False, global_feature='channels', channels_params=(0.05, 0.03, 1./3., 0))
