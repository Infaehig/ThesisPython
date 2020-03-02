import os, sys, getopt
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sa_utils
import localizations_nitsche

if __name__ == '__main__':
    meshdir = '../spe/10x2'
    prefix = '10x2'
    patch_level = 0
    level = 0

    outdir = prefix+'_'+str(level)
    with sa_utils.LogWrapper(outdir+'/'+outdir) as logger:
        localizations_nitsche.patch_script(meshdir, prefix, level, patch_level, outdir, write_shapes=True, logger=logger, max_procs=1, minimal=2, spe=True, dofmax=200, skip=[], elasticity=False)
