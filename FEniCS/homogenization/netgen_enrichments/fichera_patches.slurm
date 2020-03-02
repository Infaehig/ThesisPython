#!/bin/bash
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=fichera_patches
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_enrichments

source /home/wu/fenics.env

prefix='fichera'
meshdir='../netgen_meshes/fichera'

echo "$(date): ${prefix} START"
for patchlevel in 0; do
    echo "$(date):     patchlevel ${patchlevel} START"
    for level in {3..4}; do
        echo "$(date):         level ${level} START"
        for beta in 2.0; do
            echo "$(date):             beta ${beta} START"
            for contrast in 1.0; do
                echo "$(date):                 contrast ${contrast} START"
                for patch in 13 14 16 17 22 23 25 {0..12} 15 {18..21} 24; do
                    echo "$(date):                     patch ${patch} START"
                    dir=${prefix}_logs/patchlevel_${patchlevel}/level_${level}/beta_${beta}/contrast_$contrast
                    mkdir -p ${dir}
                    srun -N1 -n1 --exclusive python3 -u nodeadlock.py --prefix=${prefix} --meshdir=${meshdir} --patchlevel=${patchlevel} --level=${level} --beta=${beta} --contrast=${contrast} --patches=${patch} --dofmax=200 --batchprocs=1 >${dir}/patch_${patch}.log 2>&1 &
                done
                wait
                echo "$(date):                 contrast ${contrast} END"
            done
            echo "$(date):             beta ${beta} END"
        done
        echo "$(date):         level ${level} END"
    done
    echo "$(date):     patchlevel ${patchlevel} END"
done
wait
echo "$(date): ${prefix} END"
