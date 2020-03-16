#!/bin/bash
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=square_x_patches
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/enrichments_inside_refined

source /home/wu/fenics.env

prefix='square_x'
echo "$(date): ${prefix} START"

for patchlevel in 0; do
    echo "$(date):     patchlevel ${patchlevel} START"
    for level in 7; do
        echo "$(date):         level ${level} START"
        for beta in 5.0; do
            echo "$(date):             beta ${beta} START"
            for contrast in 1e4 1e-4; do
                echo "$(date):                 contrast ${contrast} START"
                for patch in 4 {0..3} {5..8}; do
                    echo "$(date):                     patch ${patch} START"
                    dir=${prefix}_logs/stdlog/patchlevel${patchlevel}_level${level}_beta${beta}_contrast${contrast}
                    mkdir -p ${dir}
                    srun -N1 -n1 --exclusive python3 -u nodeadlock.py --prefix=${prefix} --patchlevel=${patchlevel} --level=${level} --beta=${beta} --contrast=${contrast} --patches=${patch} --enrichments=3 >${dir}/patch_${patch}.log 2>&1 &
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
