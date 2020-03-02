#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=fem_square_few_balls_split
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/fem_inside_refined

source /home/wu/fenics.env

prefix='square_few_balls'
rangelow=5
rangehigh=13

mkdir -p ${prefix}_logs

echo "$(date): ${prefix} START"
for contrast in 1.0; do # 1e4 0.0 2.0 0.5 1e-4; do
    for count in {0..5}; do
        echo "$(date): heat, contrast ${contrast}, bc case ${count} start"
        srun -N1 -n1 --exclusive python3 -u fem_split.py --prefix=${prefix} --rangelow=${rangelow} --rangehigh=${rangehigh} --bccase=${count} --contrast=${contrast} >${prefix}_logs/${prefix}_range${rangelow}-${rangehigh}_heat_bc${count}_contrast${contrast}.log 2>&1
        res=$?
        echo "$(date): heat, contrast ${contrast}, bc case ${count} end"
    done
    for count in {0..5}; do
        echo "$(date): elasticity, contrast ${contrast}, bc case ${count} start"
        srun -N1 -n1 --exclusive python3 -u fem_split.py --prefix=${prefix} --rangelow=${rangelow} --rangehigh=${rangehigh} --bccase=${count} --contrast=${contrast} -e >${prefix}_logs/${prefix}_range${rangelow}-${rangehigh}_elasticity_bc${count}_contrast${contrast}.log 2>&1
        res=$?
        echo "$(date): elasticity, contrast ${contrast}, bc case ${count} end"
    done
done
echo "$(date): ${prefix} END"
