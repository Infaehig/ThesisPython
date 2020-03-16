#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=fem_fichera_split
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_meshes

source /home/wu/fenics.env

prefix='fichera'
rangelow=0
rangehigh=5

mkdir -p ${prefix}_logs

echo "$(date): ${prefix} START"
for contrast in 1; do
    for count in {0..5}; do 
        echo "$(date): heat, contrast ${contrast}, bc case ${count} start"
        srun -N1 -n1 --exclusive python3 -u fem_split.py --prefix=${prefix} --rangelow=${rangelow} --rangehigh=${rangehigh} --bccase=${count} --contrast=${contrast} -l >${prefix}_logs/${prefix}_range${rangelow}-${rangehigh}_heat_bc${count}_contrast${contrast}.log 2>&1 &
        res=$?
        echo "$(date): heat, contrast ${contrast}, bc case ${count} end"
    done
    wait
    for count in {0..5}; do
        echo "$(date): elasticity, contrast ${contrast}, bc case ${count} start"
        srun -N1 -n1 --exclusive python3 -u fem_split.py --prefix=${prefix} --rangelow=${rangelow} --rangehigh=${rangehigh} --bccase=${count} --contrast=${contrast} -l -e >${prefix}_logs/${prefix}_range${rangelow}-${rangehigh}_elasticity_bc${count}_contrast${contrast}.log 2>&1 &
        res=$?
        echo "$(date): elasticity, contrast ${contrast}, bc case ${count} end"
    done
    wait
done
wait
echo "$(date): ${prefix} END"
