#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=netgen_fichera
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_meshes

source /home/wu/fenics.env

echo "$(date): Meshing fichera END"
srun python3 -u fichera.py -C
wait
echo "$(date): Meshing fichera END"
