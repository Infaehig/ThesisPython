#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=netgen_missing_cone
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_meshes

source /home/wu/fenics.env

echo "$(date): Meshing missing_cone START"
srun python3 -u missing_cone.py -C
wait
echo "$(date): Meshing missing_cone END"
