#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=netgen_cube_few_balls
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_meshes

source /home/wu/fenics.env

echo "$(date): Meshing cube_few_balls START"
srun python3 -u cube_few_balls.py -C
wait
echo "$(date): Meshing cube_few_balls END"
