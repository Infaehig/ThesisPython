#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=netgen_oht_8layers_5patches
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/netgen_meshes

source /home/wu/fenics.env

echo "$(date): Meshing oht_8layers_5patches END"
srun python3 -u oht_8layers_5patches.py
wait
echo "$(date): Meshing oht_8layers_5patches END"
