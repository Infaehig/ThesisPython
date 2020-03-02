#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=spe_mesh_phi
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/spe

source /home/wu/fenics.env

srun python3 -u spe_mesh_phi.py
