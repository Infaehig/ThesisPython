#!/bin/zsh

for dir in {debug_square,debug_square_balls,ldomain,square,square_hole,square_few_balls,square_channels,square_ring,square_channel,square_balls,square_grid_balls}; do
    sbatch ${dir}_patches.slurm
done
