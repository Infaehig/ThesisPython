#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --job-name=paunt_square_hole_heat
#SBATCH --workdir=/data/wu/work/codes/fenics/homogenization/paunt_new

coarsest_level=2
finest_level=5
mesh_name=square_hole

source ~/paunt.env

mkdir -p ${mesh_name}
pushd ${mesh_name}
echo \
"$(date): ${mesh_name} start"
#for res in $(seq 6 -1 4); do
for res in 4; do
    mkdir -p res_${res}
    pushd res_${res} >/dev/null
    echo \
"    $(date): res ${res} start"
    for bc in 0 5 4; do
        mkdir -p bc_${bc}
        pushd bc_${bc} >/dev/null
        echo \
"        $(date): bc ${bc} start"
        for beta in 2.0; do
            mkdir -p beta_${beta}
            pushd beta_${beta} >/dev/null
            echo \
"            $(date): beta ${beta} start"
            for contrast in 1.00e+00; do 
                mkdir -p contrast_${contrast}/heat
                pushd contrast_${contrast}/heat >/dev/null
                echo \
"                $(date): contrast ${contrast} heat start"
                for patchlvl in 0; do
                    mkdir -p patchlvl_${patchlvl}
                    pushd patchlvl_${patchlvl} >/dev/null
                    echo \
"                    $(date): patchlvl ${patchlvl} start"
                    for cc in liptonE efendiev_extended subspaceE harmonic_polys_neumann_extended; do
                        mkdir -p type_$cc
                        pushd type_$cc >/dev/null
                        echo \
"                        $(date): enrichment type ${cc} start"
                        for ii in $(seq $coarsest_level $finest_level); do
                            mkdir -p coarsest_level_$ii
                            pushd coarsest_level_$ii >/dev/null
                            echo \
"                            $(date): coarse level ${ii} start"
                            for jj in $(seq $ii $ii); do
                                mkdir -p finest_level_$jj
                                pushd finest_level_$jj >/dev/null
                                echo \
"                                $(date): fine level ${jj} start"
                                for ee in $(seq 1 5); do
                                    mkdir -p case_1e-$ee
                                    pushd case_1e-$ee >/dev/null
                                    echo \
"                                    $(date): number of enrichments for 1e-${ee} start"
                                    srun -N1 -n1 --exclusive python3 -u ../../../../../../../../../../../newer_run.py --pumcoarsestlevel=$ii --pumfinestlevel=$jj --accuracy=1e-${ee} --basistype=$cc --meshname=$mesh_name --meshresolution=$res --patchlevel=$patchlvl --contrast=$contrast --beta=$beta --bccase=$bc >stdout.log 2>stderr.log &
                                    popd >/dev/null
                                done
                                wait
                                popd >/dev/null
                                echo \
"                                $(date): fine level ${jj} end"
                            done
                            popd >/dev/null
                            echo \
"                            $(date): coarse level ${ii} end"
                        done
                        popd >/dev/null
                        echo \
"                        $(date): enrichment type ${cc} end"
                    done
                    popd >/dev/null
                    echo \
"                    $(date): patchlvl ${patchlvl} end"
                done
                popd >/dev/null
                echo \
"                $(date): contrast ${contrast} heat end" 
            done
            popd >/dev/null
            echo \
"            $(date): beta ${beta} end"
        done
        popd >/dev/null
        echo \
"        $(date): bc ${bc} end"
    done
    popd >/dev/null
    echo \
"    $(date): res ${res} end"
done
popd >/dev/null
echo \
"$(date): ${mesh_name} end"
