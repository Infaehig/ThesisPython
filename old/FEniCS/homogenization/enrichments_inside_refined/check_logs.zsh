#!/bin/zsh
clear
#for dir in {debug_square,debug_square_balls,ldomain,square,square_hole,square_few_balls,square_channels,square_channel,square_ring,square_balls,square_grid_balls}; do
#for dir in debug_square; do
for dir in debug_square_{direct_{,no}lagrange_{,no}ortho,krylov{,harmonic,neumann}_{,no}ortho}{,_heat,_elasticity}; do
    if [[ -d ${dir} ]]; then
        errors=$(if [[ -d ${dir}_logs ]]; then
            for ii in $(ls -t ${dir}_logs/stdlog/**/*.log); do
                if [[ -s $ii ]]; then
                    grep -i Error $ii >/dev/null 2>&1
                    if [[ $? -eq 0 ]]; then
                        echo "File: $ii"
                        echo "Stat: $(stat $ii)"
                        grep --color=always -C 10 -i Error $ii
                        echo
                    fi
                fi
            done
        fi)
        (echo "$dir\n"
        if ! [[ -z $errors ]]; then
            echo "Errors:\n$errors\n"
        else
            echo "No errors\n"
        fi
        echo 'Log:'
        for ii in $(ls -t ${dir}_logs/*.log); do
            echo $ii
            tail -n 6 $ii | egrep --color=always 'Finished|$'
            echo
        done
        echo "XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX\n\n")
    fi
done | less -r
