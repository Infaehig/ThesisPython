#!/bin/zsh

for dir in {oht_3layers_5patches_ortho,oht_3layers_5patches_ortho_noz,oht_8layers_5patches_ortho,oht_8layers_5patches_ortho_noz}; do
#for dir in {oht_8layers_5patches_noz,oht_3layers_5patches_noz,fichera,oht_8layers_5patches,oht_3layers_5patches}; do
    if [[ -d ${dir} ]]; then
        clear
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
        done) | less -r
    fi
done
