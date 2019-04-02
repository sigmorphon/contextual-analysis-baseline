#!/bin/bash
#set -e  # exit on error

UD_PATH=$1

for lang in ${UD_PATH}/*/ ;
do
    echo ${lang}
    #lower_lang=$(echo ${lang} | tr '[:upper:]' '[:lower:]')
    if [ ! -f ${lang}/*um-train.conllu ]; then
        echo "No training set found!"
    else
        python -u scripts/gen_scripts_baseline_jackknife.py ${lang}
    fi
    echo
done >> "logs/task2_training_output.txt" 
