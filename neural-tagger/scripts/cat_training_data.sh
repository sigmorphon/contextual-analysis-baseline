#!/bin/bash
#set -e  # exit on error

UD_PATH=$1
JKP_PATH=$2 # location of jackknifing output

for lang in ${UD_PATH}/*/ ;
do
    result=$(find ${lang} -type f -iname "*um-train.conllu")
    prefix=${result%um-train.conllu}
    lang_code=${prefix##*/}
    for i in $(seq 0 9);
    do
        cat ${JKP_PATH}/${lang_code}um-train.conllu${i}.baseline.pred >> ${JKP_PATH}/${lang_code}um-train.conllu.baseline.pred
    done
    wc -l ${JKP_PATH}/${lang_code}um-train.conllu.baseline.pred
    wc -l ${result}
    echo
done 
