#!/bin/bash
while getopts i:l:o: opt; do
    case "$opt" in
        i) input=$OPTARG ;;
        l) lines=$OPTARG ;;
        o) output=$OPTARG ;;
    esac
done

for line in $(echo ${lines} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g' | sed -e 's/,/ /g')
do
    echo "check line ${line}"
    species=(`awk 'NR=="'$line'"{print $7}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)
    cultivar=(`awk 'NR=="'$line'"{print $8}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)
    tissue=(`awk 'NR=="'$line'"{print $10}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)

    OUT_FILE="/cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/err_out/${species}_${cultivar}_${tissue}.out"
    HICPRO_DIR="/cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/${species}_${cultivar}_${tissue}"

    # Check each file
    all_exist=true
    if [ -e ${OUT_FILE} ]
    then
        if cat ${OUT_FILE} | grep -q "Successfully completed" 
        then
            all_exist=true
        else
            all_exist=false
        fi
    else
        all_exist=false
    fi

    if ${all_exist}
    then
        echo -e "${line}\t1" >> ${output}
        rm -rf ${HICPRO_DIR}/${species}_${cultivar}_${tissue}/bowtie_results
    else
        echo -e "${line}\t0" >> ${output}
    fi
done