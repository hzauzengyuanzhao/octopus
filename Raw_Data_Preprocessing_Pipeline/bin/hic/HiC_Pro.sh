#!/bin/bash

###
### Perform HiC-Pro
###
### Usage:
###   bash HiC_Pro.sh -i -l
###
### Options:
###  -i  Input download list (Project Species SRAs Tissue Institute Enzyme; \t as the delimiter).
###  -l  Specific line (eg. 1,2,3).
###  -o  Output folder.
###  -h  Show this message.

help() {
        awk -F'### ' '/^###/ { print $2 }' "$0"
}

if [[ $# == 0 ]] || [[ "$1" == "-h" ]]; then
        help
        exit 1
fi

while getopts i:l:o: opt; do
    case "$opt" in
        i) input=$OPTARG ;;
        l) lines=$OPTARG ;;
        o) output=$OPTARG ;;
    esac
done

declare -A dict
dict["mboi"]="GATCGATC"
dict["dpnii"]="GATCGATC"
dict["bglii"]="AGATCGATCT"
dict["hindiii"]="AAGCTAGCTT"

module purge
module load HiC-Pro/2.11.1

for line in $(echo ${lines} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g' | sed -e 's/,/ /g')
do
    species=(`awk 'NR=="'$line'"{print $7}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)
    cultivar=(`awk 'NR=="'$line'"{print $8}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)
    tissue=(`awk 'NR=="'$line'"{print $10}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)
    enzyme=(`awk 'NR=="'$line'"{print $12}' ${input} | sed -e 's/^[ ]*//g' | sed -e 's/[ ]*$//g'`)

    if [ ! -d  "${output}/${species}_${cultivar}_${tissue}" ]
    then
        mkdir ${output}/${species}_${cultivar}_${tissue}
        mkdir ${output}/${species}_${cultivar}_${tissue}/data
        mkdir ${output}/${species}_${cultivar}_${tissue}/data/${species}_${cultivar}_${tissue}
    else
        rm -rf ${output}/${species}_${cultivar}_${tissue}/*
        mkdir ${output}/${species}_${cultivar}_${tissue}/data
        mkdir ${output}/${species}_${cultivar}_${tissue}/data/${species}_${cultivar}_${tissue}
    fi
    

    ###make link
    source_folder=(`find /cotton/yxlong/All_Plant_3D_Genome/raw_data -type d -path *${species}/${cultivar}/${tissue} -print`)
    link_folder="${output}/${species}_${cultivar}_${tissue}/data/${species}_${cultivar}_${tissue}"
    ln -s ${source_folder}/${species}_${cultivar}_${tissue}_R1.fastq.gz ${link_folder}/${species}_${cultivar}_${tissue}_R1.fastq.gz
    ln -s ${source_folder}/${species}_${cultivar}_${tissue}_R2.fastq.gz ${link_folder}/${species}_${cultivar}_${tissue}_R2.fastq.gz

    ###HiC-Pro configure file
    site=${dict[${enzyme}]}
    sed -e 's/BOWTIE2_IDX_PATH = /BOWTIE2_IDX_PATH = \/cotton\/yxlong\/All_Plant_3D_Genome\/Genome\/'$species'_'$cultivar'/g' \
        -e 's/REFERENCE_GENOME = /REFERENCE_GENOME = '$species'_'$cultivar'/g' \
        -e 's/GENOME_SIZE = /GENOME_SIZE = \/cotton\/yxlong\/All_Plant_3D_Genome\/Genome\/'$species'_'$cultivar'\/'$species'_'$cultivar'.genome.lst/g' \
        -e 's/GENOME_FRAGMENT = /GENOME_FRAGMENT = \/cotton\/yxlong\/All_Plant_3D_Genome\/Genome\/'$species'_'$cultivar'\/'$species'_'$cultivar'_'$enzyme'.txt/g' \
        -e 's/LIGATION_SITE = /LIGATION_SITE = '$site'/g' \
        /cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/config_hicpro.txt \
        > ${output}/${species}_${cultivar}_${tissue}/config_hicpro.txt
    
    ###perform HiC-Pro
    bsub -J ${species}_${cultivar}_${tissue} -q high -n 30 -M 200G -R span[hosts=1] \
        -e /cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/err_out/${species}_${cultivar}_${tissue}.err \
        -o /cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/err_out/${species}_${cultivar}_${tissue}.out "\
        HiC-Pro -i ${output}/${species}_${cultivar}_${tissue}/data \
                -o ${output}/${species}_${cultivar}_${tissue}/${species}_${cultivar}_${tissue} \
                -c ${output}/${species}_${cultivar}_${tissue}/config_hicpro.txt \
                > /cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/err_out/${species}_${cultivar}_${tissue}.out \
                2> /cotton/yxlong/All_Plant_3D_Genome/HiC_Pro/err_out/${species}_${cultivar}_${tissue}.err"        
done
