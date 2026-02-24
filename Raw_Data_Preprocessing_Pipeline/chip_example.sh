#!/bin/bash
#Before running this pipeline, please ensure that you have prepared two types of sequencing data: Input (background control) and IP
# --- Argument Check ---
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <INPUT_SRA> <IP_REP1_SRA> <IP_REP2_SRA> <REF_FASTA> <PART_NAME> <WORK_DIR>"
    echo "Example: $0 ./data/in.sra ./data/ip1.sra ./data/ip2.sra ./ref/gen.fa MyChIP /home/user/chip_project"
    exit 1
fi

# Arguments
INPUT_SRA=$1    # Control/Input sample
IP_REP1_SRA=$2  # IP replicate 1
IP_REP2_SRA=$3  # IP replicate 2
REF=$4          # Reference Fasta
PART=$5         # Base name for merged results (e.g., H3K27ac)
WORK_DIR=$6     # Project root directory
THREADS=16      # CPU threads

# Sub-scripts location (Update paths if necessary)
HAVE_BAM_SH="./bin/chip/have_bam_chip.sh"
BAM_TO_BW_SH="./bin/chip/bam_to_bw_chip.sh"

# Extract IDs for directory navigation
IN_ID=$(basename "$INPUT_SRA" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')
IP1_ID=$(basename "$IP_REP1_SRA" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')
IP2_ID=$(basename "$IP_REP2_SRA" | sed -E 's/\.(sra|fastq\.gz|fastq|fq\.gz|fq)$//; s/_[12]$//')

# Define internal paths
BAM_IN="${WORK_DIR}/chip_seq/${IN_ID}/${IN_ID}_sorted_rmd.bam"
BAM_IP1="${WORK_DIR}/chip_seq/${IP1_ID}/${IP1_ID}_sorted_rmd.bam"
BAM_IP2="${WORK_DIR}/chip_seq/${IP2_ID}/${IP2_ID}_sorted_rmd.bam"

MERGED_IP_DIR="${WORK_DIR}/chip_seq/${PART}_IP_merged"
MERGED_IP_BAM="${MERGED_IP_DIR}/${PART}_IP_merged.bam"
LOG_FILE="${WORK_DIR}/pipeline.log"

module load SAMtools/1.9
module load deepTools/3.5.0

# --- 1. Process All Samples Individually (Input + 2 IPs) ---
echo "Status: Processing individual samples (Input and 2 IPs)..."
bash "$HAVE_BAM_SH" "$INPUT_SRA" "$REF" "$THREADS" "$WORK_DIR" && \
bash "$HAVE_BAM_SH" "$IP_REP1_SRA" "$REF" "$THREADS" "$WORK_DIR" && \
bash "$HAVE_BAM_SH" "$IP_REP2_SRA" "$REF" "$THREADS" "$WORK_DIR"

# --- 2. Merge IP Biological Replicates ---
if [ ! -f "$MERGED_IP_BAM" ]; then
    if [ -f "$BAM_IP1" ] && [ -f "$BAM_IP2" ]; then
        echo "Status: Merging IP replicates into $MERGED_IP_BAM"
        mkdir -p "$MERGED_IP_DIR"
        samtools merge -@ "$THREADS" "$MERGED_IP_BAM" "$BAM_IP1" "$BAM_IP2"
        samtools index -@ "$THREADS" "$MERGED_IP_BAM"
    else
        echo "Error: IP replicate BAM files not found!"
        exit 1
    fi
fi

# --- 3. Generate Individual BigWigs ---
# Generate BW for Input and Merged IP
echo "Status: Generating BigWigs for Input and Merged IP..."
bash "$BAM_TO_BW_SH" "$BAM_IN" "$WORK_DIR" "${PART}_input" "$THREADS"
bash "$BAM_TO_BW_SH" "$MERGED_IP_BAM" "$WORK_DIR" "${PART}_ip" "$THREADS"

# --- 4. Calculate Ratio (IP / Input) ---
BW_INPUT="${WORK_DIR}/bw/${PART}_input.bw"
BW_IP="${WORK_DIR}/bw/${PART}_ip.bw"
BW_FINAL="${WORK_DIR}/bw/${PART}_ratio.bw"

if [ ! -f "$BW_FINAL" ]; then
    echo "Status: Calculating IP/Input ratio using bigwigCompare..."
    bigwigCompare \
        -b1 "$BW_IP" \
        -b2 "$BW_INPUT" \
        --operation ratio \
        --binSize 1 \
        -p "$THREADS" \
        -o "$BW_FINAL"
    
    echo "$(date "+%Y-%m-%d %H:%M:%S") | ${PART} | Pipeline Complete" >> "$LOG_FILE"
fi

echo "------------------------------------------------"
echo "ChIP-seq Complete for ${PART}"
echo "Final Ratio BigWig: ${BW_FINAL}"
echo "------------------------------------------------"