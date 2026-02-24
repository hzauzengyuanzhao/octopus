# Raw_Data_Preprocessing_Pipeline

A comprehensive suite of automated scripts for preprocessing ATAC-seq, ChIP-seq, and Hi-C sequencing data.

##  Project Structure

```text
Raw_Data_Preprocessing_Pipeline/
├── bin/
│   ├── atac/
│   │   ├── bam_to_bw_atac.sh      # BAM to BigWig conversion for ATAC
│   │   └── have_bam_atac.sh       # Raw data to BAM pipeline for ATAC
│   ├── chip/
│   │   ├── bam_to_bw_chip.sh      # BAM to BigWig conversion for ChIP
│   │   └── have_bam_chip.sh       # Raw data to BAM pipeline for ChIP
│   └── hic/
│       ├── HiC_Pro.sh             # Main HiC-Pro execution script
│       ├── check_hicpro_result.sh # Output integrity validation
│       └── to_cool.sh             # Matrix to .cool/.mcool conversion
├── atac_example.sh                # Execution templates for ATAC-seq
└── chip_example.sh                # Execution templates for ChIP-seq

```

##  Usage Instructions

### ATAC-seq & ChIP-seq

The workflows for ATAC and ChIP are standardized to automate SRA-to-FASTQ conversion, quality control (fastp), alignment (Bowtie2), and deduplication (sambamba).

* **How to run**: Refer to the example scripts (`atac_example.sh`, `chip_example.sh`) in the root directory for parameter ordering and command-line templates.
* **Key Steps**:
1. Process individual replicates to generate filtered, deduplicated BAMs.
2. Merge biological replicates using `samtools merge`.
3. Generate RPKM-normalized BigWig tracks via `bamCoverage` for visualization.



```bash
# Example execution from the root directory
bash atac_example.sh
bash chip_example.sh

```

### Hi-C

The Hi-C module handles complex alignment and matrix generation requirements for chromosome conformation capture data.

* **HiC_Pro.sh**: The primary entry point for running the HiC-Pro pipeline.
* **check_hicpro_result.sh**: Run this after the pipeline finishes to verify that all chromosome-specific maps and matrices were generated successfully.
* **to_cool.sh**: Converts the high-resolution matrices produced by HiC-Pro into `.cool` or `.mcool` formats for use with visualization tools like HiGlass or Juicebox.

## Technical Requirements

### Environment Modules

The following tools must be loaded in your environment (e.g., via `module load`):

* **Preprocessing**: `sratoolkit`, `fastp`
* **Alignment**: `Bowtie2`, `SAMtools`, `sambamba`
* **Analysis**: `deepTools`, `ucsc_kentUtils`
* **Hi-C**: `HiC-Pro`, `cooler`

### Global Parameters

All scripts require a mandatory `WORK_DIR` argument to maintain a clean and fixed output structure.

* **Global Reference**: The `chrom.sizes` file is automatically generated and stored at `${WORK_DIR}/chrom.sizes`.
* **Centralized Logging**: All pipeline activities, including successes and failures across different samples, are recorded in `${WORK_DIR}/pipeline.log`.
