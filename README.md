![octopus](docs/octopus_logo.png)

# Octopus
A multimodal deep learning framework for cross-species prediction of plant 3D chromatin architecture.

Octopus is a deep learning framework for predicting 3D chromatin architecture across plant species. It integrates genomic sequence and epigenomic signals to model chromatin folding, and further supports DNA-only prediction in species lacking epigenomic data through a mapping-based knowledge distillation strategy.

## Overview

Three-dimensional genome organization is a key layer of transcriptional regulation, yet plant 3D genomics remains underexplored because matched Hi-C and epigenomic datasets are available for only a limited number of species. This limits the development of predictive models tailored to plant genome architecture.

Octopus addresses this challenge with a unified deep learning framework for chromatin contact prediction in both data-rich and data-limited settings. In species with multimodal data, Octopus integrates DNA sequence and chromatin accessibility signals to predict high-resolution contact maps. In species with sequence data only, Octopus uses a DNA-to-multimodal mapping strategy to reconstruct informative latent representations, enabling cross-species transfer of 3D genome prediction models.


## Installation
Octopus has been tested in the following environment: 

### Tested environment
- Ubuntu 22.04.1 LTS
- GLIBC 2.35
- NVIDIA GeForce RTX 4090
- NVIDIA Driver 580.76.05
- CUDA 13.0


### 1. Install Miniforge / mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Miniforge3-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

### 2. Clone the repository and create the environment

```bash
git clone https://github.com/zzy-hzau/octopus.git
cd octopus
mamba env create -f environment.yml
mamba activate 3D_chromatin
```

## Data Preparation

To quickly get started, please download the example dataset [Example_data.tar.gz]() and place it under the `data/` directory.
For raw data preprocessing, please refer to the [Raw Data Preprocessing Pipeline](https://github.com/zzy-hzau/octopus/tree/master/Raw_Data_Preprocessing_Pipeline).

## Training

If you want to train your own Octopus, please refer to the configuration class below for setup.
## Class Configuration Documentation
#### Distributed training setup

| name         | value                                                                        | Description                       |
|--------------|------------------------------------------------------------------------------|-----------------------------------|
| `world_size` | `torch.cuda.device_count()`                                                  | Number of available GPUs          |
| `local_rank` | `int(os.environ.get("LOCAL_RANK", 0))`                                       | Local GPU rank of current process |
| `device`     | `torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')` | Training device                   |

#### Data and Preprocessing Configuration

| name         | value                                                                        | Description                       |
|--------|------|-----------------------------------|
| `use_aug` | `True` |  Whether to use data augmentation |
| `bwfile` | `{'sub_merged.bin1.rpkm.bw': 'log','sub_merged.bin1.rpkm_cuttag.bw': 'log'}` | Epigenomic data files             |
| `species` | `'cotton'` | Experimental species              |
| `windows` | `2097152` | Window size (~2Mb)                |
| `res` | `10000` | Resolution (10kb)                 |
| `output` | `256` | Output feature dimension          |
| `epi` | `len(bwfile)` | Number of epigenomic features     |
| `genomic_features` | `True if epi > 0 else False` | Whether to use genomic features   |

#### Path Configuration

| name               | Path value                                   | Description                   |
|--------------------|----------------------------------------------|-------------------------------|
| `output_path`      | `"output"`                                   | Root directory for outputs    |
| `data_path`        | `"data"`                                     | Root directory for input data |
| `fasta_path`       | `data_path + '/genome/{species}/genome.fa'`  | Reference genome FASTA file   |
| `genomic_path`     | `data_path + '/genomic_features/{species}/'` | Genomic features directory    |
| `hic_dir`          | `data_path + "/hic/{species}/"`              | Store the Hi-C files for each chromosome converted from .cool to .npz|
| `exclude_bed_path` | `None or XXX.bed`                            | BED file for excluded regions |

#### Chromosome mapping

| name           | value                                              | Description |
|----------------|----------------------------------------------------|-------------|
| `valid_chroms` | `['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']` | Validation chromosomes |
| `test_chroms`  | `['HC04_A05', 'HC04_D05']`                         | Test chromosomes |

#### Model Configuration

| name          | value                  | Description |
|---------------|------------------------|-----------------|
| `model_class` | `Octopus`              | Model class |
| `model_name`  | `model_class.__name__` | Model name |

#### Hyperparameter

| name                 | value                                | Description |
|----------------------|--------------------------------------|-------------|
| `num_workers`        | `4`                                  | Data loader workers |
| `warmup_epochs`      | `10`                                 | Learning rate warmup epochs |
| `batch_size`         | `4`                                  | Batch size per GPU |
| `base_learning_rate` | `2e-4`                               | Base learning rate per GPU |
| `learning_rate`      | `base_learning_rate * √(world_size)` | Adjusted learning rate |
| `weight_decay`       | `1e-5`                               | Weight decay |
| `epochs`             | `200`                                | Total training epochs |
| `patience`           | `20`                                 | Early stopping patience |

#### Save and Log Configuration

| name              | path                                                                         | Description |
|-------------------|------------------------------------------------------------------------------|-------------|
| `model_dir`       | `output_path + "/saved_models/{species}/{model_name}_{genomic_features}/"`   | Model save directory |
| `best_model_path` | `os.path.join(model_dir, "best_model.pth") if local_rank == 0 else None`     | Best model path |
| `log_dir`         | `output_path + "/logs/{species}/{model_name}_{genomic_features}/"`           | Log directory |
| `results_file`    | `os.path.join(log_dir, "training_results.txt") if local_rank == 0 else None` | Training results file |
| `plot_file`       | `os.path.join(log_dir, "training_plot.png") if local_rank == 0 else None`    | Training Loss Record Chart |
| `plot_dis_path`   | `os.path.join(log_dir, "val_dis_plot.png") if local_rank == 0 else None`     | Layered Distance Correlation Plot |

After configuring these parameters and files:
```bash
python train.py
```
## Evaluation

To evaluate a trained model on the test set, run:

```bash
python test.py
```

## Virtual Deletion
Octopus supports in silico virtual deletion analysis for studying the sequence basis of chromatin folding. To run this workflow, please prepare:

- genome sequence file (required)
- epigenomic data (optional)
- Hi-C data for comparison (optional)
- trained model weights (required)

Then run:

```bash
python virtual_deletion.py
```
For genomes with excluded regions, use the following logic to filter masked intervals:

```python
exclude_bed_path = data_path + f"/genome/hg38.bed"
exclude_regions = GenomicDataset._load_exclude_regions_static(exclude_bed_path)

if GenomicDataset._is_position_excluded_static(chrom, seq_start, seq_start + windows, exclude_regions):
                    print(f'{seq_start} had exclude!')
                    continue
```
## Whole-genome Prediction

To generate chromosome-scale predictions, run:
```bash
python whole_chrom_prediction.py --species cotton
```
Here is the [Hi-C](https://zenodo.org/records/18740232) predicted by our fine-tuning model.

## Web Server

We provide a [web server](http://cotton-bigdata.hzau.edu.cn/octopus) for browsing and visualizing predicted Hi-C maps across multiple species.

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub.

