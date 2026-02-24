# Section 1 : Setup Environment
You can follow the instructions to setup the environment
Our test env Info bellow
```bash
System Info
-----------------------------------------
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.1 LTS
Release:        22.04
Codename:       jammy
ldd --version
ldd (Ubuntu GLIBC 2.35-0ubuntu3.1) 2.35
-----------------------------------------
GPU NVIDIA GeForce RTX 4090
NVIDIA-SMI 580.76.05
Driver Version: 580.76.05
CUDA Version: 13.0
```

First, install mamba for dependency management (as a fast alternative to Anaconda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Miniforge3-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```
Install dependencies
```bash
git clone https://github.com/zzy-hzau/octopus.git
cd octopus
mamba env create -f environment.yml
conda activate 3D_chromatin
```
# Section 2 : Dataset



# Section 3 : Train Octopus
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
| `batch_size`         | `3`                                  | Batch size per GPU |
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
# Section 4 : Test Octopus

```bash
python test.py
```

# Section 5 : Virtual Deletion
If you need to perform a virtual deletion, please prepare the corresponding genome file (required), epigenome (optional), 
Hi-C data (optional), and the corresponding model weights (required). Usage:
```bash
python virtual_deletion.py
```
For genomes with excluded regions, these codes can be freed
```python
exclude_bed_path = data_path + f"/genome/hg38.bed"
exclude_regions = GenomicDataset._load_exclude_regions_static(exclude_bed_path)

if GenomicDataset._is_position_excluded_static(chrom, seq_start, seq_start + windows, exclude_regions):
                    print(f'{seq_start} had exclude!')
                    continue
```
# Section 6 : Whole genome prediction