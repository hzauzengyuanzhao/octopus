import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from preprocess.get_dataset import GenomicDataset
# from your_project.dataset import GenomicDataset

class MultiSpeciesDataset(Dataset):
    """
    Multi-species dataset packaging: internally maintains multiple GenomicDatasets and returns
    (dna, hic, species_id) when __getitem__ is called.
    """
    def __init__(self, config, mode="train"):
        super().__init__()
        self.config = config
        self.mode = mode
        self.datasets = []
        self.entries = []  # (sp_id, idx)

        # Traverse the list of species and load the GenomicDataset for each species
        for sp_id, sp in enumerate(config.species_list):
            dataset = GenomicDataset(
                fasta_path=os.path.join(config.data_path, "genome", sp, "genome.fa"),
                hic_dir=os.path.join(config.data_path, "hic", sp),
                genomic_path=os.path.join(config.data_path, "genomic_features", sp),
                mode=mode,
                windows=config.windows,
                res=config.res,
                output=config.output,
                bw=None,
                genomic_features=False,  # DNA-only
                use_aug=config.use_aug,
                exclude_bed_path=None,
            )
            self.datasets.append(dataset)

            # Save all sample indices and label them with species IDs
            for idx in range(len(dataset)):
                self.entries.append((sp_id, idx))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sp_id, idx = self.entries[index]
        dna, hic = self.datasets[sp_id][idx]
        return dna, hic, sp_id


def collate_fn(batch):
    """
    Customize collate_fn to ensure the batch contains species_id.
    """
    dna_batch = torch.stack([item[0] for item in batch])
    hic_batch = torch.stack([item[1] for item in batch])
    species_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return dna_batch, hic_batch, species_batch


class MultiSpeciesDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = MultiSpeciesDataset(self.config, mode="train")
        self.val_dataset = MultiSpeciesDataset(self.config, mode="valid")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
