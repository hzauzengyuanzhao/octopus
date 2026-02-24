import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import argparse
###############################################
from model.Octopus import Octopus
from model.MappingModel import MappingModel
from utils.mapping_train_process import GenomicModel, GenomicDataModule

from utils.get_model import get_model
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '600'

class Config:
    def __init__(self, args):
        self.world_size = torch.cuda.device_count()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        self.species = args.species
        self.model_class = MappingModel

        self.output_path = f"/root/autodl-tmp/output_mapping"
        self.data_path = f"/root/autodl-tmp/data"

        self.use_aug = args.use_aug

        self.label_model = None

        if self.species == 'Zea':
            self.bwfile = {'atac_b73.bw':'log'}
            self.valid_chroms = ['Zm_B73_Chr3']
            self.test_chroms = ['Zm_B73_Chr7']
            self.fasta_path = self.data_path + f'/genome/{self.species}/Zm-B73-REFERENCE-GRAMENE-5.0.fa'
        elif self.species == 'rice':
            self.bwfile = {'atac_mh63.bw': 'log'}
            self.valid_chroms = ['Chr09']
            self.test_chroms = ['Chr11']
            self.fasta_path = self.data_path + f'/genome/{self.species}/MH63RS3.fa'
        else:
            self.bwfile = {'atac_gaoliang.bw': 'log'}
            self.valid_chroms = ['Sb_BTx623_Chr5']
            self.test_chroms = ['Sb_BTx623_Chr9']
            self.fasta_path = self.data_path + f'/genome/{self.species}/Sorghum-bicolor_BTx623.fa'

        # if self.species == 'cotton':
        #
        #     self.bwfile = {'sub_merged.bin1.rpkm.bw': 'log'}
        #
        #     self.valid_chroms = ['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']
        #     self.test_chroms = ['HC04_A05', 'HC04_D05']
        #     self.fasta_path = self.data_path + f'/genome/{self.species}/genome.fa'
        # elif self.species == 'tomato':
        #     self.bwfile = {'atac_M82.bw': 'log'}
        #     self.valid_chroms = ['chr10']
        #     self.test_chroms = ['chr12']
        #     self.fasta_path = self.data_path + f'/genome/{self.species}/M82_v1.fa'
        # elif self.species == 'bean':
        #     self.bwfile = {'atac_bean.bw': 'log'}
        #     self.valid_chroms = ['Chr11', 'Chr12']
        #     self.test_chroms = ['Chr04', 'Chr05']
        #     self.fasta_path = self.data_path + f'/genome/{self.species}/Wm82_new.fa'
        # else:
        #     self.bwfile = {'2end_bowtie2_rep.bw':'log'}
        #     self.valid_chroms = ['Chr4']
        #     self.test_chroms = ['Chr3']
        #     self.fasta_path = self.data_path + f'/genome/{self.species}/TAIR10.fa'


        self.windows = args.windows
        self.res = args.res
        self.output = args.output

        # mapping model no epi
        self.epi = 0

        # path
        self.genomic_path = self.data_path + f'/genomic_features/{self.species}/'
        self.hic_dir = self.data_path + f"/hic/{self.species}/"
        self.exclude_bed_path = None

        # train
        self.num_workers = args.num_workers
        self.warmup_epochs = 10
        self.batch_size = args.batch_size
        self.base_learning_rate = args.learning_rate
        self.learning_rate = self.base_learning_rate * np.sqrt(self.world_size)
        self.weight_decay = 1e-5
        self.epochs = args.epochs
        self.patience = args.patience
        # atac for teacher model
        self.genomic_features = True

        # save path
        self.model_name = self.model_class.__name__
        self.model_dir = self.output_path + f"/saved_models/{self.species}/{self.model_name}_{self.genomic_features}/"
        self.best_model_path = os.path.join(self.model_dir, "best_model.pth") if self.local_rank == 0 else None

        self.log_dir = self.output_path + f"/logs/{self.species}/{self.model_name}_{self.genomic_features}/"
        self.results_file = os.path.join(self.log_dir, "training_results.txt") if self.local_rank == 0 else None
        self.plot_file = os.path.join(self.log_dir, "training_plot.png") if self.local_rank == 0 else None
        self.plot_dis_path = os.path.join(self.log_dir, "val_dis_plot.png") if self.local_rank == 0 else None

def main():
    parser = argparse.ArgumentParser(description="Train genomic model with Lightning")
    parser.add_argument("--species", type=str, default="cotton", help="Species name")
    parser.add_argument("--output_path", type=str, default="/root/autodl-tmp/output_mapping", help="Output path")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data", help="Data path")
    parser.add_argument("--use_aug", type=bool, default=True, help="Use data augmentation")
    parser.add_argument("--windows", type=int, default=2097152, help="Window size")
    parser.add_argument("--res", type=int, default=10000, help="Resolution")
    parser.add_argument("--output", type=int, default=256, help="Model output dimension")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Base learning rate (single GPU)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()


    config = Config(args)

    label_model = Octopus(1)

    config.pretrained_model_path=None

    label_model_path = f'/root/autodl-tmp/output_atac/saved_models/{config.species}/{label_model.__name__}_True/best_model.pth'
    # Load weights
    label_model = get_model(label_model, label_model_path)

    label_model.eval()
    # label_model.to(config.device)
    config.label_model = label_model


    pl.seed_everything(42, workers=True)


    model = GenomicModel(config)
    data_module = GenomicDataModule(config)

    # Create a callback function
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Configure distributed strategy
    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    # Create Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.world_size,
        strategy=strategy,
        max_epochs=config.epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        default_root_dir=config.output_path,
        enable_progress_bar=config.local_rank == 0,
        log_every_n_steps=10,
        precision="16-mixed",
        deterministic=True,
    )

    # Start Training
    trainer.fit(model, datamodule=data_module)

    print(f"Best validation loss: {model.best_val_loss:.8f}")
    print(f"Best validation Mse: {model.best_val_mse:.8f}")
    print(f"Best validation Insu correlation: {model.best_val_corr:.4f}")
    print(f"Best validation Pearson correlation: {model.best_val_pear:.4f}")
    print(f"Best validation Observed vs expected: {model.best_val_os:.4f}")


if __name__ == "__main__":
    main()
