import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

###############################################
from model.Octopus import Octopus
from utils.train_process import GenomicModel, GenomicDataModule

os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '600'

class Config:
    # Distributed training setup
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Whether to use data augmentation
    use_aug = True
    # Number of multimodal types
    bwfile = {'sub_merged.bin1.rpkm.bw': 'log', 'sub_merged.bin1.rpkm_cuttag.bw': 'log'}
    species = 'cotton'


    output_path = f"output"
    data_path = f"data"
    # model name
    model_class = Octopus

    windows = 2097152
    res = 10000
    output = 256

    epi = len(bwfile)
    # Cotton Verification and Testing Chromosome Mapping
    valid_chroms = ['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']
    test_chroms = ['HC04_A05', 'HC04_D05']

    fasta_path = data_path + f'/genome/{species}/genome.fa'
    genomic_path = data_path + f'/genomic_features/{species}/'
    hic_dir = data_path + f"/hic/{species}/"
    exclude_bed_path = None
    # Training parameters
    num_workers = 4
    warmup_epochs = 10
    batch_size = 4
    # Single card learning rate
    base_learning_rate = 2e-4
    # Adjusted learning rate
    learning_rate = base_learning_rate * np.sqrt(world_size)
    weight_decay = 1e-5
    epochs = 200
    # Early stopping patience
    patience = 20
    genomic_features = True if epi > 0 else False

    # Model save path (created in the main process)
    model_name = model_class.__name__
    model_dir = output_path + f"/saved_models/{species}/{model_name}_{genomic_features}/"
    best_model_path = os.path.join(model_dir, "best_model.pth") if local_rank == 0 else None

    # Logs and results saving (created on the main process)
    log_dir = output_path + f"/logs/{species}/{model_name}_{genomic_features}/"
    results_file = os.path.join(log_dir, "training_results.txt") if local_rank == 0 else None
    plot_file = os.path.join(log_dir, "training_plot.png") if local_rank == 0 else None
    plot_dis_path = os.path.join(log_dir, "val_dis_plot.png") if local_rank == 0 else None

config = Config()


def main():
    # Set random seed
    pl.seed_everything(42, workers=True)

    # Create model and data modules
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
    # start training
    trainer.fit(model, datamodule=data_module)

    print(f"Best validation loss: {model.best_val_loss:.8f}")
    print(f"Best validation Mse: {model.best_val_mse:.8f}")
    print(f"Best validation Insu correlation: {model.best_val_corr:.4f}")
    print(f"Best validation Pearson correlation: {model.best_val_pear:.4f}")
    print(f"Best validation Observed vs expected: {model.best_val_os:.4f}")

if __name__ == "__main__":
    main()
