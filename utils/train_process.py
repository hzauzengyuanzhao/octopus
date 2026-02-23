import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

###############################################
from preprocess.get_dataset import GenomicDataset, collate_fn
from metrics.metrics import insulation_pearson, mse, pearson_correlation, observed_vs_expected, \
    distance_stratified_correlation


class GenomicModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = config.model_class(config.epi)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

        # save training history
        self.train_losses = []
        self.train_insu_corrs = []
        self.train_mses = []
        self.train_pears = []
        self.train_oes = []
        self.val_losses = []
        self.val_insu_corrs = []
        self.val_mses = []
        self.val_pears = []
        self.val_oes = []

        self.best_val_loss = float('inf')
        self.best_val_corr = 0.0
        self.best_val_mse = 0.0
        self.best_val_pear = 0.0
        self.best_val_os = 0.0
        self.best_val_dises = []

        # Used to store the output of verification steps
        self.validation_step_outputs = []

        if self.global_rank == 0:
            os.makedirs(self.config.model_dir, exist_ok=True)
            os.makedirs(self.config.log_dir, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        dna, hic_target = batch
        outputs = self(dna)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = self.criterion(outputs, hic_target)

        #  Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        # Calculate and log other metrics
        with torch.no_grad():
            insu_corr = insulation_pearson(outputs.cpu().numpy(), hic_target.cpu().numpy())
            mse_val = mse(outputs.cpu().numpy(), hic_target.cpu().numpy())
            pear_corr = pearson_correlation(outputs.cpu().numpy(), hic_target.cpu().numpy())
            oe_val = observed_vs_expected(outputs.cpu().numpy(), hic_target.cpu().numpy())

            # Log average
            self.log("train_insu_corr", np.nanmean(insu_corr), on_epoch=True, prog_bar=True, logger=True,
                     sync_dist=False)
            self.log("train_mse", np.nanmean(mse_val), on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            self.log("train_pearson", np.nanmean(pear_corr), on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
            self.log("train_oe", np.nanmean(oe_val), on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        dna, hic_target = batch
        outputs = self(dna)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        loss = self.criterion(outputs, hic_target)

        # Log verification loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Save the output for computation at the end of the epoch
        self.validation_step_outputs.append({
            "loss": loss,
            "outputs": outputs.detach().cpu(),
            "targets": hic_target.detach().cpu()
        })

        return loss

    def on_validation_epoch_end(self):
        # Aggregate the outputs of all verification steps
        all_outputs = []
        all_targets = []
        val_loss = 0.0
        total_samples = 0

        for out in self.validation_step_outputs:
            all_outputs.append(out["outputs"])
            all_targets.append(out["targets"])
            val_loss += out["loss"].item() * out["outputs"].shape[0]
            total_samples += out["outputs"].shape[0]

        if total_samples == 0:
            return

        all_outputs = torch.cat(all_outputs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        avg_val_loss = val_loss / total_samples

        # Compute evaluation metrics
        insu_corr = np.nanmean(insulation_pearson(all_outputs, all_targets))
        mse_val = np.nanmean(mse(all_outputs, all_targets))
        pear_corr = np.nanmean(pearson_correlation(all_outputs, all_targets))
        oe_val = np.nanmean(observed_vs_expected(all_outputs, all_targets))
        dis_corr = np.nanmean(distance_stratified_correlation(all_outputs, all_targets), axis=0)


        self.log("val_insu_corr", insu_corr, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mse", mse_val, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_pearson", pear_corr, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_oe", oe_val, prog_bar=True, logger=True, sync_dist=True)

        # Update Training History
        self.val_losses.append(avg_val_loss)
        self.val_insu_corrs.append(insu_corr)
        self.val_mses.append(mse_val)
        self.val_pears.append(pear_corr)
        self.val_oes.append(oe_val)

        # Check if it is the best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_val_corr = insu_corr
            self.best_val_mse = mse_val
            self.best_val_pear = pear_corr
            self.best_val_os = oe_val
            self.best_val_dises = dis_corr

            # Save the best model
            if self.global_rank == 0:
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_insu_corr': insu_corr,
                    'val_mse': mse_val,
                    'val_pearson': pear_corr,
                    'val_oe': oe_val,
                }, self.config.best_model_path)

        # Clear verification step output
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        lr_scheduler_warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / self.config.warmup_epochs) if self.config.warmup_epochs > 0 else 1.0
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler_warmup,
                "interval": "epoch"
            }
        }

    def on_train_epoch_end(self):
        # Update Training History
        self.train_losses.append(self.trainer.callback_metrics["train_loss_epoch"].item())
        self.train_insu_corrs.append(self.trainer.callback_metrics["train_insu_corr_epoch"].item())
        self.train_mses.append(self.trainer.callback_metrics["train_mse_epoch"].item())
        self.train_pears.append(self.trainer.callback_metrics["train_pearson_epoch"].item())
        self.train_oes.append(self.trainer.callback_metrics["train_oe_epoch"].item())

        # Save training history to file
        if self.global_rank == 0:
            with open(self.config.results_file, 'a') as f:
                f.write(
                    f"{self.current_epoch + 1}\t{self.train_losses[-1]:.8f}\t"
                    f"{self.val_losses[-1] if len(self.val_losses) > 0 else 0:.8f}\t"
                    f"{self.train_insu_corrs[-1]:.4f}\t"
                    f"{self.val_insu_corrs[-1] if len(self.val_insu_corrs) > 0 else 0:.4f}\t"
                    f"{self.train_mses[-1]:.8f}\t"
                    f"{self.val_mses[-1] if len(self.val_mses) > 0 else 0:.8f}\t"
                    f"{self.train_pears[-1]:.4f}\t"
                    f"{self.val_pears[-1] if len(self.val_pears) > 0 else 0:.4f}\t"
                    f"{self.train_oes[-1]:.4f}\t"
                    f"{self.val_oes[-1] if len(self.val_oes) > 0 else 0:.4f}\n"
                )

    def on_train_end(self):
        if self.global_rank == 0:

            plt.figure(figsize=(12, 8))

            # # Loss curve
            plt.subplot(2, 1, 1)
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            #  Pearson curve
            plt.subplot(2, 1, 2)
            plt.plot(self.train_insu_corrs, label='Train Pearson')
            plt.plot(self.val_insu_corrs, label='Validation Pearson')
            plt.xlabel('Epochs')
            plt.ylabel('Pearson Correlation')
            plt.title('Training and Validation Pearson Correlation')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(self.config.plot_file)
            plt.close()

            # Distance Correlation curve
            plt.figure(figsize=(12, 8))
            plt.plot(self.best_val_dises, marker='o', linestyle='-', color='b')
            plt.title('best_val_dises')
            plt.xlabel('The position from the diagonal')
            plt.ylabel('Pearson Correlation')
            plt.grid(True)
            plt.savefig(self.config.plot_dis_path)



class GenomicDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = GenomicDataset(
            fasta_path=self.config.fasta_path,
            hic_dir=self.config.hic_dir,
            genomic_path=self.config.genomic_path,
            mode='train',
            windows=self.config.windows,
            res=self.config.res,
            output=self.config.output,
            bw=self.config.bwfile,
            val_chroms=self.config.valid_chroms,
            test_chroms=self.config.test_chroms,
            genomic_features=self.config.genomic_features,
            use_aug=self.config.use_aug,
            exclude_bed_path=self.config.exclude_bed_path
        )

        self.val_dataset = GenomicDataset(
            fasta_path=self.config.fasta_path,
            hic_dir=self.config.hic_dir,
            genomic_path=self.config.genomic_path,
            mode='valid',
            windows=self.config.windows,
            res=self.config.res,
            output=self.config.output,
            bw=self.config.bwfile,
            val_chroms=self.config.valid_chroms,
            test_chroms=self.config.test_chroms,
            genomic_features=self.config.genomic_features,
            exclude_bed_path=self.config.exclude_bed_path
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )

    def teardown(self, stage=None):
        self.train_dataset.close()
        self.val_dataset.close()


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []  # Store all predicted values
    all_targets = []  # Store all true values
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Test")
        for batch_idx, (dna, hic) in enumerate(progress_bar):
            # 移动到设备
            dna = dna.to(device)
            hic_target = hic.to(device)

            # Forward propagation
            outputs = model(dna)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            # Calculate loss
            loss = criterion(outputs, hic_target)

            # Update Statistics
            batch_size = dna.size(0)
            batch_loss = loss.item()
            running_loss += batch_loss * batch_size
            total_samples += batch_size

            # Collect predicted and actual values
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(hic_target.detach().cpu().numpy())

            # Calculate the current cumulative average loss value
            avg_loss = running_loss / total_samples

            # Update progress bar - show the loss of the current batch
            progress_bar.set_postfix({
                'batch_loss': f"{batch_loss:.8f}",
                'avg_loss': f"{avg_loss:.8f}"
            })

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate the average performance metrics for the entire epoch
    all_preds_array = np.array(all_preds)  # shape: [N, 256, 256]
    all_targets_array = np.array(all_targets) # shape: [N, 256, 256]

    all_mse = mse(all_preds_array, all_targets_array)
    all_insu = insulation_pearson(all_preds_array, all_targets_array)
    all_pear = pearson_correlation(all_preds_array, all_targets_array)
    all_oe = observed_vs_expected(all_preds_array, all_targets_array)

    avg_insu = np.nanmean(all_insu)
    avg_mse = np.nanmean(all_mse)
    avg_pear = np.nanmean(all_pear)
    avg_oe = np.nanmean(all_oe)
    avg_dis = np.nanmean(distance_stratified_correlation(all_preds_array, all_targets_array),axis=0)
    epoch_loss = running_loss / total_samples

    return epoch_loss, avg_insu, avg_mse, avg_pear, avg_oe, avg_dis,all_mse,all_insu,all_pear,all_oe