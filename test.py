import os
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from model.Octopus import Octopus
from preprocess.get_dataset import GenomicDataset, collate_fn
from utils.get_model import get_model
from utils.train_process import test_epoch

torch.manual_seed(42)
np.random.seed(42)


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    species = 'cotton'
    model_species = 'cotton'
    bwfile = {'sub_merged.bin1.rpkm.bw': 'log', 'sub_merged.bin1.rpkm_cuttag.bw': 'log'}

    valid_chroms = ['HC04_A06', 'HC04_D06', 'HC04_A07', 'HC04_D07']
    test_chroms = ['HC04_A05', 'HC04_D05']


    output_path = f"output"
    data_path = f"data"

    epi = len(bwfile)
    model = Octopus(epi).to(device)
    genomic_features = True if epi > 0 else False
    windows = 2097152

    res = 10000
    output = 256
    batch_size = 4

    # Path Configuration
    fasta_path = data_path + f'/genome/{species}/genome.fa'
    genomic_path = data_path + f'/genomic_features/{species}/'
    hic_dir = data_path + f"/hic/{species}/"

    # Model Save Path
    model_path = f'data/model_weights/cotton_best_model.pth'
    # Logs and Results Saving
    log_dir = output_path + f"/logs/{species}/{model.__class__.__name__}_{genomic_features}/{model_species}_model_test/"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, f"{model_species}_model_test_results.txt")
    plot_dis_path = os.path.join(log_dir, f"{model_species}_model_test_dis_plot.png")
    excel_path = os.path.join(log_dir, f"{model_species}_model_test_{species}_per_sample_correlations.xlsx")
config = Config()



if __name__ == '__main__':
    model = get_model(config.model, config.model_path)
    model.eval()

    test_dataset = GenomicDataset(
        fasta_path=config.fasta_path,
        hic_dir=config.hic_dir,
        genomic_path=config.genomic_path,
        mode='test',
        windows=config.windows,
        res=config.res,
        output=config.output,
        bw=config.bwfile,
        val_chroms=config.valid_chroms,
        test_chroms=config.test_chroms,
        genomic_features=config.genomic_features
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    criterion = nn.MSELoss()

    test_loss, test_insu_corr, test_mse, test_pear, test_oe, test_dis, all_mse, all_insu, all_pear, all_oe = test_epoch(
        model, test_loader, criterion, config.device
    )

    log_lines = [
        f"Best Test loss: {test_loss:.8f}",
        f"Best Test Mse: {test_mse:.8f}",
        f"Best Test Insu correlation: {test_insu_corr:.4f}",
        f"Best Test Pearson correlation: {test_pear:.4f}",
        f"Best Test Observed vs expected: {test_oe:.4f}"
    ]

    # Print to terminal
    for line in log_lines:
        print(line)

    # Write to file
    with open(config.results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    plt.figure(figsize=(12, 8))
    plt.plot(test_dis, marker='o', linestyle='-', color='b')
    plt.title('baes_val_dises')
    plt.xlabel('The position from the diagonal')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.savefig(config.plot_dis_path)
    plt.close()

    if config.species == config.model_species:
        df = pd.DataFrame({
            "Mse": all_mse,
            "Pearson Correlation": all_pear,
            "Insulation Correlation": all_insu,
            "OE": all_oe
        })
        df.to_excel(config.excel_path, index=False)
        print(f"Saved the Pearson and insulation correlation coefficients for each sample to:{config.excel_path}")
    # Close dataset
    test_dataset.close()
