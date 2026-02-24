import os
import numpy as np
import torch
from torch import nn

from metrics.metrics import insulation_pearson, pearson_correlation,mse
import matplotlib.pyplot as plt
import pandas as pd
from model.Octopus import Octopus
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature
from preprocess.get_dataset import GenomicDataset
from utils.get_model import get_model
from utils.plot_utils import MatrixPlot
from skimage.transform import resize
class DeleteModel(nn.Module):
    def __init__(self, Octopus, force=None):
        super().__init__()
        self.encoder_seq = Octopus.encoder.conv_start_seq
        self.encoder_seq_res = Octopus.encoder.res_blocks_seq

        self.conv_start_epi = Octopus.encoder.conv_start_epi
        self.res_blocks_epi = Octopus.encoder.res_blocks_epi

        self.Inception = Octopus.encoder.Inception

        self.moe = Octopus.moe
        self.decoder = Octopus.decoder
        self.force = force

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()

        dna = x[:, :5, :]
        dna = self.encoder_seq(dna)
        dna = self.encoder_seq_res(dna)

        epi = x[:, 5:, :]
        epi = self.conv_start_epi(epi)
        epi = self.res_blocks_epi(epi)


        cross = self.Inception(x)

        # Mandatory expert selection
        if self.force == "dna":
            feat = self.moe.ua_seq(dna)
            weights = torch.zeros(dna.size(0), 3, device=dna.device)
            weights[:, 0] = 1.0
        elif self.force == "epi":
            feat = self.moe.ua_epi(epi)
            weights = torch.zeros(dna.size(0), 3, device=dna.device)
            weights[:, 1] = 1.0
        elif self.force == "cross":
            feat = self.moe.cma(cross)
            weights = torch.zeros(dna.size(0), 3, device=dna.device)
            weights[:, 2] = 1.0
        else:
            # 正常 MoE 路由
            feat, weights = self.moe(dna, epi, cross)

        x = feat

        # 5. diagonalize + decoder
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)

        return x, weights

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        return torch.cat([x_i, x_j], dim=1)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_species = 'H1-HESC'
    species = 'H1-HESC'
    bwfile = {'H1hesc.bw': 'log', 'H1hesc_ctcf.bw': 'log'}
    #bwfile = {'sub_merged.bin1.rpkm.bw': 'log', 'sub_merged.bin1.rpkm_cuttag.bw': 'log'}

    epi = len(bwfile)

    model = Octopus(epi).to(device)
    model_name = model.__class__.__name__

    output_path = f"/root/autodl-tmp/output"
    data_path = f"/root/autodl-tmp/data"
    hic_fig_path = output_path + f'/hic_fig/{species}'
    exclude_bed_path = data_path + f"/genome/hg38.bed"
    exclude_regions = GenomicDataset._load_exclude_regions_static(exclude_bed_path)

    genomic_features = True if epi > 0 else False

    model_path = output_path + f'/saved_models/{model_species}/{model_name}_{genomic_features}/best_model.pth'

    model = get_model(model, model_path)

    model.eval()

    chrom = 'chr3'

    fasta_path = data_path + f'/genome/{species}/genome.fa'
    genomic_path = data_path + f'/genomic_features/{species}/'
    hic_dir = data_path + f"/hic/{species}/"
    path = hic_dir + f'{chrom}.npz'
    chrom_hic_bins = GenomicDataset._preload_hic_bins_static(hic_dir)

    fa = DNAFeature(path=fasta_path)
    fa._load()
    chrom_length = fa.chrom_lengths[chrom]
    print(f"{chrom}:{chrom_length}")
    windows = 2097152
    res = 10000
    output = 256

    start_result = {}

    insu_result = {}
    pear_result = {}
    weights = {}

    epi_insu_result = {}
    epi_pear_result = {}
    epi_weight = {}

    dna_insu_result = {}
    dna_pear_result = {}
    dna_weight = {}

    cross_insu_result = {}
    cross_pear_result = {}
    cross_weight = {}


    for seq_start in range(0, chrom_length-2097152, 500000):
        seq_end = seq_start + windows
        if GenomicDataset._is_position_excluded_static(chrom, seq_start, seq_start + windows, exclude_regions):
            print(f'{seq_start} had exclude!')
            continue
        if not GenomicDataset._hic_bin_safe(chrom, seq_start, seq_end, 10000, chrom_hic_bins):
            continue
        os.makedirs(hic_fig_path, exist_ok=True)
        dna_feature = DNAFeature(path=fasta_path)
        key = list(bwfile.keys())
        atac_feater = GenomicFeature(path=genomic_path + key[0], norm=bwfile[key[0]])
        ctcf1_feater = GenomicFeature(path=genomic_path + key[1], norm=bwfile[key[1]])
        dna = dna_feature.get(chrom, seq_start, seq_end)
        deature_tensor = torch.tensor(dna, dtype=torch.float32)  # [self.windows,5]
        input_tensor = deature_tensor.unsqueeze(0).to(device)
        if genomic_features:
            atac = atac_feater.get(chrom, seq_start, seq_end).reshape(-1, 1)
            ctcf1 = ctcf1_feater.get(chrom, seq_start, seq_end).reshape(-1, 1)  # [self.windows,1]
            combined_features = np.concatenate((dna, atac, ctcf1), axis=1)

            input_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)

        hic_feature = HiCFeature(path=path)
        targets = hic_feature.get(seq_start, windows, res)
        targets = resize(targets, (256, 256), anti_aliasing=True)
        targets = np.log(targets + 1)


        pre, weight = model(input_tensor)
        pre = pre.squeeze(0).detach().cpu().numpy()
        weight = weight.squeeze(0).detach().cpu().numpy()

        a2_insu = insulation_pearson(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        a2_pear = pearson_correlation(pre.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        print(f"Octopus Insu:{a2_insu}")

        insu_result[seq_start] = a2_insu
        pear_result[seq_start] = a2_pear
        start_result[seq_start] = seq_start
        weights[seq_start] = weight

        # delete dna
        input_only_epi = input_tensor.clone()
        epi_model = DeleteModel(model, force="epi")

        pre_epi, w1 = epi_model(input_only_epi)

        pre_epi = pre_epi.squeeze(0).detach().cpu().numpy()
        w1 = w1.squeeze(0).detach().cpu().numpy()
        epi_insu = insulation_pearson(pre_epi.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        epi_pear = pearson_correlation(pre_epi.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]

        epi_insu_result[seq_start] = epi_insu
        epi_pear_result[seq_start] = epi_pear
        epi_weight[seq_start] = w1

        # delete epi
        input_only_dna = input_tensor.clone()
        dna_model = DeleteModel(model, force="dna")
        pre_dna, w2 = dna_model(input_only_dna)
        pre_dna = pre_dna.squeeze(0).detach().cpu().numpy()
        w2 = w2.squeeze(0).detach().cpu().numpy()
        dna_insu = insulation_pearson(pre_dna.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        dna_pear = pearson_correlation(pre_dna.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        dna_insu_result[seq_start] = dna_insu
        dna_pear_result[seq_start] = dna_pear
        dna_weight[seq_start] = w2

        input_only_dna = input_tensor.clone()
        cross_model = DeleteModel(model, force="cross")
        pre_cross, w3 = cross_model(input_only_dna)
        pre_cross = pre_cross.squeeze(0).detach().cpu().numpy()
        w3 = w3.squeeze(0).detach().cpu().numpy()
        cross_insu = insulation_pearson(pre_cross.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        cross_pear = pearson_correlation(pre_cross.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
        cross_insu_result[seq_start] = cross_insu
        cross_pear_result[seq_start] = cross_pear
        cross_weight[seq_start] = w3



    records = []
    for start in start_result.keys():
        record = {
            'seq_start': start,
            'insu': insu_result[start][0] if isinstance(insu_result[start], (list, np.ndarray)) else insu_result[start],
            'pear': pear_result[start][0] if isinstance(pear_result[start], (list, np.ndarray)) else pear_result[start],
        }

        # Original weights (3 experts)
        weight = weights[start]
        record['weight1'] = weight[0]
        record['weight2'] = weight[1]
        record['weight3'] = weight[2]

        # epi modal results
        record['epi_insu'] = epi_insu_result[start][0] if isinstance(epi_insu_result[start], (list, np.ndarray)) else \
        epi_insu_result[start]
        record['epi_pear'] = epi_pear_result[start][0] if isinstance(epi_pear_result[start], (list, np.ndarray)) else \
        epi_pear_result[start]
        epi_w = epi_weight[start]
        record['epi_weight1'] = epi_w[0]
        record['epi_weight2'] = epi_w[1]
        record['epi_weight3'] = epi_w[2]

        # DNA modal results
        record['dna_insu'] = dna_insu_result[start][0] if isinstance(dna_insu_result[start], (list, np.ndarray)) else \
        dna_insu_result[start]
        record['dna_pear'] = dna_pear_result[start][0] if isinstance(dna_pear_result[start], (list, np.ndarray)) else \
        dna_pear_result[start]
        dna_w = dna_weight[start]
        record['dna_weight1'] = dna_w[0]
        record['dna_weight2'] = dna_w[1]
        record['dna_weight3'] = dna_w[2]


        record['cross_insu'] = cross_insu_result[start][0] if isinstance(cross_insu_result[start], (list, np.ndarray)) else \
            cross_insu_result[start]
        record['cross_pear'] = cross_pear_result[start][0] if isinstance(cross_pear_result[start], (list, np.ndarray)) else \
            cross_pear_result[start]
        dna_w = cross_weight[start]
        record['cross_weight1'] = dna_w[0]
        record['cross_weight2'] = dna_w[1]
        record['cross_weight3'] = dna_w[2]

        records.append(record)


    df = pd.DataFrame(records)

    excel_path = os.path.join(output_path, f'result_{model_species}_model_test_{species}_data_all.xlsx')
    df.to_excel(excel_path, index=False)
    print(f'All results have been saved to {excel_path}')