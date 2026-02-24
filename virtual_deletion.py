import torch
from skimage.transform import resize
import time
import pandas as pd
from model.Octopus import Octopus
import numpy as np
import matplotlib.pyplot as plt
import os
###############################################
from metrics.metrics import insulation_pearson, pearson_correlation, mse
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature
from preprocess.get_dataset import GenomicDataset
from utils.get_model import get_model

def compute_all_importance(pre_before, pre_after, bin_idx):
    diff = np.abs(pre_before - pre_after)
    # global sum
    global_sum = diff.sum()
    # normalized global
    normalized_global = diff.sum() / (np.sum(np.abs(pre_before)) + 1e-8)

    return {
        "global_sum": global_sum,
        "normalized_global": normalized_global,
    }


def fine_scan_bin(model,
                  combined_features,
                  chrom,
                  seq_start,
                  pre_before,
                  bin_idx,
                  bin_size,
                  del_len=10,
                  step=10,
                  batch_size=8,
                  device=torch.device('cuda:0')):
    """
    Perform a fine-grained scan within the given bin_idx (deleting a 10bp window every 10bp),
    using batch processing to speed up.

    Return a DataFrame.
    """
    feat_dim = combined_features.shape[1]
    bin_start_bp = seq_start + bin_idx * bin_size
    bin_end_bp = bin_start_bp + bin_size

    # 1.Collect all starting positions to be deleted
    del_starts = list(range(bin_start_bp, bin_end_bp - del_len + 1, step))
    total_dels = len(del_starts)
    results = []

    # 2.Pre-create the modified sequence (numpy array) for all delete operations
    modified_seqs = []
    for del_start in del_starts:
        new_seq = combined_features.copy()
        rel_start = del_start - seq_start
        rel_end = rel_start + del_len
        # DNA -> N
        new_seq[rel_start:rel_end, :5] = 0.0
        new_seq[rel_start:rel_end, 4] = 1.0  # One-hot encoding representing 'N'
        if feat_dim > 5:
            new_seq[rel_start:rel_end, 5:] = 0.0  # Reset epigenetic signals
        modified_seqs.append(new_seq)

    # 3.Perform model predictions in batches
    all_after_preds = []
    with torch.no_grad():
        for i in range(0, total_dels, batch_size):
            batch_seqs = modified_seqs[i:i+batch_size]
            # Stack the numpy arrays in the list into a batch tensor
            batch_tensor = torch.tensor(np.stack(batch_seqs, axis=0), dtype=torch.float32).to(device)
            preds = model(batch_tensor)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            # Convert the prediction results to numpy and store them
            batch_preds = preds.detach().cpu().numpy()
            all_after_preds.append(batch_preds)
        # Merge the prediction results of all batches
        all_after_preds = np.vstack(all_after_preds)  # [total_dels, 256, 256]

    # 4. Calculate the score after each deletion operation
    for idx, del_start in enumerate(del_starts):
        del_end = del_start + del_len
        after = all_after_preds[idx]  # Obtain the corresponding prediction results
        scores = compute_all_importance(pre_before, after, bin_idx)
        row = {
            "chrom": chrom,
            "window_start": seq_start,
            "bin_idx": bin_idx,
            "del_start": del_start,
            "del_end": del_end
        }
        row.update(scores)
        results.append(row)

    return pd.DataFrame(results)


def segment_deletion_importance(
    model,
    combined_features,
    pre_before,
    seq_start,
    windows=2097152,
    segments=256,
    del_times=10,
    del_len=10,
    batch_size=8,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):
    """
    For a 2M region divided into segments fragments (default 256).
    For each fragment, simultaneously select del_times starting positions uniformly within that fragment, and delete del_len bp consecutively from each starting point.
    Set the DNA at these positions to N (one-hot vector all zeros with the 4th column set to 1), and set ATAC/CTCF signals to all zeros.
    Then compare the model predictions pre_before and pre_after, and compute the interaction change between this fragment and all other fragments as the importance score for the fragment:
    importance[i] = sum_{j != i} |pre_before[i, j] - pre_after[i, j]|

    Return:
        importances (np.array, length = segments)
        pre_before (256 x 256 numpy array)
        pre_after_all (segments x 256 x 256 numpy array)
        # predictions after deletion for each fragment (if memory usage is too high, this can be omitted)
    Notes:
        Supports batch prediction (batch_size).
        May be memory-intensive on GPU; batch_size can be reduced if needed.
    """

    feat_dim = combined_features.shape[1]

    # Calculate segment boundaries
    bin_size = windows // segments  # bp per segment
    importances = np.zeros(segments, dtype=float)

    # We will generate the 'input after deletion' for each segment and feed them into the model in batches.
    # But constructing 256 large tensors would be huge, so use batching (batch_size).
    # Pre-construct all the modified numpy arrays into a list, then convert them to tensors in batches and make predictions
    modified_inputs = []  # will hold numpy arrays shape [windows, feat_dim] for each segment

    for seg_idx in range(segments):
        seg_start_bp = seq_start + seg_idx * bin_size
        seg_end_bp = seg_start_bp + bin_size

        # Calculate the del_times starting positions to be deleted in the segment, distribute them as evenly as possible,
        # and ensure that the deleted segments are completely within the segment.

        # Use linspace to generate starting coordinates for del_times positions evenly
        starts = np.linspace(seg_start_bp, seg_end_bp - del_len, num=del_times)
        starts = np.floor(starts).astype(int)

        # Copy the original combined_features, then perform deletion at each start
        new_seq = combined_features.copy()
        for s in starts:
            rel_start = s - seq_start  # Relative index to new_seq
            rel_end = rel_start + del_len
            # boundary safety
            rel_start = max(0, rel_start)
            rel_end = min(windows, rel_end)
            # DNA to N:
            new_seq[rel_start:rel_end, :5] = 0.0
            new_seq[rel_start:rel_end, 4] = 1.0
            # epi tracks (from col 5 onwards) to 0
            if feat_dim > 5:
                new_seq[rel_start:rel_end, 5:] = 0.0

        modified_inputs.append(new_seq)   # (256, 2097152, 7)

    pre_after_all = []  # will store numpy arrays (256,256) per segment
    total = len(modified_inputs)  # 256
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch_np = np.stack(modified_inputs[i:i+batch_size], axis=0)  # [B, windows, feat]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)
            preds = model(batch_tensor)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            preds = preds.detach().cpu().numpy()  # [B, 256, 256]
            # ensure shape
            for b in range(preds.shape[0]):
                pre_after_all.append(preds[b])

    pre_after_all = np.stack(pre_after_all, axis=0)  # [segments, 256, 256]
    # Calculate the impact score of each segment on other segments
    for seg_idx in range(segments):
        before = pre_before   # length 256
        after = pre_after_all[seg_idx]
        diff = np.abs(before - after)
        diff[seg_idx,:] = 0.0  # Exclude oneself
        diff[:,seg_idx] = 0.0

        # Calculate global relative change
        importance_global = np.sum(diff) / (np.sum(np.abs(pre_before)) + 1e-8)

        importances[seg_idx] = importance_global

    return importances, pre_after_all


def plot_importance_epi_tracks(importances, atac_bins, ctcf_bins,
                               chrom, seq_start, windows=2097152,
                               save_dir="./", filename="importance_vs_epi.png"):
    """
    Three-row subgraph：importances, ATAC, CTCF
    Args:
        importances: (256,) importance arrays
        atac_bins, ctcf_bins: (256,)
    """
    seq_end = seq_start + windows
    x = np.arange(len(importances))

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1]})

    # --- 1. importance ---
    axes[0].plot(x, importances, color="green", linewidth=2)
    axes[0].fill_between(x, importances, alpha=0.3, color="green")
    axes[0].set_ylabel("Importance")
    axes[0].set_title(f"{chrom}:{seq_start}-{seq_end} Importance vs ATAC/CTCF")

    # --- 2. ATAC ---
    axes[1].plot(x, atac_bins, color="blue", linewidth=2)
    axes[1].fill_between(x, atac_bins, alpha=0.3, color="blue")
    axes[1].set_ylabel("ATAC")

    # --- 3. CTCF ---
    axes[2].plot(x, ctcf_bins, color="orange", linewidth=2)
    axes[2].fill_between(x, ctcf_bins, alpha=0.3, color="orange")
    axes[2].set_ylabel("CTCF")
    axes[2].set_xlabel("Bin (2M / 256)")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_epi_tracks(atac_feature, ctcf_feature, chrom, seq_start, windows=2097152, segments=256,
                    save_dir="./", filename="epi_tracks.png"):
    """
    Average the ATAC/CTCF signals over a 2M region using 256 bins and plot it

    Args:
        atac_feature, ctcf_feature: GenomicFeature
        chrom
        seq_start
        windows: Interval length (Default 2M)
        segments: Number of sections (Default 256)
    """

    seq_end = seq_start + windows
    atac = atac_feature.get(chrom, seq_start, seq_end).reshape(-1)  # shape: (windows,)
    ctcf = ctcf_feature.get(chrom, seq_start, seq_end).reshape(-1)

    bin_size = windows // segments
    atac_bins = []
    ctcf_bins = []
    for i in range(segments):
        s = i * bin_size
        e = (i + 1) * bin_size
        atac_bins.append(np.mean(atac[s:e]))
        ctcf_bins.append(np.mean(ctcf[s:e]))

    atac_bins = np.array(atac_bins)
    ctcf_bins = np.array(ctcf_bins)

    return  atac_bins, ctcf_bins

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    species = 'cotton'
    model_species = 'cotton'
    bwfile = {'sub_merged.bin1.rpkm.bw': 'log', 'sub_merged.bin1.rpkm_cuttag.bw': 'log'}
    epi = len(bwfile)

    model = Octopus(epi).to(device)
    model_name = model.__class__.__name__
    output_path = f"output"
    data_path = f"data"
    hic_fig_path = output_path + f'/hic_fig/{species}'

    genomic_features = True if epi > 0 else False

    model_path = f'data/model_weights/cotton_best_model.pth'
    model = get_model(model, model_path)
    model.eval()

    windows = 2097152
    del_len = 10
    batch_size = 16
    fasta_path = data_path + f'/genome/{species}/genome.fa'
    genomic_path = data_path + f'/genomic_features/{species}/'
    hic_dir = data_path + f"/hic/{species}/"

    # exclude_bed_path = data_path + f"/genome/hg38.bed"
    # exclude_regions = GenomicDataset._load_exclude_regions_static(exclude_bed_path)

    chrom_hic_bins = GenomicDataset._preload_hic_bins_static(hic_dir)
    dna_feature = DNAFeature(path=fasta_path)
    dna_feature._load()
    key = list(bwfile.keys())
    atac_feature = GenomicFeature(path=genomic_path + key[0], norm=bwfile[key[0]])
    ctcf_feature = GenomicFeature(path=genomic_path + key[1], norm=bwfile[key[1]])


    #chrom_list = ['HC04_A01', 'HC04_D01','HC04_A02', 'HC04_D02','HC04_A03', 'HC04_D03']
    #chrom_list = [ 'HC04_A04', 'HC04_D04','HC04_A05', 'HC04_D05','HC04_A06', 'HC04_D06']
    #chrom_list = ['HC04_A07', 'HC04_D07','HC04_A08', 'HC04_D08','HC04_A09', 'HC04_D09','HC04_A10', 'HC04_D10']
    chrom_list = ['HC04_A11', 'HC04_D11','HC04_A12', 'HC04_D12', 'HC04_A13', 'HC04_D13']
    for chrom in dna_feature.chrom_lengths:
        if chrom in chrom_list:
            path = hic_dir + f'{chrom}.npz'
            hic_feature = HiCFeature(path=path)
            chrom_length = dna_feature.chrom_lengths[chrom]
            print(f"{chrom}:{chrom_length}")
            importance_fig = output_path + f"/explanation_{species}/importance_fig/{chrom}"
            os.makedirs(importance_fig, exist_ok=True)
            importance_excel = output_path + f"/explanation_{species}/importance_10bp/{chrom}"
            os.makedirs(importance_excel, exist_ok=True)
            print(f'chrom_bins:{chrom_hic_bins[chrom]}')
            chrom_bin_scores = []  # Save scores of all bins in the current chromosome
            for seq_start in range(0, chrom_length - 5000000, 2097152):
                seq_end = seq_start + windows
                '''if GenomicDataset._is_position_excluded_static(chrom, seq_start, seq_start + windows, exclude_regions):
                    print(f'{seq_start} had exclude!')
                    continue'''
                if not GenomicDataset._hic_bin_safe(chrom, seq_start, seq_end, 10000, chrom_hic_bins):
                    continue
                start_time = time.time()
                model.eval()

                dna = dna_feature.get(chrom, seq_start, seq_end)  # [windows, 5]
                atac = atac_feature.get(chrom, seq_start, seq_end).reshape(-1, 1)
                ctcf = ctcf_feature.get(chrom, seq_start, seq_end).reshape(-1, 1)
                combined_features = np.concatenate((dna, atac, ctcf), axis=1)
                # Original input tensor
                input_before = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(
                    device)  # [1, windows, feat]
                targets = hic_feature.get(seq_start, windows, 10000)
                targets = resize(targets, (256, 256), anti_aliasing=True)
                targets = np.log(targets + 1)

                # Model Prediction Original
                with torch.no_grad():
                    pre_before = model(input_before)
                    if isinstance(pre_before, (tuple, list)):
                        pre_before = pre_before[0]
                pre_before = pre_before.squeeze(0).detach().cpu().numpy()  # (256, 256)

                before_insu = insulation_pearson(pre_before.reshape(-1, 256, 256), targets.reshape(-1, 256, 256))[0]
                if before_insu < 0.8:
                    print(f'before_insu:{before_insu} exclude')
                    continue

                importances, pre_after_all = segment_deletion_importance(
                    model, combined_features, pre_before, seq_start=seq_start, windows=windows,
                    segments=256, del_times=10, del_len=del_len,
                    batch_size=batch_size, device=device
                )

                for bin_idx, score in enumerate(importances):
                    chrom_bin_scores.append((seq_start, bin_idx, score))

                atac_bins, ctcf_bins = plot_epi_tracks(
                    atac_feature, ctcf_feature, chrom=chrom, seq_start=seq_start,
                    windows=windows, segments=256,
                )

                savefig = plot_importance_epi_tracks(
                    importances, atac_bins, ctcf_bins,
                    chrom=chrom, seq_start=seq_start,
                    save_dir=importance_fig, filename=f"{chrom}_{seq_start}_importance_vs_epi.png"
                )
                print(f"Save comparison image to:{savefig}")

                elapsed = time.time() - start_time
                print(f"{chrom}:{seq_start}-{seq_start + windows} execution time: {elapsed / 60:.2f} min ({elapsed:.1f} s)")

            start_time = time.time()
            df = pd.DataFrame(chrom_bin_scores, columns=["seq_start", "bin_idx", "score"])
            threshold = df["score"].quantile(0.99)
            df_high = df[df["score"] >= threshold]
            print(f"{chrom}: Reserved {len(df_high)}/{len(df)} bins (>= {threshold:.4f})")

            for _, row in df_high.iterrows():
                seq_start, bin_idx = int(row["seq_start"]), int(row["bin_idx"])
                bin_size = windows // 256

                path = hic_dir + f'{chrom}.npz'
                hic_feature = HiCFeature(path=path)

                dna = dna_feature.get(chrom, seq_start, seq_start + windows)
                atac = atac_feature.get(chrom, seq_start, seq_start + windows).reshape(-1, 1)
                ctcf = ctcf_feature.get(chrom, seq_start, seq_start + windows).reshape(-1, 1)
                combined_features = np.concatenate((dna, atac, ctcf), axis=1)

                targets = hic_feature.get(seq_start, windows, 10000)
                targets = resize(targets, (256, 256), anti_aliasing=True)
                targets = np.log(targets + 1)

                input_before = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pre_before = model(input_before)
                    if isinstance(pre_before, (tuple, list)):
                        pre_before = pre_before[0]
                pre_before = pre_before.squeeze(0).detach().cpu().numpy()

                fine_df = fine_scan_bin(
                    model, combined_features,
                    chrom=chrom, seq_start=seq_start,
                    pre_before=pre_before, bin_idx=bin_idx,bin_size=bin_size,
                    del_len=10, step=10, batch_size=batch_size, device=device
                )

                excel_name = f"{seq_start}_bin{bin_idx}_fine_scan.xlsx"
                excel_path = os.path.join(importance_excel, excel_name)
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                fine_df.to_excel(excel_path, index=False)
                print(f"Save fine-grained scan results to:{excel_path}")
            elapsed = time.time() - start_time
            print(f"chrom:{chrom} has processed!!! Execution time: {elapsed / 60:.2f} min ({elapsed:.1f} s)")
