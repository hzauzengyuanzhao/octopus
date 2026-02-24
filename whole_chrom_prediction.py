import os
import math
import argparse
import numpy as np
import torch
import pandas as pd
import glob
import time
import cooler
import scipy.sparse
##########################

from model.MappingModel import MappingModel
from utils.get_model import get_model, get_mapping_model

def save_matrix_as_cool(matrix, chrom, resolution, output_cool):

    n_bins = matrix.shape[0]
    matrix = matrix.astype(np.float64, copy=False)

    bins = pd.DataFrame({
        "chrom": [chrom] * n_bins,
        "start": np.arange(n_bins) * resolution,
        "end": (np.arange(n_bins) + 1) * resolution
    })

    M_upper = scipy.sparse.triu(matrix, k=0, format="coo")
    mask = M_upper.data > 0

    pixels = {
        "bin1_id": M_upper.row[mask].astype(np.int32),
        "bin2_id": M_upper.col[mask].astype(np.int32),
        "count": M_upper.data[mask].astype(np.float64)
    }
    os.makedirs(os.path.dirname(output_cool), exist_ok=True)

    if os.path.exists(output_cool):
        os.remove(output_cool)

    cooler.create_cooler(
        output_cool,
        bins,
        pixels,
        dtypes={"count": np.float64}
    )


def save_matrices_as_single_cool(matrices_dict, resolution, output_cool, species_name):
    """
    Save the matrices of all chromosomes into a single cool file

    Parameters:
    -----------
    matrices_dict : dict
        The key is the chromosome name, and the value is the contact matrix of that chromosome.
    resolution : int
    output_cool : str
        Output cool file path
    species_name : str
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_cool), exist_ok=True)
    # If the file already exists, delete it first
    if os.path.exists(output_cool):
        os.remove(output_cool)

    # Processed in chromosome order (sorted by name)
    chroms = sorted(matrices_dict.keys())

    # Collect all bin information
    all_bins = []
    chrom_bin_offsets = {}

    global_bin_idx = 0
    for chrom in chroms:
        matrix = matrices_dict[chrom]
        n_bins = matrix.shape[0]

        # Record the starting position of this chromosome in the global bin
        chrom_bin_offsets[chrom] = global_bin_idx

        # Create the bin information for this chromosome
        chrom_bins = pd.DataFrame({
            "chrom": [chrom] * n_bins,
            "start": np.arange(n_bins) * resolution,
            "end": (np.arange(n_bins) + 1) * resolution
        })

        all_bins.append(chrom_bins)
        global_bin_idx += n_bins

    # Merge all bins
    bins_df = pd.concat(all_bins, ignore_index=True)

    all_pixels = []

    for chrom in chroms:
        matrix = matrices_dict[chrom]
        offset = chrom_bin_offsets[chrom]
        n_bins = matrix.shape[0]

        # Only process the upper triangular part (including the diagonal)
        M_upper = scipy.sparse.triu(matrix, k=0, format="coo")
        mask = M_upper.data > 0

        if np.any(mask):
            # Adjust the index to the global bin number
            global_rows = M_upper.row[mask] + offset
            global_cols = M_upper.col[mask] + offset

            pixels_chrom = pd.DataFrame({
                "bin1_id": global_rows.astype(np.int32),
                "bin2_id": global_cols.astype(np.int32),
                "count": M_upper.data[mask].astype(np.float64)
            })

            all_pixels.append(pixels_chrom)

    if all_pixels:
        pixels_df = pd.concat(all_pixels, ignore_index=True)
    else:
        pixels_df = pd.DataFrame({
            "bin1_id": [],
            "bin2_id": [],
            "count": []
        })

    # Save as a cool file
    cooler.create_cooler(
        output_cool,
        bins_df,
        pixels_df,
        dtypes={"count": np.float64},
        assembly=species_name
    )

    print(f"Saved {len(chroms)} chromosomes to {output_cool}")
    print(f"Total bins: {len(bins_df)}")
    print(f"Total pixels: {len(pixels_df)}")


# =========================
# merge utilities
# =========================
def make_weight(shape, mode="hann", eps=1e-8):
    H, W = shape
    if mode == "uniform":
        return np.ones((H, W), dtype=np.float64)
    hy = np.hanning(H)
    hx = np.hanning(W)
    return np.outer(hy, hx) + eps


def interval_overlap_matrix(in_start, in_step, n_in, out_start, n_out, out_step):
    in_edges = in_start + np.arange(n_in + 1) * in_step
    out_edges = out_start + np.arange(n_out + 1) * out_step
    left = np.maximum(in_edges[:-1][:, None], out_edges[:-1][None, :])
    right = np.minimum(in_edges[1:][:, None], out_edges[1:][None, :])
    return np.clip(right - left, 0.0, None)


def allocate_accumulators(n_bins, band, dtype=np.float32):
    if band is None:
        return (
            np.zeros((n_bins, n_bins), dtype=dtype),
            np.zeros((n_bins, n_bins), dtype=dtype)
        )
    bw = 2 * band + 1
    return (
        np.zeros((n_bins, bw), dtype=dtype),
        np.zeros((n_bins, bw), dtype=dtype)
    )


def add_local(sum_arr, wsum_arr, numer, denom, row0, n_bins, band):
    h, w = numer.shape  # [209,209]
    # row0 is the starting bin index of the current window across the entire chromosome
    # row1 is the end index
    row1 = min(n_bins, row0 + h)

    if band is None:
        sum_arr[row0:row1, row0:row0 + w] += numer[:row1 - row0]
        wsum_arr[row0:row1, row0:row0 + w] += denom[:row1 - row0]
        return
    b = band
    for i_loc in range(row1 - row0):
        i = row0 + i_loc  # Global row index
        # Determine the column range [j0, j1) that needs to be processed for the current row globally
        j0 = max(row0, i - b)  # left Boundary
        j1 = min(row0 + w, i + b + 1)  # right Boundary
        if j0 >= j1:
            continue
        jl = j0 - row0  # Calculate the left boundary in the local window 'numer'
        jr = j1 - row0  # Calculate the right boundary in the local window 'numer'
        bl = j0 - (i - b)  # Convert the global column coordinate j0 to the storage column index in the banded matrix
        br = bl + (j1 - j0)  # right boundary of the corresponding banded matrix
        sum_arr[i, bl:br] += numer[i_loc, jl:jr]
        wsum_arr[i, bl:br] += denom[i_loc, jl:jr]


def finalize(sum_arr, wsum_arr, band):
    if band is None:
        M = np.where(wsum_arr > 0, sum_arr / wsum_arr, 0.0)
        M = (M + M.T) / 2
        return M

    n_bins, bw = sum_arr.shape
    b = (bw - 1) // 2
    out = np.zeros((n_bins, n_bins), dtype=sum_arr.dtype)

    for i in range(n_bins):
        j0 = max(0, i - b)
        j1 = min(n_bins, i + b + 1)
        bl = j0 - (i - b)
        br = bl + (j1 - j0)
        num = sum_arr[i, bl:br]
        den = wsum_arr[i, bl:br]
        out[i, j0:j1] = np.where(den > 0, num / den, 0.0)

    out = (out + out.T) / 2
    return out


def merge_one_patch(A, start, end, sum_arr, wsum_arr,
                    resolution, patch_weight, band):
    P = A.shape[0]  # 256
    s_in = (end - start) / P  # 8152

    out_i0 = start // resolution  # 开始位置的bin
    out_i1 = math.ceil(end / resolution)  # 结尾位置的bin
    h = out_i1 - out_i0  # 209
    if h <= 0:
        return

    # 计算重叠的区间权重矩阵 (n_in, n_out)
    L = interval_overlap_matrix(
        start, s_in, P,
        out_i0 * resolution, h, resolution
    )

    Aw = A * patch_weight
    numer = L.T @ Aw @ L  # [209,209]
    denom = L.T @ patch_weight @ L  # [209,209]

    add_local(sum_arr, wsum_arr, numer, denom, out_i0,
              n_bins=sum_arr.shape[0], band=band)



def chunked(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--resolution", type=int, default=10000)
    parser.add_argument("--window", type=int, default=2097152)
    parser.add_argument("--step", type=int, default=2097152 // 8)
    parser.add_argument("--band", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # load model
    """model = MappingModel(0, teacher_model=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() """

    model = MappingModel(0, teacher_model=None).to(device)
    # model_name = model.__class__.__name__
    model_path = f'data/model_weights/Fine-tune_model.pth'
    model = get_mapping_model(model, model_path)
    model.eval()

    # load DNA
    from preprocess.data_feature import DNAFeature
    data_path = f"data/select_species/"
    output_path = f"output/pre_cool/"
    genome_dir = os.path.join(data_path, args.species)
    fa_files = glob.glob(os.path.join(genome_dir, "*.fa")) + glob.glob(os.path.join(genome_dir, "*.fasta"))
    if len(fa_files) == 0:
        raise FileNotFoundError(f"No fasta found in: {genome_dir}")
    fasta_path = fa_files[0]

    dna = DNAFeature(path=fasta_path)
    dna._load()

    species_output_dir = os.path.join(output_path, args.species)
    os.makedirs(species_output_dir, exist_ok=True)

    all_chrom_matrices = {}
    t_species = time.time()

    for chrom in dna.chroms:
        t_chrom = time.time()
        chrom_length = dna.chrom_lengths[chrom]
        # merge init
        n_bins = math.ceil(chrom_length / args.resolution)
        sum_arr, wsum_arr = allocate_accumulators(
            n_bins, args.band, dtype=np.float32
        )
        patch_weight = make_weight((256, 256), "hann")
        # streaming predict + merge

        starts = list(range(0, chrom_length - args.window + 1, args.step))

        # Fill in the last 'Shift Left Window'
        last_start = chrom_length - args.window
        if starts[-1] != last_start:
            starts.append(last_start)

        for batch_starts in chunked(starts, args.batch_size):
            seqs, valid_starts = [], []

            for start in batch_starts:
                seqs.append(dna.get(chrom, start, start + args.window))
                valid_starts.append(start)

            x = torch.tensor(np.stack(seqs), dtype=torch.float32, device=device)

            with torch.no_grad():
                preds = model(x)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]

            # during model training the hic values extracted from cool were log(x + 1) transformed for training,
            # here we take the inverse function to restore the values in cool
            preds = np.expm1(np.maximum(preds.cpu().numpy(), 0))

            for i, start in enumerate(valid_starts):
                merge_one_patch(
                    preds[i],
                    start,
                    start + args.window,
                    sum_arr,
                    wsum_arr,
                    args.resolution,
                    patch_weight,
                    args.band
                )

        # obtain the final matrix
        M = finalize(sum_arr, wsum_arr, args.band)

        all_chrom_matrices[chrom] = M

        print(f"{args.species} [{chrom}] processed | shape: {M.shape} | time: {time.time() - t_chrom:.2f}s")

    output_cool_path = os.path.join(species_output_dir, f"{args.species}_10k.cool")
    save_matrices_as_single_cool(
        all_chrom_matrices,
        args.resolution,
        output_cool_path,
        args.species
    )

    print(f"\n{args.species} all chromosomes saved to single file: {output_cool_path}")
    print(f"Total time: {time.time() - t_species:.2f}s")
    print(f"Chromosomes processed: {list(all_chrom_matrices.keys())}")
    print(f"Shapes: { {chrom: M.shape for chrom, M in all_chrom_matrices.items()} }")


if __name__ == "__main__":
    main()