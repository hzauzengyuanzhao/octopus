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
    将所有染色体的矩阵保存到单个cool文件中

    Parameters:
    -----------
    matrices_dict : dict
        键为染色体名称，值为该染色体的接触矩阵
    resolution : int
        分辨率
    output_cool : str
        输出cool文件路径
    species_name : str
        物种名称
    """

    from itertools import chain

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_cool), exist_ok=True)

    # 如果文件已存在，先删除
    if os.path.exists(output_cool):
        os.remove(output_cool)

    # 按染色体顺序处理（可以按名称排序）
    chroms = sorted(matrices_dict.keys())

    # 收集所有bin信息
    all_bins = []
    chrom_bin_offsets = {}  # 记录每个染色体在全局bin中的偏移量

    global_bin_idx = 0
    for chrom in chroms:
        matrix = matrices_dict[chrom]
        n_bins = matrix.shape[0]

        # 记录这个染色体在全局bin中的起始位置
        chrom_bin_offsets[chrom] = global_bin_idx

        # 创建这个染色体的bin信息
        chrom_bins = pd.DataFrame({
            "chrom": [chrom] * n_bins,
            "start": np.arange(n_bins) * resolution,
            "end": (np.arange(n_bins) + 1) * resolution
        })

        all_bins.append(chrom_bins)
        global_bin_idx += n_bins

    # 合并所有bin
    bins_df = pd.concat(all_bins, ignore_index=True)

    # 收集所有像素数据
    all_pixels = []

    for chrom in chroms:
        matrix = matrices_dict[chrom]
        offset = chrom_bin_offsets[chrom]
        n_bins = matrix.shape[0]

        # 只处理上三角部分（包括对角线）
        M_upper = scipy.sparse.triu(matrix, k=0, format="coo")
        mask = M_upper.data > 0

        if np.any(mask):
            # 调整索引到全局bin编号
            global_rows = M_upper.row[mask] + offset
            global_cols = M_upper.col[mask] + offset

            pixels_chrom = pd.DataFrame({
                "bin1_id": global_rows.astype(np.int32),
                "bin2_id": global_cols.astype(np.int32),
                "count": M_upper.data[mask].astype(np.float64)
            })

            all_pixels.append(pixels_chrom)

    # 合并所有像素数据
    if all_pixels:
        pixels_df = pd.concat(all_pixels, ignore_index=True)
    else:
        # 如果没有有效像素，创建空的DataFrame
        pixels_df = pd.DataFrame({
            "bin1_id": [],
            "bin2_id": [],
            "count": []
        })

    # 保存为cool文件
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
    # 是当前窗口在全染色体中的起始 Bin 索引。
    # row1 是结束索引。使用 min 是为了防止染色体末端的窗口超出数组边界（越界保护）。
    row1 = min(n_bins, row0 + h)

    if band is None:
        sum_arr[row0:row1, row0:row0 + w] += numer[:row1 - row0]
        wsum_arr[row0:row1, row0:row0 + w] += denom[:row1 - row0]
        return
    b = band
    for i_loc in range(row1 - row0):
        i = row0 + i_loc  # 全局的行索引
        # 确定当前行在全局范围内需要处理的列区间 [j0, j1)
        # 全局指的是 n_bins * n_bins这个尺度下，每一个 (h, w) 的对角线和 (n_bins , n_bins)对角线重合的全局列索引j0,j1
        j0 = max(row0, i - b)  # 全局列索引 左边界
        j1 = min(row0 + w, i + b + 1)  # 全局列索引 右边界
        if j0 >= j1:
            continue
        jl = j0 - row0  # 计算在局部窗口 numer 中的左边界
        jr = j1 - row0  # 计算在局部窗口 numer 中的右边界
        bl = j0 - (i - b)  # 关键映射：将全局列坐标 j0 转换为带状矩阵中的存储列索引
        br = bl + (j1 - j0)  # 对应的带状矩阵右边界
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
    numer = L.T @ Aw @ L  # [209,209]  一次预测的真实值*权重
    denom = L.T @ patch_weight @ L  # [209,209]  一次预测的真实权重

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

    # ===== 加载模型（你自己改这里）=====
    """model = MappingModel(0, teacher_model=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() """

    model = MappingModel(0, teacher_model=None).to(device)
    # model_name = model.__class__.__name__
    model_path = f'/autodl-fs/data/Fine-tune_meta_test/all/train/all/saved_models/best_model.pth'
    model = get_mapping_model(model, model_path)
    model.eval()

    # ===== DNA =====
    from preprocess.data_feature import DNAFeature
    data_path = f"/root/autodl-fs/zs_data1/select_species"
    output_path = f"/root/autodl-fs/pre_hic_output/zs_data1"
    genome_dir = os.path.join(data_path, args.species)
    fa_files = glob.glob(os.path.join(genome_dir, "*.fa")) + glob.glob(os.path.join(genome_dir, "*.fasta"))
    if len(fa_files) == 0:
        raise FileNotFoundError(f"No fasta found in: {genome_dir}")
    fasta_path = fa_files[0]

    dna = DNAFeature(path=fasta_path)
    dna._load()

    species_output_dir = os.path.join(output_path, args.species)
    os.makedirs(species_output_dir, exist_ok=True)

    # 存储所有染色体的矩阵
    all_chrom_matrices = {}
    t_species = time.time()

    for chrom in dna.chroms:
        t_chrom = time.time()
        chrom_length = dna.chrom_lengths[chrom]
        # ===== merge init =====
        n_bins = math.ceil(chrom_length / args.resolution)
        sum_arr, wsum_arr = allocate_accumulators(
            n_bins, args.band, dtype=np.float32
        )
        patch_weight = make_weight((256, 256), "hann")
        # ===== streaming predict + merge =====

        starts = list(range(0, chrom_length - args.window + 1, args.step))

        # 补最后一个“左移窗口”
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

        # ===== 得到最终的矩阵 =====
        M = finalize(sum_arr, wsum_arr, args.band)

        # 存储这个染色体的矩阵
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