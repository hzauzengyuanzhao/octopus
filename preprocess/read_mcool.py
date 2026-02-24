import numpy as np
from cooler import Cooler
import cooler
import os

def main(path, save_path, resolution, window_size, balance=False):
    full_path = f'{path}::/resolutions/{resolution}'
    hic = Cooler(full_path)
    data = hic.matrix(balance=balance, sparse=True)
    for chrom in hic.chromnames:
        mat = data.fetch(chrom)
        diags = compress_diag(mat, window_size)
        ucsc_chrom = f'{chrom}.npz'
        chrom_path = f'{save_path}/{ucsc_chrom}'
        os.makedirs(f'{save_path}/', exist_ok=True)
        np.savez(chrom_path, **diags)
        print(f'npz has saved {chrom_path}')

def compress_diag(mat, window):
    # NOTE: dict is probably suboptimal here. We could have a big list double the window_size
    diag_dict = {}
    for d in range(window):
        diag_dict[str(d)] = np.nan_to_num(mat.diagonal(d).astype(np.float32))
        diag_dict[str(-d)] = np.nan_to_num(mat.diagonal(-d).astype(np.float32))
    return diag_dict

if __name__ == '__main__':
    path = f'/root/autodl-tmp/data/hic/Theobroma-cacao/SRR28464201_R1.mcool'
    save_path = f'/root/autodl-tmp/data/hic/Theobroma-cacao'
    resolution = 10000
    window_size = 256
    main(path, save_path, resolution, window_size)
    print(f'All sample has processed!!')
# nohup python  > cool_output.log 2>&1 &