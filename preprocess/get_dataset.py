import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pysam import FastaFile
from skimage.transform import resize
from preprocess.data_feature import HiCFeature, DNAFeature, GenomicFeature

# 常量定义
OFFSET = 500000
BLOCK_SIZE = 5000000  #
MARGIN = 0  #
class GenomicDataset(Dataset):
    def __init__(self, fasta_path, hic_dir, genomic_path, mode='train',
                 windows=2097152, res=10000, output=256, bw=None, train_chroms=None,
                 val_chroms=None, test_chroms=None,
                 genomic_features=False, use_aug=False, exclude_bed_path=None):
        """
        初始化基因组数据集
        参数:
            fasta_path: Path to the reference genome FASTA file
            hic_dir: Directory for storing Hi-C sample data
            genomic_path: Path to genome feature file
            mode: Dataset Pattern ('train', 'valid', 'test')
            windows: sequence length
            res: hic resolution
            output: Output matrix size
            bw: A dictionary containing bw files and normalization methods (log, None)
            genomic_features: Whether to use ATAC and CTCF features
            exclude_bed_path : Exclusion files specific to human data, excluding large N regions of the genome
        """
        # Read data parameters
        self.windows = windows
        self.res = res
        self.output = output
        self.mode = mode
        self.genomic_features = genomic_features
        self.use_aug = use_aug

        # Storage path information
        self.fasta_path = fasta_path
        self.hic_dir = hic_dir
        self.genomic_path = genomic_path

        # Definition of chromosome allocation
        self.test_chroms = test_chroms
        self.valid_chroms = val_chroms
        self.train_chroms = train_chroms
        # Preload chromosome length information
        self.chrom_lengths = self._preload_chrom_lengths(fasta_path)
        self.chrom_hic_bins = self._preload_hic_bins(hic_dir)

        self.exclude_bed_path = exclude_bed_path
        self.exclude_regions = self._load_exclude_regions() if exclude_bed_path else None

        # Sampling location
        self.entries = self._generate_samples()

        # Delay resource initialization
        self.fasta = None
        self.dna_feature = None
        self.bw_files = bw
        self.feature_extractors = {}
        self.hic_features = {}

        print(f"Initialized {mode} dataset with {len(self.entries)} samples")
        print(f"Data augmentation: {'Enabled' if use_aug else 'Disabled'}")
        #print(f"self.chrom_lengths:{self.chrom_lengths}")

    def _load_exclude_regions(self):
        """Load exclusion regions from BED file"""
        exclude_regions = {}
        try:
            with open(self.exclude_bed_path, 'r') as f:
                print(f"opean successfor!")
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    chrom = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])

                    if chrom not in exclude_regions:
                        exclude_regions[chrom] = []
                    exclude_regions[chrom].append((start, end))

            # 对每个染色体的排除区间排序以便高效查询
            for chrom in exclude_regions:
                exclude_regions[chrom].sort(key=lambda x: x[0])

        except Exception as e:
            print(f"Error loading exclude regions: {e}")
            return None

        return exclude_regions

    def _preload_hic_bins(self, hic_dir):
        """预加载每个染色体的HiC矩阵最大bin数"""
        chrom_hic_bins = {}
        for chrom in self.chrom_lengths:
            hic_chrom = f"{chrom}"
            hic_path = os.path.join(hic_dir, f"{hic_chrom}.npz")
            if not os.path.exists(hic_path):
                print(f"Warning: HiC file not found for {chrom}")
                continue
            try:
                # 加载HiC矩阵但不加载实际数据
                with np.load(hic_path) as data:
                    if 'hic' in data:
                        chrom_hic_bins[chrom] = data['hic'].shape[0]
                    else:
                        # 获取第一个数组作为HiC矩阵
                        first_key = data.files[0]
                        chrom_hic_bins[chrom] = data[first_key].shape[0]
            except Exception as e:
                print(f"Error loading HiC bins for {chrom}: {e}")
        return chrom_hic_bins

    def _is_position_excluded(self, chrom, start, end):
        """检查给定位置是否在排除区域内"""
        if not self.exclude_regions or chrom not in self.exclude_regions:
            return False

        # 使用二分查找高效检查重叠
        regions = self.exclude_regions[chrom]
        low, high = 0, len(regions) - 1

        while low <= high:
            mid = (low + high) // 2
            r_start, r_end = regions[mid]

            # 检查重叠：我们的区间 [start, end) 与排除区间 [r_start, r_end) 重叠
            if start < r_end and end > r_start:
                return True

            if end <= r_start:
                high = mid - 1
            else:
                low = mid + 1

        return False

    def _generate_samples(self):
        """根据染色体划分生成样本位置"""
        entries = []
        # 获取所有染色体
        chroms = list(self.chrom_lengths.keys())

        for chrom in chroms:
            if chrom == 'chrY' or chrom == 'chrX':
                continue  # 忽略chrY X，不生成任何样本
            # 根据模式选择染色体
            if self.train_chroms is not None:
                if self.mode == 'train' and chrom not in self.train_chroms:
                    continue
            else:
                if self.mode == 'train' and (chrom in self.test_chroms or chrom in self.valid_chroms):
                    continue
            if self.mode == 'test' and chrom not in self.test_chroms:
                continue
            if self.mode == 'valid' and chrom not in self.valid_chroms:
                continue


            # 获取染色体长度
            chrom_length = self.chrom_lengths[chrom]

            if chrom not in self.chrom_hic_bins:
                print(f"Skipping chromosome {chrom} (no HiC data)")
                continue

            max_hic_bin = self.chrom_hic_bins[chrom]
            max_valid_bp = max_hic_bin * self.res  # 计算有效的最大bp位置

            # 计算有效区域
            start_pos = MARGIN
            end_pos = min(chrom_length, max_valid_bp) - MARGIN

            if end_pos - start_pos < self.windows:
                print(f"Skipping chromosome {chrom} (insufficient valid region: {end_pos - start_pos} < {self.windows})")
                continue
            # 划分采样区域
            current = start_pos
            while current + BLOCK_SIZE <= end_pos:
                if self.exclude_regions and self._is_position_excluded(chrom, current, current + BLOCK_SIZE):
                    current += OFFSET
                    continue  # 跳过排除区域
                entries.append({
                    'chrom': chrom,
                    'start': current,
                    'end': current + BLOCK_SIZE,
                })
                current += OFFSET
        return entries

    @staticmethod
    def _preload_chrom_lengths(fasta_path):
        """预加载所有染色体长度信息"""
        fasta = FastaFile(fasta_path)
        chrom_lengths = {chrom: length for chrom, length in zip(fasta.references, fasta.lengths)}
        fasta.close()
        return chrom_lengths

    def _get_hic_feature(self, chrom):
        """获取样本的HiC特征提取器（带缓存）"""
        cache_key = f"{chrom}"
        if cache_key not in self.hic_features:
            hic_path = os.path.join(self.hic_dir, f"{chrom}.npz")
            if not os.path.exists(hic_path):
                raise FileNotFoundError(f"HiC file not found: {hic_path}")
            self.hic_features[cache_key] = HiCFeature(path=hic_path)
        return self.hic_features[cache_key]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # 延迟初始化资源 在每个 worker 首次访问数据时，打开文件 每个 worker 访问自己的资源副本
        if self.fasta is None:
            self.fasta = FastaFile(self.fasta_path)

        if self.dna_feature is None:
            self.dna_feature = DNAFeature(self.fasta_path)

        if self.genomic_features:
            for bw_file, norm_method in self.bw_files.items():
                if bw_file not in self.feature_extractors:
                    feature_name = os.path.splitext(bw_file)[0]  # 去掉扩展名作为特征名
                    self.feature_extractors[feature_name] = GenomicFeature(
                        os.path.join(self.genomic_path, bw_file),
                        norm=norm_method
                    )

        pos_entry = self.entries[idx]
        chrom = pos_entry['chrom']
        block_start = pos_entry['start']
        block_end = pos_entry['end']

        if self.use_aug and self.mode == 'train':
            start = self.shift_aug(block_start, block_end)
        else:
            start = block_start  # 固定位置

        end = start + self.windows

        # 获取DNA序列
        dna = self.dna_feature.get(chrom, start, end)

        # 获取HiC矩阵
        hic_feature = self._get_hic_feature(chrom)
        hic_mat = hic_feature.get(start, window=self.windows, res=self.res)
        hic_mat = resize(hic_mat, (self.output, self.output), anti_aliasing=True)
        hic_mat = np.log(hic_mat + 1)

        if self.genomic_features:
            genomic_features_list = []
            for feature_name, extractor in self.feature_extractors.items():
                feature_data = extractor.get(chrom, start, end)
                genomic_features_list.append(feature_data)
            if self.use_aug and self.mode == 'train':
                dna = self.gaussian_noise(dna, 0.1)
                # Genomic features
                genomic_features_list = [self.gaussian_noise(item, 0.1) for item in genomic_features_list]
                # Reverse complement all data
                dna, genomic_features_list, hic_mat = self.reverse(dna, hic_mat, genomic_features_list)
            genomic_features_array = np.array(genomic_features_list).T
            combined_features = np.concatenate((dna, genomic_features_array), axis=1)
        else:
            if self.use_aug and self.mode == 'train':
                dna = self.gaussian_noise(dna, 0.1)
                dna, _, hic_mat = self.reverse(dna, hic_mat, None)
            combined_features = dna

        feature_tensor = torch.tensor(combined_features, dtype=torch.float32)
        hic_tensor = torch.tensor(hic_mat, dtype=torch.float32)
        return feature_tensor, hic_tensor

    def shift_aug(self, block_start, block_end):
        """在区块内随机位移窗口位置"""
        max_shift = block_end - block_start - self.windows
        if max_shift > 0:
            shift = np.random.randint(0, max_shift)
            return block_start + shift
        return block_start

    def gaussian_noise(self, inputs, std=1.0):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse(self, seq, mat, features=None, chance=0.5):
        '''
        Reverse sequence and matrix
        '''
        r_bool = np.random.rand(1)
        features_r = None
        if r_bool < chance:
            seq_r = np.flip(seq, 0).copy()  # n x 5 shape
            if features != None:
                features_r = [np.flip(item, 0).copy() for item in features]  # n
            mat_r = np.flip(mat, [0, 1]).copy()  # n x n

            # Complementary sequence
            seq_r = self.complement(seq_r)
        else:
            seq_r = seq
            if features != None:
                features_r = features
            mat_r = mat
        return seq_r, features_r, mat_r

    def complement(self, seq, chance=0.5):
        '''
        Complimentary sequence
        '''
        r_bool = np.random.rand(1)
        if r_bool < chance:
            seq_comp = np.concatenate([seq[:, 3:4],
                                       seq[:, 2:3],
                                       seq[:, 1:2],
                                       seq[:, 0:1],
                                       seq[:, 4:5]], axis=1)
        else:
            seq_comp = seq
        return seq_comp

    @staticmethod
    def _is_position_excluded_static(chrom, start, end, exclude_dict):
        if not exclude_dict or chrom not in exclude_dict:
            return False
        regions = exclude_dict[chrom]
        low, high = 0, len(regions) - 1
        while low <= high:
            mid = (low + high) // 2
            rs, re = regions[mid]
            if start < re and end > rs:  # 有重叠
                return True
            if end <= rs:
                high = mid - 1
            else:
                low = mid + 1
        return False
    @staticmethod
    def _load_exclude_regions_static(bed_path):
        """静态版本：不依赖 self，只返回 dict"""
        exclude = {}
        with open(bed_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                c, s, e = line.split()[:3]
                exclude.setdefault(c, []).append((int(s), int(e)))
        for c in exclude:
            exclude[c].sort(key=lambda x: x[0])
        return exclude
    @staticmethod
    def _preload_hic_bins_static(hic_dir):
        """静态版：仅返回 chrom->max_bin 字典"""
        chrom_bins = {}
        for fname in os.listdir(hic_dir):
            if not fname.endswith('.npz'):
                continue
            chrom = fname.replace('.npz', '')  # chr1.npz -> chr1
            path = os.path.join(hic_dir, fname)
            try:
                with np.load(path) as f:
                    key = 'hic' if 'hic' in f else f.files[0]
                    chrom_bins[chrom] = f[key].shape[0]
            except Exception as e:
                print(f'[warn] skip {path}: {e}')
        return chrom_bins
    @staticmethod
    def _hic_bin_safe(chrom, start_bp, end_bp, res, chrom_bins):
        """返回 True 表示该 bp 区间在 Hi-C 范围内"""
        if chrom not in chrom_bins:
            return False
        max_bp = chrom_bins[chrom] * res
        return end_bp <= max_bp
    def close(self):
        """关闭所有打开的文件句柄"""
        if self.fasta is not None:
            self.fasta.close()
            self.fasta = None

        if self.dna_feature is not None:
            self.dna_feature.close()
            self.dna_feature = None

        for extractor in self.feature_extractors.values():
            extractor.close()

        for feature in self.hic_features.values():
            if hasattr(feature, 'close'):
                feature.close()
        self.hic_features = {}


def collate_fn(batch):
    """批处理样本并跳过错误"""
    dna_batch = []
    hic_batch = []

    for item in batch:
        if item is None:
            continue
        dna, hic = item
        dna_batch.append(dna)
        hic_batch.append(hic)

    if len(dna_batch) == 0:
        return None, None

    dna_tensors = torch.stack(dna_batch)
    hic_tensors = torch.stack(hic_batch)
    return dna_tensors, hic_tensors