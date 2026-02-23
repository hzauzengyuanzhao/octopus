import numpy as np
from pysam import FastaFile, VariantFile
import pyBigWig as pbw
import os

class Feature:
    """Feature base class, defines a common interface"""

    def load(self, **kwargs):
        """Load resources, subclass must implement"""
        raise NotImplementedError('load method not implemented')

    def get(self, *args, **kwargs):
        """To obtain data, the subclass must implement it"""
        raise NotImplementedError('get method not implemented')

    def __len__(self):
        """Returns the number of resources, subclasses must implement"""
        raise NotImplementedError('__len__ method not implemented')

    def close(self):
        """Release resources, optional implementation for subclasses"""
        pass

    def __enter__(self):
        """Supports context managers"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Automatically close resources when exiting the context"""
        self.close()


class DNAFeature(Feature):
    """DNA Sequence Feature Processor"""

    def __init__(self, path):
        """
        Initialize DNA sequence processor

        Args:
            path (str): FASTA file path
        """
        self.path = path
        self.fasta = None
        self.chrom_lengths = {}
        self.chroms = []

    def _load(self):
        """Load FASTA file and validate"""
        if self.fasta is not None:
            return

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"FASTA file not found: {self.path}")

        try:
            self.fasta = FastaFile(self.path)
            self.chrom_lengths = {k: v for k, v in zip(self.fasta.references, self.fasta.lengths)}
            self.chroms = list(self.fasta.references)
        except Exception as e:
            raise IOError(f"Failed to load FASTA file: {self.path}\nError: {str(e)}")

    def get(self, chrom, start, end, **kwargs):
        """
        Obtain the DNA sequence of a specified region (one-hot encoding)

        Args:
            chrom (str): Chromosome Name
            start (int): Starting position
            end (int): End position

        Returns:
            np.ndarray: one-hot encoded sequence (L, 5)
        """
        # Lazy loading
        if self.fasta is None:
            self._load()

        seq = self.get_seq(chrom, start, end)
        return self.onehot_encode(seq)

    def get_seq(self, chrom, start, end):
        """
        Obtain the DNA sequence of a specified region (integer encoding)

        Args:
            chrom (str): Chromosome Name
            start (int): Starting position
            end (int): End position

        Returns:
            np.ndarray: Integer encoded sequence (L,)
        """
        # Lazy loading
        if self.fasta is None:
            self._load()

        # Verify coordinate validity
        self._validate_coordinates(chrom, start, end)

        # Obtain and encode the sequence
        seq = self.fasta.fetch(chrom, start, end).upper()
        en_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        return np.array([en_dict.get(ch, 4) for ch in seq], dtype=np.int8)

    def _validate_coordinates(self, chrom, start, end):
        """Validate genome coordinate validity"""
        if chrom not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chrom} not found in FASTA")

        chrom_length = self.chrom_lengths[chrom]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def read_all_chrom(self):
        """Get a list of all chromosome names"""
        if self.fasta is None:
            self._load()
        return self.chroms.copy()

    @staticmethod
    def onehot_encode(seq):
        """
        Encode an integer sequence into a one-hot matrix

        Args:
            seq (np.ndarray): Integer encoded sequence

        Returns:
            np.ndarray: one-hot encoded sequence (L, 5)
        """
        seq_emb = np.zeros((len(seq), 5), dtype=np.float32)
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

    def __len__(self):
        """Return to chromosome count"""
        if self.fasta is None:
            self._load()
        return len(self.chroms)

    def close(self):
        """Safely close file resources"""
        if self.fasta is not None:
            self.fasta.close()
            self.fasta = None
            self.chrom_lengths = {}
            self.chroms = []

    def __repr__(self):
        return f"DNAFeature(path='{self.path}', chroms={len(self.chroms)})"


class GenomicFeature(Feature):
    """Genome Feature Processor (supports bigWig files)"""

    def __init__(self, path, norm=None):
        """
        Initialize Genome Feature Processor

        Args:
            path (str): bigWig file path
            norm (str, optional): Normalization Method ('log' or None)
        """
        self.path = path
        self.norm = norm
        self.bw_file = None
        self.chrom_lengths = {}
        self.chroms = []

    def _load(self):
        """Load bigWig file and validate"""
        if self.bw_file is not None:
            return

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"BigWig file not found: {self.path}")

        try:
            self.bw_file = pbw.open(self.path)
            self.chroms = list(self.bw_file.chroms().keys())
            self.chrom_lengths = {chrom: self.bw_file.chroms(chrom) for chrom in self.chroms}
        except Exception as e:
            raise IOError(f"Failed to load bigWig file: {self.path}\nError: {str(e)}")

    def get(self, chr_name, start, end):
        """
        Obtain genomic feature values for a specified region

        Args:
            chr_name (str): Chromosome Name
            start (int): Starting position
            end (int): End position

        Returns:
            np.ndarray: eigenvalue array (L,)
        """

        if self.bw_file is None:
            self._load()

        # Verify coordinate validity
        self._validate_coordinates(chr_name, start, end)

        # Read signal value
        signals = self.bw_file.values(chr_name, start, end)
        feature = np.array(signals, dtype=np.float32)

        # Handle missing values
        feature = np.nan_to_num(feature, nan=0.0)

        # Apply normalization
        return self._apply_normalization(feature)

    def _apply_normalization(self, data):
        """Apply the specified normalization method"""
        if self.norm == 'log':
            data = np.log(data + 1)
            return data
        elif self.norm is None or self.norm == '':
            return data
        else:
            raise ValueError(f'Unsupported normalization type: {self.norm}')

    def _validate_coordinates(self, chr_name, start, end):
        """Validate genome coordinate validity"""
        if chr_name not in self.chrom_lengths:
            raise ValueError(f"Chromosome {chr_name} not found in bigWig")

        chrom_length = self.chrom_lengths[chr_name]
        if start < 0 or end > chrom_length:
            raise IndexError(f"Coordinates {start}-{end} out of range (0-{chrom_length})")
        if start >= end:
            raise ValueError(f"Start ({start}) must be less than end ({end})")

    def length(self, chr_name):
        """Get the length of a specified chromosome"""
        if self.bw_file is None:
            self._load()
        return self.chrom_lengths.get(chr_name, 0)

    def __len__(self):
        """Return to chromosome count"""
        if self.bw_file is None:
            self._load()
        return len(self.chroms)

    def close(self):
        """Safely close file resources"""
        if self.bw_file is not None:
            self.bw_file.close()
            self.bw_file = None
            self.chrom_lengths = {}
            self.chroms = []

    def __repr__(self):
        return f"GenomicFeature(path='{self.path}', norm='{self.norm}', chroms={len(self.chroms)})"


class HiCFeature(Feature):
    """Hi-C Contact Matrix Processor"""

    def __init__(self, path):
        """
        Initialize Hi-C processor

        Args:
            path (str): NPZ file path
        """
        self.path = path
        self.hic = None

    def _load(self):
        """Load Hi-C data file and validate"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Hi-C file not found: {self.path}")
        try:
            #print(f'Loading Hi-C data: {self.path}')
            self.hic = dict(np.load(self.path))

            if '0' not in self.hic:
                raise ValueError("Invalid Hi-C format: missing '0' diagonal")
        except Exception as e:
            raise IOError(f"Failed to load Hi-C file: {self.path}\nError: {str(e)}")

    def get(self, start, window=2000000, res=10000):
        """
        Obtain the Hi-C contact matrix for a specified region

        Args:
            start (int): Starting position
            window (int): Window size (default 2Mb)
            res (int): Resolution (default 10kb)

        Returns:
            np.ndarray: Hi-C contact matrix (bins, bins)
        """
        if self.hic is None:
            self._load()
        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin

        # Validate boundary
        max_bin = len(self.hic['0'])
        if end_bin > max_bin:
            raise IndexError(f"Requested bins {start_bin}-{end_bin} exceed max {max_bin}")

        return self._diag_to_mat(start_bin, end_bin)

    def _diag_to_mat(self, start, end):
        """
        Reconstructing the contact matrix from diagonal data

        Args:
            start (int): start bin
            end (int): end bin

        Returns:
            np.ndarray: Contact matrix (L, L)
        """
        square_len = end - start
        diag_load = {}

        for diag_i in range(square_len):
            diag_key = str(diag_i)
            if diag_key in self.hic:
                data = self.hic[diag_key][start: start + square_len - diag_i]
                diag_load[diag_key] = data

            neg_key = str(-diag_i)
            if neg_key in self.hic:
                data = self.hic[neg_key][start: start + square_len - diag_i]
                diag_load[neg_key] = data

        matrix = np.zeros((square_len, square_len), dtype=np.float32)
        for i in range(square_len):
            for j in range(square_len):
                diag_index = j - i
                diag_key = str(diag_index)

                if diag_key in diag_load:
                    pos = min(i, j) if diag_index >= 0 else min(i - diag_index, j)
                    if 0 <= pos < len(diag_load[diag_key]):
                        matrix[i, j] = diag_load[diag_key][pos]

        return matrix

    def __len__(self):
        """Return main diagonal length"""
        if self.hic is None:
            self._load()
        return len(self.hic['0']) if '0' in self.hic else 0

    def close(self):
        """Hi-C data is in memory and does not need to be explicitly closed."""
        pass

    def __repr__(self):
        return f"HiCFeature(path='{self.path}', bins={len(self)})"


class VCFFeature(DNAFeature):
    """VCF sequence feature processor, based on DNAFeature and supports variant information"""

    def __init__(self, fasta_path, vcf_path):
        """
        Initialize VCF sequence feature processor

        Args:
            fasta_path (str): FASTA reference sequence path
            vcf_path (str): VCF file path
        """
        super().__init__(fasta_path)
        self.vcf_path = vcf_path
        self.vcf = None

    def _load(self):
        """Load FASTA and VCF resources"""
        if self.fasta is not None and self.vcf is not None:
            return

        # Call the parent class to load FASTA
        super()._load()

        if not os.path.exists(self.vcf_path):
            raise FileNotFoundError(f"VCF file not found: {self.vcf_path}")

        try:
            self.vcf = VariantFile(self.vcf_path)
        except Exception as e:
            raise IOError(f"Failed to load VCF file: {self.vcf_path}\nError: {str(e)}")

    def get(self, chrom, start, end, sample=None, haplotype=0, **kwargs):
        """
        Obtain one-hot encoded sequences with VCF mutations

        Args:
            chrom (str): Chromosome Name
            start (int): Starting position（0-based）
            end (int): End position（0-based）
            sample (str): sample name
            haplotype (int): Haplotype Index (0 or 1)

        Returns:
            np.ndarray: one-hot编码序列 (L, 5)
        """
        if self.fasta is None or self.vcf is None:
            self._load()

        ref_seq = self.get_seq(chrom, start, end)
        vcf_seq = self._apply_variants(ref_seq, chrom, start, end, sample, haplotype)
        return self.onehot_encode(vcf_seq)

    def _apply_variants(self, ref_seq, chrom, start, end, sample, haplotype):
        """
        Apply VCF variants to the reference sequence

        Args:
            ref_seq (np.ndarray): Integer-encoded reference sequence
            chrom (str):  Chromosome
            start (int):   Starting position
            end (int): End position
            sample (str): Sample name
            haplotype (int): Haplotype Index

        Returns:
            np.ndarray: Integer encoding sequence after applying mutation
        """
        vcf_seq = ref_seq.copy()

        for record in self.vcf.fetch(chrom, start, end):
            if not self._is_snp(record):
                continue

            pos_in_seq = record.pos - start - 1
            if not (0 <= pos_in_seq < len(ref_seq)):
                continue

            if sample not in record.samples:
                continue

            sample_data = record.samples[sample]
            gt = sample_data.get("GT")
            if gt is None or len(gt) <= haplotype or gt[haplotype] is None:
                continue

            allele_idx = gt[haplotype]
            if allele_idx == 0:
                continue

            try:
                allele = record.alleles[allele_idx]
                if len(allele) != 1:
                    continue
                en_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
                new_base = en_dict.get(allele.upper(), 4)
                vcf_seq[pos_in_seq] = new_base
            except IndexError:
                continue

        return vcf_seq

    @staticmethod
    def _is_snp(record):
        """Determine whether it is an SNP variation"""
        return len(record.ref) == 1 and all(len(alt) == 1 for alt in record.alts)

    def __len__(self):
        """Return the number of chromosomes"""
        if self.fasta is None:
            self._load()
        return len(self.chroms)

    def close(self):
        """Close all resources"""
        if self.vcf is not None:
            self.vcf.close()
            self.vcf = None
        super().close()

    def __repr__(self):
        return f"VCFFeature(fasta='{self.path}', vcf='{self.vcf_path}', chroms={len(self)})"
