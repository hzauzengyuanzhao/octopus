import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Inception1D(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(Inception1D, self).__init__()
        branch_channels = out_channels // 4

        # Branch 1:  1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 3: 1x1 conv -> 3x3 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 4: MaxPool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        self.residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat([x1,x2,x3,x4], dim=1)

        res = self.residual(x)
        out = F.leaky_relu(out + res)
        out = self.maxpool(out)
        return out

class InceptionEncoder(nn.Module):
    def __init__(self, num_genomic_features, output_dim, base_channels=32,  num_layers = 12, num_bins=256):
        super(InceptionEncoder, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(5 + num_genomic_features, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList()
        current_channels = base_channels

        # Channel Multiplication Plan: Double the number of channels every 2 levels
        channel_multipliers = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32][:num_layers]

        for i, mult in enumerate(channel_multipliers):
            out_channels = min(output_dim, base_channels * mult)

            # Add Inception block
            self.layers.append(
                Inception1D(current_channels, out_channels)
            )
            current_channels = out_channels

    def forward(self, x):
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class MoeModule(nn.Module):
    def __init__(self, hidden_dim, seq_len=256):
        super().__init__()
        # Unimodal Adapters
        self.ua_seq = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.ua_epi = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Cross-Modal Adapter
        self.cma = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.weight = nn.Parameter(torch.rand(3,2688))
        self.conv_last = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim//2, 3, 2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        self.fla = nn.Flatten()

        self.soft = nn.Softmax(dim=1)

    def forward(self, seq, epi, cross):
        # Concatenated features as router input
        concat_feat = cross  # [batch, 2*hidden_dim, seq_len]
        crosses = self.conv_last(cross)
        ver = self.fla(crosses)
        # Expand dimensions for easier broadcasting
        dist = -torch.norm(self.weight.unsqueeze(0) - ver.unsqueeze(1),
                          dim = 2, keepdim=False)  # shape:[B, 3]

        weights = self.soft(dist)

        ua_seq_out = self.ua_seq(seq)
        ua_epi_out = self.ua_epi(epi)
        cma_out = self.cma(concat_feat)

        w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]

        w1 = w1.view(-1, 1, 1)
        w2 = w2.view(-1, 1, 1)
        w3 = w3.view(-1, 1, 1)

        # Weighted Fusion of Expert Outputs
        fused = w1 * ua_seq_out + w2 * ua_epi_out + w3 * cma_out
        return fused, weights

class EncoderSplit(nn.Module):
    def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=6):
        super().__init__()
        self.num_epi = num_epi
        self.filter_size = filter_size

        hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)

        # DNA Encoder
        self.conv_start_seq = nn.Sequential(
            nn.Conv1d(5, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)

        # Epi Encoder
        if self.num_epi > 0:
            self.conv_start_epi = nn.Sequential(
                nn.Conv1d(num_epi, 16, 3, 2, 1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            )
            self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)

        self.Inception = InceptionEncoder(num_epi, output_size//2)

    def forward(self, x):
        seq = x[:, :5, :]
        seq = self.conv_start_seq(seq)
        seq = self.res_blocks_seq(seq)

        if self.num_epi > 0:
            epi = x[:, 5:, :]
            epi = self.conv_start_epi(epi)
            epi = self.res_blocks_epi(epi)
            cross = self.Inception(x)
            return seq, epi, cross

        return seq

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(Inception1D(in_channels = hi, out_channels = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
