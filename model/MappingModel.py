import torch
import torch.nn as nn
import numpy as np
from blocks import Decoder, Inception1D, InceptionEncoder, MoeModule

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
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)

        self.conv_start_seq = nn.Sequential(
            nn.Conv1d(5, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        # Epi Encoder
        self.epi_from_dna = nn.Sequential(
            nn.Conv1d(5, 16, 3, 2, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            *self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        )

        self.Inception = InceptionEncoder(num_epi if num_epi > 0 else 0, output_size // 2)


    def forward(self, x):

        seq = self.conv_start_seq(x)
        dna = self.res_blocks_seq(seq)

        epi = self.epi_from_dna(x)
        cross = self.Inception(x)
        return dna, epi, cross

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(Inception1D(in_channels = hi, out_channels = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class MappingModel(nn.Module):
    def __init__(self, num_genomic_features, teacher_model=None, mid_hidden=256, record_attn=False):
        super(MappingModel, self).__init__()
        self.num_genomic_features = num_genomic_features
        self.encoder = EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        self.moe = MoeModule(mid_hidden//2)
        self.decoder = Decoder(mid_hidden, hidden=256, filter_size=3, num_blocks=5)
        self.record_attn = record_attn
        self.mid_hidden = mid_hidden
        self.teacher_model = teacher_model  # Pre-trained Model

    def forward(self, x):
        seq = x.transpose(1, 2).contiguous()

        x = seq[:, :5, :]
        dna, epi, cross = self.encoder(x)
        feat, weights = self.moe(dna, epi, cross)
        x = self.diagonalize(feat)
        x = self.decoder(x).squeeze(1)
        t_epi = None
        t_cross = None
        if self.teacher_model is not None:
            with torch.no_grad():
                t_dna, t_epi, t_cross = self.teacher_model.encoder(seq)
        return x, weights, epi, cross, t_epi, t_cross

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

def print_output_shape(module, input, output):
    if isinstance(output, tuple):
        print(f"Module: {module.__class__.__name__}, Output is a tuple with {len(output)} elements")
        for i, item in enumerate(output):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: Shape {item.shape}")
            else:
                print(f"  Element {i}: Not a tensor")
    else:
        print(f"Module: {module.__class__.__name__}, Output Shape: {output.shape}")


def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hook = module.register_forward_hook(print_output_shape)
            hooks.append(hook)
    return hooks


if __name__ == '__main__':
    input = torch.rand(1, 2097152, 5)
    print(f'input.shape:{input.shape}')
    model = MappingModel(0)
    hooks = register_hooks(model)
    output, w, epi, cross,_,_ = model(input)
    print(f'out.shape:{output.shape}')
    w1, w2, w3 = w[:, 0], w[:, 1], w[:, 2]
    print(f'Moe dna weight:{w1}')
    print(f'Moe epi weight:{w1}')
    print(f'Moe cross weight:{w1}')
