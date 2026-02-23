import torch
import torch.nn as nn
from model.blocks import Decoder, EncoderSplit, MoeModule

class Octopus(nn.Module):
    def __init__(self, num_genomic_features, mid_hidden=256, record_attn=False):
        super(Octopus, self).__init__()
        self.num_genomic_features = num_genomic_features
        self.encoder = EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        if num_genomic_features > 0:
            self.moe = MoeModule(mid_hidden//2)
        self.decoder = Decoder(mid_hidden, hidden=256, filter_size=3, num_blocks=5)
        self.record_attn = record_attn
        self.mid_hidden = mid_hidden

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()

        weights = None
        if self.num_genomic_features > 0:
            # Obtain separated features
            seq_feat, epi_feat, cross = self.encoder(x)
            # Dynamically fuse features through the MoE module
            feat, weights = self.moe(seq_feat, epi_feat, cross)
            x = feat
        else:
            seq_feat = self.encoder(x)
            x = seq_feat
        x = self.diagonalize(x)
        # Decoder
        x = self.decoder(x).squeeze(1)
        return x, weights

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
    input = torch.rand(1, 2097152, 7)
    print(f'input.shape:{input.shape}')
    model = Octopus(2)
    hooks = register_hooks(model)
    output, w = model(input)
    print(f'out.shape:{output.shape}')
    w1, w2, w3 = w[:, 0], w[:, 1], w[:, 2]
    print(f'Moe dna weight:{w1}')
    print(f'Moe epi weight:{w1}')
    print(f'Moe cross weight:{w1}')
