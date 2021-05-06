import torch
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLP_Layer(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        # input size = (bs, num_patch, token_dim)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d-> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLP_Mixer(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, num_classes, dim, depth, token_dim, channel_dim, pool='mean'):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(MLP_Layer(
                dim, self.num_patch, token_dim, channel_dim))
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = MLP_Mixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                      dim=512, depth=8, token_dim=256, channel_dim=2048, pool='mean')
    out = model(x)
    print('Total params: %.3fM' % (sum(p.numel()
                                       for p in model.parameters())/1000000.0))
    print("out.shape = ", out.shape)
