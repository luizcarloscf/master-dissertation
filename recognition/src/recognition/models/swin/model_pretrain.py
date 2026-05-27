import random
from typing import Literal, Union, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .model_finetune import SwinTransformer


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: float):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        if channels % self.upscale_factor != 0:
            raise ValueError(f"The channels ({channels}) must be multiples of the upscale factor ({self.upscale_factor})")

        new_channels = channels // self.upscale_factor
        x = x.view(batch_size, new_channels, self.upscale_factor, height, width)
        x = x.permute(0, 1, 3, 2, 4)  # (batch, new_channels, height, upscale_factor, width)
        x = x.reshape(batch_size, new_channels, height * self.upscale_factor, width)
        return x


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, encoder_stride: int = 16, **kwargs):
        super().__init__(**kwargs)

        self.encoder_stride = encoder_stride
        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=encoder_stride * 3, kernel_size=1),
            PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.dim_in

    def forward_encoder(self, x, mask):

        x = self.joints_embed(x)
        NM, TP, VP, _ = x.shape

        assert mask is not None

        mask_tokens = self.mask_token.expand(NM, TP, VP, -1)
        w = mask.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1.0 - w) + mask_tokens * w

        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]

        for _, blk in enumerate(self.blocks):
            # print("x", x.shape)
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, M, T, V, C = x.shape
        x = x.contiguous().view(N * M, T, V, C)
        # print("mask", mask.shape)

        if isinstance(mask, list):
            mask = torch.cat((mask[0], mask[1]), dim=0)

        z = self.forward_encoder(x, mask)
        z = z.permute(0, 3, 1, 2).contiguous()

        # print("z", z.shape)

        x_rec = self.decoder(z)
        # print("x_rec", x_rec.shape)

        x_rec = x_rec.permute(0, 2, 3, 1).contiguous()
        # print("mask", mask.shape)

        mask = mask.repeat_interleave(self.t_patch_size, 1).unsqueeze(-1).contiguous()

        # print("mask", mask.shape)

        loss_recon = F.l1_loss(x, x_rec, reduction="none")
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss
