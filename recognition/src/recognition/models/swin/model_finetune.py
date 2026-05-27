import math
from typing import List, Tuple, Literal, Optional
from functools import reduce, lru_cache
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_

from .heads import ActionHeadFinetune, ActionHeadLinprobe


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], B: int, H: int, W: int):
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_window_size(x_size, window_size: Tuple[int, int], shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (2, 5),
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x, mask_matrix):
        B, H, W, C = x.shape

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        NM, T, V, C = x.shape

        x0 = x[:, 0::2, :, :]  # B H/2 W C
        x1 = x[:, 1::2, :, :]  # B H/2 W C
        x = torch.cat([x0, x1], -1)  # B H/2 W 2*C

        x = x.view(NM, -1, 2 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(NM, T // 2, V, 2 * C)
        return x


# cache each stage results
@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for h in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for w in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[PatchMerging] = None,
        use_checkpoint: bool = False,
    ):

        super().__init__()
        self.window_size = window_size
        # self.shift_size = tuple(i // 2 for i in window_size)
        self.shift_size = (window_size[0] // 2, 0)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ],
        )

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)
        Hp = int(math.ceil(H / window_size[0])) * window_size[0]
        Wp = int(math.ceil(W / window_size[1])) * window_size[1]

        attn_mask = compute_mask(Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SkeleEmbed(nn.Module):

    def __init__(
        self,
        dim_in: int = 3,
        dim_feat: int = 256,
        num_frames: int = 120,
        num_joints: int = 25,
        patch_size: int = 1,
        t_patch_size: int = 4,
    ):
        super().__init__()
        assert num_frames % t_patch_size == 0
        num_patches = (num_joints // patch_size) * (num_frames // t_patch_size)
        self.input_size = (num_frames // t_patch_size, num_joints // patch_size)
        print(f"num_joints {num_joints} patch_size {patch_size} num_frames {num_frames} t_patch_size {t_patch_size}")

        self.num_joints = num_joints
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = num_joints // patch_size
        self.t_grid_size = num_frames // t_patch_size

        kernel_size = [t_patch_size, patch_size]
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, V, _ = x.shape
        assert V == self.num_joints, f"Input skeleton size ({V}) doesn't match model ({self.num_joints})."
        assert T == self.num_frames, f"Input skeleton length ({T}) doesn't match model ({self.num_frames})."

        x = torch.einsum("ntsc->ncts", x)
        x = self.proj(x)
        x = torch.einsum("ncts->ntsc", x)
        return x


class SwinTransformer(nn.Module):

    def __init__(
        self,
        dim_in: int = 3,
        num_classes: int = 60,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 4, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int] = (4, 25),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        num_frames: int = 128,
        num_joints: int = 25,
        patch_size: int = 1,
        t_patch_size: int = 4,
        protocol: Literal["finetune", "linprobe"] = "finetune",
    ):
        super().__init__()

        self.dim_in = dim_in
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.joints_embed = SkeleEmbed(
            dim_in=dim_in,
            dim_feat=self.embed_dim,
            num_frames=num_frames,
            num_joints=num_joints,
            patch_size=patch_size,
            t_patch_size=t_patch_size,
        )
        num_patches = self.joints_embed.num_patches
        self.patches_resolution = self.joints_embed.input_size

        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames // t_patch_size, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_joints // patch_size, embed_dim))
        trunc_normal_(self.temp_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.blocks.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)

        if num_classes > 0:
            if protocol == "linprobe":
                self.head = ActionHeadLinprobe(
                    dim_feat=self.num_features,
                    num_classes=num_classes,
                    num_joints=num_joints,
                )
            elif protocol == "finetune":
                self.head = ActionHeadFinetune(
                    dropout_ratio=0.0,
                    dim_feat=self.num_features,
                    num_classes=num_classes,
                    num_joints=num_joints,
                )
            else:
                raise TypeError("Unrecognized evaluation protocol!")
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temp_embed", "pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, M, T, V, C = x.shape
        x = x.contiguous().view(N * M, T, V, C)

        x = self.joints_embed(x)
        NM, TP, VP, _ = x.shape
        x = x + self.pos_embed[:, :, :VP, :] + self.temp_embed[:, :TP, :, :]
        x = self.pos_drop(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(N, M, -1, VP, self.num_features)
        x = self.head(x)
        return x
