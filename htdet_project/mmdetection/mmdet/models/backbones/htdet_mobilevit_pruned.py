import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


# ----------------------------
# Basic Conv Block
# ----------------------------
class ConvBNAct(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ----------------------------
# MobileNetV2 Block (Prunable)
# ----------------------------
class MBConv(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride,
                 expand_ratio=6):

        super().__init__()

        hidden_dim = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []

        if expand_ratio != 1:
            layers.append(
                ConvBNAct(in_ch, hidden_dim, kernel_size=1, padding=0)
            )

        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim
            )
        )

        layers.append(
            ConvBNAct(
                hidden_dim,
                out_ch,
                kernel_size=1,
                padding=0,
                act=False
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


# ----------------------------
# Transformer Block (Prunable)
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=2.0):

        super().__init__()

        # Ensure dim is divisible by num_heads for multi-head attention
        # Adjust num_heads if needed
        actual_num_heads = num_heads
        while dim % actual_num_heads != 0 and actual_num_heads > 1:
            actual_num_heads -= 1

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, actual_num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        # Ensure hidden_dim is reasonable
        hidden_dim = max(hidden_dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h

        return x


# ----------------------------
# MobileViT Block (Prunable)
# ----------------------------
class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, transformer_dim, depth, patch_size=2):
        super().__init__()

        self.patch_size = patch_size

        # Local representation - keeps same channels
        self.conv1 = ConvBNAct(in_ch, in_ch, 3)
        self.conv2 = ConvBNAct(in_ch, transformer_dim, 1, padding=0)

        # Linear embedding
        self.token_proj = nn.Linear(transformer_dim * patch_size * patch_size,
                                    transformer_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(transformer_dim)
            for _ in range(depth)
        ])

        # Reverse projection
        self.token_unproj = nn.Linear(transformer_dim,
                                      transformer_dim * patch_size * patch_size)

        # Projection back
        self.conv3 = ConvBNAct(transformer_dim, in_ch, 1, padding=0)

        # Fusion
        self.conv4 = ConvBNAct(in_ch * 2, in_ch, 3)

    def forward(self, x):
        res = x

        # Local conv
        x = self.conv1(x)
        x = self.conv2(x)

        B, C, H, W = x.shape
        ph = self.patch_size
        pw = self.patch_size

        # Padding if needed
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        B, C, H_pad, W_pad = x.shape

        # Unfold to patches
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x = x.contiguous().view(B, C, -1, ph * pw)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, C * ph * pw)

        # Linear projection
        x = self.token_proj(x)

        # Transformer
        x = self.transformer(x)

        # Reverse projection
        x = self.token_unproj(x)

        # Restore patches
        x = x.view(B, -1, C, ph * pw)
        x = x.permute(0, 2, 1, 3).contiguous()

        h = H_pad // ph
        w = W_pad // pw

        x = x.view(B, C, h, w, ph, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H_pad, W_pad)

        # Remove padding
        x = x[:, :, :H, :W]

        # Projection
        x = self.conv3(x)

        # Fusion
        x = torch.cat([x, res], dim=1)
        x = self.conv4(x)

        return x


# ----------------------------
# HTDet MobileViT Backbone (Pruned)
# ----------------------------
@BACKBONES.register_module()
class HTDetMobileViTPruned(BaseModule):
    """
    MobileViT Backbone with channel pruning via width multiplier.

    width_mult controls channel reduction:
    - 1.0 = full model (same as original)
    - 0.7 = 30% channel reduction
    - 0.5 = 50% channel reduction

    All channel dimensions scale proportionally.
    """

    def __init__(self,
                 in_channels=3,
                 width=16,
                 width_mult=1.0,
                 init_cfg=None):

        super().__init__(init_cfg)

        # Scale all dimensions by width_mult
        pruned_width = max(int(width * width_mult), 8)

        # Transformer dimensions also scale
        transformer_dim_s = max(int(96 * width_mult), 32)   # Stage 3
        transformer_dim_m = max(int(128 * width_mult), 48)  # Stage 4
        transformer_dim_l = max(int(160 * width_mult), 64)  # Stage 5

        # Stem
        self.stem = ConvBNAct(in_channels, pruned_width, 3, stride=2)

        # Stage 1 (1/2)
        self.stage1 = MBConv(pruned_width, pruned_width, 1)

        # Stage 2 (1/4)
        self.stage2 = nn.Sequential(
            MBConv(pruned_width, pruned_width * 2, 2),
            MBConv(pruned_width * 2, pruned_width * 2, 1)
        )

        # Stage 3 (1/8) + MobileViT
        self.stage3 = nn.Sequential(
            MBConv(pruned_width * 2, pruned_width * 4, 2),
            MobileViTBlock(pruned_width * 4, transformer_dim_s, depth=2)
        )

        # Stage 4 (1/16) + MobileViT
        self.stage4 = nn.Sequential(
            MBConv(pruned_width * 4, pruned_width * 8, 2),
            MobileViTBlock(pruned_width * 8, transformer_dim_m, depth=4)
        )

        # Stage 5 (1/32) + MobileViT
        self.stage5 = nn.Sequential(
            MBConv(pruned_width * 8, pruned_width * 16, 2),
            MobileViTBlock(pruned_width * 16, transformer_dim_l, depth=3)
        )

        self.out_channels = [
            pruned_width,
            pruned_width * 2,
            pruned_width * 4,
            pruned_width * 8,
            pruned_width * 16
        ]

    def forward(self, x):
        outs = []

        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)
        outs.append(x)

        # Stage 2
        x = self.stage2(x)
        outs.append(x)

        # Stage 3
        x = self.stage3(x)
        outs.append(x)

        # Stage 4
        x = self.stage4(x)
        outs.append(x)

        # Stage 5
        x = self.stage5(x)
        outs.append(x)

        return tuple(outs)
