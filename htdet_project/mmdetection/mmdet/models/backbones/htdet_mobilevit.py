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
# MobileNetV2 Block
# ----------------------------
class MBConv(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride,
                 expand_ratio=6):

        super().__init__()

        hidden_dim = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.append(
                ConvBNAct(in_ch, hidden_dim, kernel_size=1, padding=0)
            )

        # Depthwise
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

        # Project
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
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=2.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):

        # Self-attention
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + h

        # MLP
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h

        return x


# ----------------------------
# MobileViT Block
# ----------------------------
class MobileViTBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 transformer_dim,
                 depth):

        super().__init__()

        # Local conv
        self.conv1 = ConvBNAct(in_ch, in_ch, 3)
        self.conv2 = ConvBNAct(in_ch, transformer_dim, 1, padding=0)

        # Transformer
        self.transformer = nn.Sequential(*[
            TransformerBlock(transformer_dim)
            for _ in range(depth)
        ])

        # Fusion
        self.conv3 = ConvBNAct(transformer_dim, in_ch, 1, padding=0)
        self.conv4 = ConvBNAct(in_ch * 2, in_ch, 3)

    def forward(self, x):

        res = x

        # Local conv
        x = self.conv1(x)
        x = self.conv2(x)

        B, C, H, W = x.shape

        orig_H, orig_W = H, W

        # -----------------------------
        # Patch Partition (2x2)
        # -----------------------------
        ph = 2
        pw = 2

        # Pad if needed
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        B, C, H, W = x.shape

        # Unfold into patches
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # B, C, H/ph, W/pw, ph, pw

        x = x.contiguous().view(B, C, -1, ph * pw)
        # B, C, N, 4

        # Average patch
        x = x.mean(-1)
        # B, C, N

        # Tokens
        x = x.permute(0, 2, 1)
        # B, N, C

        # -----------------------------
        # Transformer
        # -----------------------------
        x = self.transformer(x)

        # -----------------------------
        # Restore Spatial
        # -----------------------------
        x = x.permute(0, 2, 1)
        # B, C, N

        h = H // ph
        w = W // pw

        x = x.view(B, C, h, w)

        # Upsample
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # Remove padding
        x = x[:, :, :orig_H, :orig_W]

        # Project
        x = self.conv3(x)

        # Fuse
        x = torch.cat([x, res], dim=1)
        x = self.conv4(x)

        return x


# ----------------------------
# HTDet MobileViT Backbone
# ----------------------------
@BACKBONES.register_module()
class HTDetMobileViT(BaseModule):
    """
    MobileViT Backbone for HTDet
    """

    def __init__(self,
                 in_channels=3,
                 width=64,
                 init_cfg=None):

        super().__init__(init_cfg)

        # Stem
        self.stem = ConvBNAct(in_channels, width, 3, stride=2)

        # Stage 1 (1/2)
        self.stage1 = MBConv(width, width, 1)

        # Stage 2 (1/4)
        self.stage2 = nn.Sequential(
            MBConv(width, width * 2, 2),
            MBConv(width * 2, width * 2, 1)
        )

        # Stage 3 (1/8) + MobileViT (L=2)
        self.stage3 = nn.Sequential(
            MBConv(width * 2, width * 4, 2),
            MobileViTBlock(width * 4, width * 4, depth=2)
        )

        # Stage 4 (1/16) + MobileViT (L=4)
        self.stage4 = nn.Sequential(
            MBConv(width * 4, width * 8, 2),
            MobileViTBlock(width * 8, width * 8, depth=4)
        )

        # Stage 5 (1/32) + MobileViT (L=3)
        self.stage5 = nn.Sequential(
            MBConv(width * 8, width * 16, 2),
            MobileViTBlock(width * 16, width * 16, depth=3)
        )

        self.out_channels = [
            width,
            width * 2,
            width * 4,
            width * 8,
            width * 16
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
