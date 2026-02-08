import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS

class AttentionBlock(nn.Module):
    """Channel + Spatial Attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Channel attention
        w = self.avg_pool(x)
        w = self.fc(w)
        x = x * w

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        s = torch.cat([avg, mx], dim=1)
        s = self.spatial(s)

        x = x * s

        return x
    
class ScaleReweight(nn.Module):
    """Multi-scale feature reweighting"""

    def __init__(self, channels, num_levels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * num_levels,
                      num_levels),
            nn.Sigmoid()
        )

    def forward(self, feats):

        # feats: list of [B,C,H,W]
        stats = []

        for f in feats:
            s = self.pool(f).flatten(1)
            stats.append(s)

        stats = torch.cat(stats, dim=1)

        w = self.fc(stats)

        return w


@NECKS.register_module()
class FGFPN(BaseModule):
    """
    Fine-Grained Feature Pyramid Network (HTDet)

    Simplified + HLS-friendly implementation
    """

    def __init__(self,
             in_channels,
             out_channels,
             num_outs=5,
             norm_cfg=dict(type='BN'),
             init_cfg=None):

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Lateral convs
        self.lateral_convs = nn.ModuleList()

        # Output convs
        self.fpn_convs = nn.ModuleList()
        # Attention refinement
        self.attentions = nn.ModuleList()
        # Scale reweighting
        self.scale_weight = ScaleReweight(
            out_channels,
            len(in_channels)
        )


        for _ in range(len(in_channels)):
            self.attentions.append(
                AttentionBlock(out_channels)
            )


        # Top-down fusion weights
        self.fusion_weights = nn.ParameterList()

    # Feature balance weights (FBM)
        self.balance_weights = nn.Parameter(
            torch.ones(len(in_channels))
        )

        for i in range(len(in_channels)):

            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg
            )

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            self.fusion_weights.append(
                nn.Parameter(torch.ones(2))
            )


    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # Lateral features
        laterals = [
            l_conv(x)
            for l_conv, x in zip(self.lateral_convs, inputs)
        ]

        # -----------------------------
        # Top-down fusion
        # -----------------------------
        for i in range(len(laterals) - 1, 0, -1):

            w = F.relu(self.fusion_weights[i])
            weight = w / (w.sum() + 1e-4)

            up = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )

            laterals[i - 1] = (
                weight[0] * laterals[i - 1] +
                weight[1] * up
            )

        # -----------------------------
        # Feature Balance Module (FBM)
        # -----------------------------
        bw = F.relu(self.balance_weights)
        bw = bw / (bw.sum() + 1e-4)

        balanced = []

        for i in range(len(laterals)):

            feat = bw[i] * laterals[i]

            # Attention refinement
            feat = self.attentions[i](feat)

            balanced.append(feat)

        # -----------------------------
        # Output
        # -----------------------------
        # -----------------------------
# Multi-scale reweighting
# -----------------------------
        sw = self.scale_weight(balanced)   # [B, L]

        reweighted = []

        for i in range(len(balanced)):
            w = sw[:, i].view(-1, 1, 1, 1)
            reweighted.append(w * balanced[i])

        # -----------------------------
        # Output
        # -----------------------------
        outs = [
            conv(feat)
            for conv, feat in zip(self.fpn_convs, reweighted)
        ]


        return tuple(outs)
