"""
Underwater Contextual Attention FPN (UCA-FPN)

Motivation: Underwater images suffer from three domain-specific degradations:
  1. Color distortion — water absorbs wavelengths selectively (red attenuated most)
  2. Non-uniform illumination / blur — scatters high-frequency detail
  3. Irregular object shapes — holothurian, starfish have high aspect ratio variance

UCA-FPN addresses these with a CBAM (Convolutional Block Attention Module) inserted
after each FPN output level:
  - Channel Attention: re-weights feature channels (corrects color bias)
  - Spatial Attention: highlights salient regions, suppresses blur/background

Both modules are < 0.1M params total and add < 2ms latency — FPGA-friendly.

Reference: Woo et al., CBAM: Convolutional Block Attention Module, ECCV 2018.
"""
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN


class ChannelAttention(nn.Module):
    """Squeeze channel context from both avg and max pool, then gate."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        gate = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * gate.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Pool across channels, then learn a spatial saliency map."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    """Channel → Spatial attention (sequential, as in original paper)."""

    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


@NECKS.register_module()
class UCAFPN(FPN):
    """Underwater Contextual Attention FPN.

    Identical to standard FPN but with a per-level CBAM gate applied to
    every output feature map (including extra levels for RetinaNet P6/P7).

    Args:
        cbam_reduction (int): Channel reduction ratio in CBAM. Default: 8.
        spatial_kernel (int): Kernel size for spatial attention conv. Default: 7.
        All other args forwarded to FPN.
    """

    def __init__(self, *args, cbam_reduction=8, spatial_kernel=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_modules = nn.ModuleList([
            CBAM(self.out_channels, cbam_reduction, spatial_kernel)
            for _ in range(self.num_outs)
        ])
        self._init_cbam()

    def _init_cbam(self):
        for m in self.cbam_modules.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @auto_fp16()
    def forward(self, inputs):
        outs = list(super().forward(inputs))
        for i, cbam in enumerate(self.cbam_modules):
            outs[i] = cbam(outs[i])
        return tuple(outs)
