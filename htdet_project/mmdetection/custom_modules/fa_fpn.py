"""
Frequency-Aware FPN (FA-FPN) for Underwater Object Detection

Motivation (physically grounded):
  Underwater image degradation is frequency-selective:
    - Water absorbs red wavelengths (low-spatial-frequency color shift)
    - Suspended particles scatter high-spatial-frequency detail (blur)
    - Different FPN levels should trust different frequency bands:
        P2 (small objects) -> high-freq edges critical for echinus clusters
        P6 (large objects) -> low-freq shape info is sufficient

FA-FPN adds two lightweight modules after each FPN output conv:

  1. FreqChannelGate (FCG):
       Computes 2D rFFT of each feature map, extracts per-channel amplitude
       spectrum, and learns a gating function over channels. Channels carrying
       mostly-attenuated frequencies (e.g. blurred high-freq in turbid water)
       are suppressed automatically.

  2. FreqBandDecomp (FBD):
       Decomposes features into low-freq (avg-pool + upsample) and high-freq
       (residual) components. Learns per-channel mixing weights so each FPN
       level can decide how much edge detail vs. smooth structure to keep.

Together these form an FAModule: FCG -> FBD

Parameter overhead per level: ~0.1M  (negligible, FPGA-friendly)
No changes to existing mmdet files.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN


class FreqChannelGate(nn.Module):
    """
    FFT-based channel attention.

    For each feature map x (B, C, H, W):
      1. rfft2  -> complex spectrum (B, C, H, W//2+1)
      2. |amp|.mean(spatial) -> (B, C) frequency energy per channel
      3. FC squeeze-excite -> per-channel gate in [0, 1]
      4. x * gate

    Intuition: channels dominated by attenuated frequencies (blurred high-freq
    or color-shifted low-freq) get down-weighted automatically during training.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # Frequency amplitude descriptor per channel
        X = torch.fft.rfft2(x, norm='ortho')          # (B, C, H, W//2+1) complex
        amp = X.abs().mean(dim=[2, 3])                 # (B, C)
        gate = torch.sigmoid(self.fc(amp))             # (B, C)
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class FreqBandDecomp(nn.Module):
    """
    Multi-frequency spatial decomposition.

    Separates x into:
      low_freq  = adaptive_avgpool(x, H//pool_ratio, W//pool_ratio) upsampled
      high_freq = x - low_freq

    Learns per-channel mixing weights (alpha, beta) so each FPN level can
    adaptively trust edge detail vs. smooth structure.

    At P2 (high res, small objects): model learns high beta (edge detail matters)
    At P5/P6 (low res, large objects): model learns high alpha (shape suffices)
    """

    def __init__(self, channels, pool_ratio=4):
        super().__init__()
        self.pool_ratio = pool_ratio
        # Per-channel learnable weights; initialized to equal mix
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))  # low-freq weight
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))   # high-freq weight

    def forward(self, x):
        H, W = x.shape[2:]
        ph = max(H // self.pool_ratio, 1)
        pw = max(W // self.pool_ratio, 1)
        low_freq = F.adaptive_avg_pool2d(x, (ph, pw))
        low_freq = F.interpolate(low_freq, size=(H, W),
                                 mode='bilinear', align_corners=False)
        high_freq = x - low_freq
        # sigmoid keeps weights in (0,1); initialized near 0.5 via zeros init
        return torch.sigmoid(self.alpha) * low_freq + torch.sigmoid(self.beta) * high_freq


class FAModule(nn.Module):
    """Frequency-Aware Module: FCG followed by FBD."""

    def __init__(self, channels, reduction=8, pool_ratio=4):
        super().__init__()
        self.fcg = FreqChannelGate(channels, reduction)
        self.fbd = FreqBandDecomp(channels, pool_ratio)

    def forward(self, x):
        x = self.fcg(x)
        x = self.fbd(x)
        return x


@NECKS.register_module()
class FAFPN(FPN):
    """
    Frequency-Aware FPN.

    Identical to standard FPN with one FAModule inserted after each output
    conv (including extra levels P6/P7 for RetinaNet).

    Args:
        fa_reduction (int): Channel reduction ratio in FreqChannelGate. Default: 8.
        fa_pool_ratio (int): Pooling ratio in FreqBandDecomp. Default: 4.
        All other args forwarded to FPN.
    """

    def __init__(self, *args, fa_reduction=8, fa_pool_ratio=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.fa_modules = nn.ModuleList([
            FAModule(self.out_channels, fa_reduction, fa_pool_ratio)
            for _ in range(self.num_outs)
        ])
        self._init_fa()

    def _init_fa(self):
        for m in self.fa_modules.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @auto_fp16()
    def forward(self, inputs):
        outs = list(super().forward(inputs))
        for i, fa in enumerate(self.fa_modules):
            outs[i] = fa(outs[i])
        return tuple(outs)
