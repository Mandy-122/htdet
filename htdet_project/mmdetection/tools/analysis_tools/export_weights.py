"""
export_weights.py
-----------------
Export a trained HTDet (RetinaNet + MobileViT-S backbone) checkpoint to flat
float32 binary files that the FPGA testbench (fpga_temp/testbench.cpp) can load.

Usage
-----
  # Export weights to a directory:
  python tools/analysis_tools/export_weights.py \\
      --config  configs/htdet/htdet_gpu.py \\
      --checkpoint work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \\
      --out-dir  weights/

  # Inspect model parameter names without exporting (useful for debugging):
  python tools/analysis_tools/export_weights.py \\
      --config  configs/htdet/htdet_gpu.py \\
      --checkpoint work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \\
      --print-params

Output files (all float32 binary, row-major / C-order)
-------------------------------------------------------
  backbone_w.bin   MobileViT-S backbone weights (BN fused into conv)
  fpn_w.bin        FPN neck weights
  cls_conv_w.bin   RetinaHead cls stacked-conv weights (GN affine exported)
  reg_conv_w.bin   RetinaHead reg stacked-conv weights (GN affine exported)
  cls_pred_w.bin   Cls prediction conv weights
  cls_pred_b.bin   Cls prediction conv bias
  reg_pred_w.bin   Reg prediction conv weights
  reg_pred_b.bin   Reg prediction conv bias

Weight layout
-------------
Each file is a flat stream of float32 values in the same order consumed by
the corresponding HLS header (mobilevit_backbone.h, fpn_neck.h, retina_head.h).

BN fusion
---------
  fused_scale[c] = gamma[c] / sqrt(var[c] + eps)
  fused_bias[c]  = beta[c]  - mean[c] * fused_scale[c]
  FPGA computes:  output = conv(x) * fused_scale + fused_bias

GN approximation
----------------
  GroupNorm cannot be fused at export time because its normalisation statistics
  depend on the activation at runtime.  The FPGA skips the normalisation step
  and applies only the affine transform (gamma, beta).  This is an approximation;
  expect small numerical differences in the RetinaHead stacked convs compared to
  the Python model.  For higher accuracy, consider replacing GN with BN in the
  Python config and retraining.

TIMM MobileVitBlock (TIMM >= 0.6 / 1.0.x)
-------------------------------------------
  TIMM's MobileVitBlock does NOT use token_proj or token_unproj linear layers.
  Instead, after conv_1x1 (in_ch→d), the feature map [d,H,W] is unfolded into
  P=patch_area=4 independent views each of shape [N, d] where N=(H/p)*(W/p).
  The transformer runs on each view independently (all views share the same weights).
  A final LayerNorm is applied after the transformer blocks.

  This means the FPGA mobilevit_block functions must:
    1. Unfold [d,H,W] → 4 views of [N,d]
    2. For each view: run depth × transformer_block (same weights)
    3. Apply final LayerNorm
    4. Fold back to [d,H,W]
  Weight layout per block:
    conv1_w+bn  conv2_w+scale=1+bias=0  depth×transformer_block  norm_w+norm_b  conv3_w+bn  conv4_w+bn

TIMM transformer dimensions (mobilevit_s, TIMM 1.0.x):
  Stage 2: in_ch=96,  d=144, depth=2
  Stage 3: in_ch=128, d=192, depth=4
  Stage 4: in_ch=160, d=240, depth=3
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

MMDET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, MMDET_ROOT)

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint


# ============================================================
# HELPERS
# ============================================================

BN_EPS = 1e-5


def to_np(t):
    return t.detach().float().cpu().numpy()


def fuse_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, eps=BN_EPS):
    """Return (flat_weight, fused_scale, fused_bias) with BN baked in."""
    scale = to_np(bn_gamma) / np.sqrt(to_np(bn_var) + eps)
    bias  = to_np(bn_beta)  - to_np(bn_mean) * scale
    return to_np(conv_w).reshape(-1), scale, bias


def export_gn(conv_w, gn_gamma, gn_beta):
    """
    Export conv + GroupNorm as (flat_weight, gamma, beta).
    The FPGA applies:  output = conv(x) * gamma + beta
    (normalisation step is skipped — see module docstring).
    """
    return to_np(conv_w).reshape(-1), to_np(gn_gamma), to_np(gn_beta)


def plain_bias(conv_w, conv_b):
    """Plain conv with bias, no norm.  Encode bias as fused_bias, scale=1."""
    oc = conv_w.shape[0]
    return to_np(conv_w).reshape(-1), np.ones(oc, np.float32), to_np(conv_b)


def write_bin(arrays, path):
    flat = np.concatenate([a.reshape(-1).astype(np.float32) for a in arrays])
    flat.tofile(path)
    kb = flat.nbytes // 1024
    print(f"  [write] {os.path.basename(path):30s}  {len(flat):>10,} elem  {kb:>7,} KB")
    return len(flat)


# ============================================================
# CONV-NORM-ACT unwrappers
# TIMM 1.0.x uses ConvNormAct containers where .bn is BatchNormAct2d.
# ============================================================

def unwrap_conv(layer):
    """Return the Conv2d inside a ConvNormAct (or the layer itself)."""
    if isinstance(layer, nn.Conv2d):
        return layer
    if hasattr(layer, 'conv') and isinstance(layer.conv, nn.Conv2d):
        return layer.conv
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError(f"Cannot find Conv2d in {type(layer)}")


def unwrap_bn(layer):
    """Return the BN module inside a ConvNormAct (or the layer itself).

    In TIMM 1.0.x ConvNormAct the BN is stored as .bn (BatchNormAct2d,
    which subclasses nn.BatchNorm2d and includes the activation).
    """
    if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return layer
    if hasattr(layer, 'bn'):
        return layer.bn     # BatchNormAct2d — has .weight/.bias/.running_mean/.running_var
    for m in layer.modules():
        if isinstance(m, nn.BatchNorm2d):
            return m
    raise ValueError(f"Cannot find BatchNorm2d in {type(layer)}")


# ============================================================
# BOTTLENECK BLOCK (TIMM mobilevit_s BottleneckBlock)
# Children: conv1_1x1 / conv2_kxk / conv3_1x1  (all ConvNormAct)
# conv2_kxk is a depthwise conv (groups == in_ch).
#
# Weight layout (per mobilevit_backbone.h mbconv_*):
#   expand_w[hid*in] bn_s[hid] bn_b[hid]   (conv1_1x1)
#   dw_w[hid*9]      bn_s[hid] bn_b[hid]   (conv2_kxk, depthwise)
#   proj_w[out*hid]  bn_s[out] bn_b[out]   (conv3_1x1)
# ============================================================

def export_bottleneck(blk, bufs):
    """Export one TIMM BottleneckBlock (expand 1x1 → DW 3x3 → project 1x1)."""
    c1 = blk.conv1_1x1
    w, s, b = fuse_bn(c1.conv.weight, c1.bn.weight, c1.bn.bias,
                      c1.bn.running_mean, c1.bn.running_var)
    bufs += [w, s, b]

    c2 = blk.conv2_kxk
    w, s, b = fuse_bn(c2.conv.weight, c2.bn.weight, c2.bn.bias,
                      c2.bn.running_mean, c2.bn.running_var)
    bufs += [w, s, b]

    c3 = blk.conv3_1x1
    w, s, b = fuse_bn(c3.conv.weight, c3.bn.weight, c3.bn.bias,
                      c3.bn.running_mean, c3.bn.running_var)
    bufs += [w, s, b]


# ============================================================
# TRANSFORMER BLOCK
# Layout (per mobilevit_backbone.h transformer_block()):
#   ln1_w[d]  ln1_b[d]
#   Wq[d*d]  bq[d]   Wk[d*d]  bk[d]   Wv[d*d]  bv[d]   Wo[d*d]  bo[d]
#   ln2_w[d]  ln2_b[d]
#   W1[2d*d]  b1[2d]   W2[d*2d]  b2[d]
# ============================================================

def export_transformer_block(block, bufs):
    """
    Export one TIMM transformer Block.
    TIMM uses combined qkv Linear [3d, d]; we split into Wq/Wk/Wv each [d, d].
    """
    norm1 = block.norm1
    bufs += [to_np(norm1.weight), to_np(norm1.bias)]

    attn = block.attn
    qkv  = attn.qkv   # Linear([3d, d])
    proj = attn.proj  # Linear([d, d])

    dim = norm1.weight.shape[0]
    qkv_w = to_np(qkv.weight)   # [3*dim, dim]
    qkv_b = to_np(qkv.bias) if qkv.bias is not None else np.zeros(3 * dim, np.float32)
    Wq, Wk, Wv = qkv_w[:dim], qkv_w[dim:2*dim], qkv_w[2*dim:]
    bq, bk, bv = qkv_b[:dim], qkv_b[dim:2*dim], qkv_b[2*dim:]

    Wo = to_np(proj.weight)
    bo = to_np(proj.bias) if proj.bias is not None else np.zeros(dim, np.float32)

    bufs += [Wq.reshape(-1), bq, Wk.reshape(-1), bk, Wv.reshape(-1), bv, Wo.reshape(-1), bo]

    norm2 = block.norm2
    bufs += [to_np(norm2.weight), to_np(norm2.bias)]

    mlp = block.mlp
    fc1 = mlp.fc1
    fc2 = mlp.fc2
    bufs += [to_np(fc1.weight).reshape(-1),
             to_np(fc1.bias) if fc1.bias is not None else np.zeros(fc1.out_features, np.float32)]
    bufs += [to_np(fc2.weight).reshape(-1),
             to_np(fc2.bias) if fc2.bias is not None else np.zeros(fc2.out_features, np.float32)]


# ============================================================
# MOBILEVIT BLOCK (TIMM 1.0.x MobileVitBlock)
#
# TIMM structure (no token_proj / token_unproj):
#   conv_kxk  : ConvNormAct  3x3(in_ch→in_ch) + BN + SiLU
#   conv_1x1  : Conv2d       1x1(in_ch→d), no bias, no BN
#   transformer: Sequential  of depth × Block
#   norm       : LayerNorm   final norm applied after transformer
#   conv_proj  : ConvNormAct 1x1(d→in_ch) + BN + SiLU
#   conv_fusion: ConvNormAct 3x3(2*in_ch→in_ch) + BN + SiLU
#
# Weight layout consumed by updated mobilevit_backbone.h:
#   conv1_w[in_ch*in_ch*9]  bn_s[in_ch]  bn_b[in_ch]
#   conv2_w[d*in_ch]        scale=1[d]   bias=0[d]    (no BN; exported as identity)
#   depth × transformer_block weights
#   norm_w[d]  norm_b[d]                               (final LayerNorm)
#   conv3_w[in_ch*d]        bn_s[in_ch]  bn_b[in_ch]
#   conv4_w[in_ch*2*in_ch*9] bn_s[in_ch] bn_b[in_ch]
# ============================================================

def export_mobilevit_block(mvit_block, bufs):
    """Export TIMM 1.0.x MobileVitBlock (4-view independent transformer)."""

    # conv1 = conv_kxk: 3x3(in_ch→in_ch) + BN + SiLU
    c1 = mvit_block.conv_kxk
    w, s, b = fuse_bn(c1.conv.weight, c1.bn.weight, c1.bn.bias,
                      c1.bn.running_mean, c1.bn.running_var)
    bufs += [w, s, b]

    # conv2 = conv_1x1: 1x1(in_ch→d), no BN, no bias
    # Export scale=1 and bias=0 so the FPGA's conv1x1_bn_silu becomes identity norm.
    c2 = mvit_block.conv_1x1
    oc = c2.weight.shape[0]
    bufs += [to_np(c2.weight).reshape(-1),
             np.ones(oc, np.float32),
             np.zeros(oc, np.float32)]

    # Transformer blocks (P=4 views share these same weights at runtime)
    for blk in mvit_block.transformer.children():
        export_transformer_block(blk, bufs)

    # Final LayerNorm (applied to each of the 4 views after transformer)
    norm = mvit_block.norm
    bufs += [to_np(norm.weight), to_np(norm.bias)]

    # conv3 = conv_proj: 1x1(d→in_ch) + BN + SiLU
    c3 = mvit_block.conv_proj
    w, s, b = fuse_bn(c3.conv.weight, c3.bn.weight, c3.bn.bias,
                      c3.bn.running_mean, c3.bn.running_var)
    bufs += [w, s, b]

    # conv4 = conv_fusion: 3x3(2*in_ch→in_ch) + BN + SiLU
    c4 = mvit_block.conv_fusion
    w, s, b = fuse_bn(c4.conv.weight, c4.bn.weight, c4.bn.bias,
                      c4.bn.running_mean, c4.bn.running_var)
    bufs += [w, s, b]


# ============================================================
# BACKBONE EXPORT
# Follows the exact order in mobilevit_backbone() in mobilevit_backbone.h
#
# FeatureListNet (TIMM features_only=True) exposes model stages as:
#   .stem       ConvNormAct  3x3(3→16, s=2) + BN + SiLU
#   .stages_0   Sequential   MBConv(16→32, expand=4, s=1) × 1
#   .stages_1   Sequential   MBConv(32→64, s=2) + MBConv(64→64,s=1) × 2
#   .stages_2   Sequential   MBConv(64→96, s=2) + MobileVitBlock(96, d=144, depth=2)
#   .stages_3   Sequential   MBConv(96→128, s=2) + MobileVitBlock(128, d=192, depth=4)
#   .stages_4   Sequential   MBConv(128→160, s=2) + MobileVitBlock(160, d=240, depth=3)
#   .final_conv ConvNormAct  1x1(160→640) + BN + SiLU
# ============================================================

def _is_mobilevit_block(blk):
    return hasattr(blk, 'conv_kxk') and hasattr(blk, 'transformer')


def export_backbone(timm_model, out_path):
    """Export MobileViT-S backbone weights to backbone_w.bin."""
    bufs = []

    # Stem: ConvNormAct 3x3(3→16, s=2) + BN + SiLU
    stem = timm_model.stem
    w, s, b = fuse_bn(stem.conv.weight, stem.bn.weight, stem.bn.bias,
                      stem.bn.running_mean, stem.bn.running_var)
    bufs += [w, s, b]

    # Collect stages from FeatureListNet (stored as stages_0 .. stages_4)
    stages = [getattr(timm_model, f'stages_{i}') for i in range(5)]

    # Stage 0: 1 × BottleneckBlock(16→32, expand=4, s=1)
    for blk in stages[0]:
        export_bottleneck(blk, bufs)

    # Stage 1: MBConv(32→64, s=2) + 2×MBConv(64→64, s=1)
    for blk in stages[1]:
        export_bottleneck(blk, bufs)

    # Stage 2: MBConv(64→96, s=2) + MobileVitBlock(96, d=144, depth=2)
    for blk in stages[2]:
        if _is_mobilevit_block(blk):
            export_mobilevit_block(blk, bufs)
        else:
            export_bottleneck(blk, bufs)

    # Stage 3: MBConv(96→128, s=2) + MobileVitBlock(128, d=192, depth=4)
    for blk in stages[3]:
        if _is_mobilevit_block(blk):
            export_mobilevit_block(blk, bufs)
        else:
            export_bottleneck(blk, bufs)

    # Stage 4: MBConv(128→160, s=2) + MobileVitBlock(160, d=240, depth=3)
    for blk in stages[4]:
        if _is_mobilevit_block(blk):
            export_mobilevit_block(blk, bufs)
        else:
            export_bottleneck(blk, bufs)

    # Final expansion Conv1x1(160→640) + BN + SiLU
    fc = timm_model.final_conv
    w, s, b = fuse_bn(fc.conv.weight, fc.bn.weight, fc.bn.bias,
                      fc.bn.running_mean, fc.bn.running_var)
    bufs += [w, s, b]

    n = write_bin(bufs, out_path)
    print(f"         → set BACKBONE_W_ELEMS = {n} in testbench.cpp")
    return n


# ============================================================
# FPN EXPORT
# Layout (per fpn_neck.h):
#   lat1_w[256*64]   lat1_bn_s[256]  lat1_bn_b[256]   (C1, in=64)
#   lat2_w[256*96]   lat2_bn_s[256]  lat2_bn_b[256]   (C2, in=96)
#   lat3_w[256*128]  lat3_bn_s[256]  lat3_bn_b[256]   (C3, in=128)
#   lat4_w[256*640]  lat4_bn_s[256]  lat4_bn_b[256]   (C4, in=640)
#   out1_w[256*256*9] out1_bn_s[256] out1_bn_b[256]   (P2 output conv)
#   out2_w  out3_w  out4_w  (P3..P5, same layout)
#
# MMDetection FPN without norm_cfg has no BN.
# Lateral convs have bias → encoded as (scale=1, bias=conv_bias).
# ============================================================

def export_fpn(neck, out_path):
    """Export FPN neck weights to fpn_w.bin."""
    bufs = []

    lat_convs = neck.lateral_convs
    fpn_convs = neck.fpn_convs

    def _export_conv_module(conv_module, bufs):
        conv = conv_module.conv if hasattr(conv_module, 'conv') else conv_module
        bn = getattr(conv_module, 'bn', None) or getattr(conv_module, 'norm', None)
        if isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm1d)):
            w, s, b = fuse_bn(conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var)
        else:
            w, s, b = plain_bias(conv.weight, conv.bias)
        bufs += [w, s, b]

    for conv_module in lat_convs:
        _export_conv_module(conv_module, bufs)

    for conv_module in fpn_convs:
        _export_conv_module(conv_module, bufs)

    write_bin(bufs, out_path)


# ============================================================
# RETINA HEAD EXPORT
# Stacked conv layout (per retina_head.h retina_stacked_convs()):
#   For each of 4 layers:
#     conv_w[HEAD_FEAT_CH * HEAD_FEAT_CH * 9]
#     gn_gamma[HEAD_FEAT_CH]    (exported as "bn_scale")
#     gn_beta[HEAD_FEAT_CH]     (exported as "bn_bias")
#
# Pred conv layout:
#   cls_pred_w[9*4 * HEAD_FEAT_CH * 9]   (no BN/GN)
#   cls_pred_b[9*4]
#   reg_pred_w[9*4 * HEAD_FEAT_CH * 9]
#   reg_pred_b[9*4]
# ============================================================

def export_stacked_convs(conv_list, gn_list, out_path):
    """Export one set of stacked convs (cls or reg branch) with GN."""
    bufs = []
    for conv_module, gn_module in zip(conv_list, gn_list):
        conv = conv_module.conv if hasattr(conv_module, 'conv') else conv_module
        gn   = gn_module if isinstance(gn_module, nn.GroupNorm) else None
        if gn is None:
            candidate = getattr(conv_module, 'gn', None) or getattr(conv_module, 'norm', None)
            if isinstance(candidate, nn.GroupNorm):
                gn = candidate
        if gn is not None:
            w, s, b = export_gn(conv.weight, gn.weight, gn.bias)
        else:
            bn = getattr(conv_module, 'bn', None) or getattr(conv_module, 'norm', None)
            if isinstance(bn, (nn.BatchNorm2d, nn.BatchNorm1d)):
                w, s, b = fuse_bn(conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var)
            else:
                oc = conv.weight.shape[0]
                bias = to_np(conv.bias) if conv.bias is not None else np.zeros(oc, np.float32)
                w, s, b = to_np(conv.weight).reshape(-1), np.ones(oc, np.float32), bias
        bufs += [w, s, b]
    write_bin(bufs, out_path)


def export_head(bbox_head, out_dir):
    """Export RetinaHead weights to cls_conv_w.bin, reg_conv_w.bin, cls/reg_pred_w/b.bin."""

    cls_convs = bbox_head.cls_convs
    reg_convs = bbox_head.reg_convs

    def get_gn_list(conv_list):
        gns = []
        for cm in conv_list:
            gn = getattr(cm, 'gn', None) or getattr(cm, 'norm', None)
            if gn is None and hasattr(cm, '_modules'):
                for m in cm._modules.values():
                    if isinstance(m, nn.GroupNorm):
                        gn = m; break
            gns.append(gn)
        return gns

    cls_gns = get_gn_list(cls_convs)
    reg_gns = get_gn_list(reg_convs)

    export_stacked_convs(cls_convs, cls_gns, os.path.join(out_dir, 'cls_conv_w.bin'))
    export_stacked_convs(reg_convs, reg_gns, os.path.join(out_dir, 'reg_conv_w.bin'))

    cls_pred = bbox_head.retina_cls
    reg_pred = bbox_head.retina_reg

    write_bin([to_np(cls_pred.weight).reshape(-1)], os.path.join(out_dir, 'cls_pred_w.bin'))
    write_bin([to_np(cls_pred.bias)],               os.path.join(out_dir, 'cls_pred_b.bin'))
    write_bin([to_np(reg_pred.weight).reshape(-1)], os.path.join(out_dir, 'reg_pred_w.bin'))
    write_bin([to_np(reg_pred.bias)],               os.path.join(out_dir, 'reg_pred_b.bin'))


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Export HTDet weights for FPGA testbench')
    parser.add_argument('--config',      required=True, help='MMDetection config file')
    parser.add_argument('--checkpoint',  required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--out-dir',     default='weights', help='Output directory')
    parser.add_argument('--print-params', action='store_true',
                        help='Print all named parameters and exit (debugging aid)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading config:     {args.config}")
    print(f"Loading checkpoint: {args.checkpoint}")
    cfg   = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    if args.print_params:
        print("\n=== Model parameter names ===")
        for name, param in model.named_parameters():
            print(f"  {name:<80s}  {list(param.shape)}")
        return

    # TIMMBackbone stores the FeatureListNet as self.model
    timm_model = model.backbone.model

    print(f"\nBackbone type: {type(timm_model).__name__}")
    print(f"Top-level children: {[n for n, _ in timm_model.named_children()]}")
    print(f"\nExporting weights to: {args.out_dir}/\n")

    export_backbone(timm_model, os.path.join(args.out_dir, 'backbone_w.bin'))
    export_fpn(model.neck, os.path.join(args.out_dir, 'fpn_w.bin'))
    export_head(model.bbox_head, args.out_dir)

    print("\nDone.  Load order in testbench.cpp:")
    print("  backbone_w.bin  fpn_w.bin")
    print("  cls_conv_w.bin  reg_conv_w.bin")
    print("  cls_pred_w.bin  cls_pred_b.bin")
    print("  reg_pred_w.bin  reg_pred_b.bin")
    print("\nNote: GN in RetinaHead stacked convs is approximated (affine only).")
    print("Note: TIMM MobileVitBlock uses 4-view transformer (no token_proj/unproj).")
    print("      Update MVIT_S*_DIM in fpga_types.h: S2=144, S3=192, S4=240.")


if __name__ == '__main__':
    main()
