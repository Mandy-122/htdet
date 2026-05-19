#!/usr/bin/env python3
"""
export_weights.py
Export HTDet checkpoint weights to binary files for FPGA inference.

For each layer the BN is FUSED into the preceding conv weights at export time:
  scale[c] = gamma[c] / sqrt(var[c] + eps)
  bias[c]  = beta[c]  - mean[c] * scale[c]
  effective conv output = x * scale + bias

All tensors are saved as raw IEEE-754 little-endian float32 binary files.
The FPGA/HLS loader must read 4 bytes per element (float/ap_float<32>).

Usage:
  python export_weights.py \\
      --checkpoint work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \\
      --outdir fpga_weights/

Output files:
  fpga_weights/backbone_w.bin   – backbone weights (all stages, fused BN)
  fpga_weights/fpn_w.bin        – FPN lateral + output conv weights
  fpga_weights/cls_conv_w.bin   – cls stacked conv weights (4 layers, ones scale + bias)
  fpga_weights/reg_conv_w.bin   – reg stacked conv weights (4 layers, ones scale + bias)
  fpga_weights/cls_pred_w.bin   – cls prediction conv weight
  fpga_weights/cls_pred_b.bin   – cls prediction conv bias
  fpga_weights/reg_pred_w.bin   – reg prediction conv weight
  fpga_weights/reg_pred_b.bin   – reg prediction conv bias
  fpga_weights/sizes.txt        – byte counts for each file (nbytes, not element count)
"""

import argparse, os
import torch
import numpy as np
from collections import OrderedDict


def save_bin(path: str, arr: np.ndarray) -> int:
    """Save array as little-endian float32. Returns number of bytes written."""
    arr = np.asarray(arr, dtype='<f4')
    arr.tofile(path)
    loaded = np.fromfile(path, dtype='<f4')
    assert np.array_equal(arr.flatten(), loaded), \
        f"Round-trip verification failed for {path}"
    return arr.nbytes


# ----------------------------------------------------------------
# BN FUSION
# ----------------------------------------------------------------
def fuse_conv_bn(conv_w: torch.Tensor,
                 bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                 bn_mean:  torch.Tensor, bn_var:  torch.Tensor,
                 eps: float = 1e-5) -> tuple:
    """
    Fuse BatchNorm into conv for inference.
    Returns (fused_scale, fused_bias) both shape [out_ch].
    Fused output = conv_output * scale + bias  (before activation).
    """
    std = torch.sqrt(bn_var + eps)
    scale = bn_gamma / std
    bias  = bn_beta - bn_mean * scale
    return scale.numpy(), bias.numpy()


# ----------------------------------------------------------------
# WEIGHT EXTRACTION HELPERS
# ----------------------------------------------------------------

def get(sd: dict, key: str) -> torch.Tensor:
    """Get tensor from state dict, raise if missing."""
    if key not in sd:
        raise KeyError(f"Key '{key}' not found in state_dict. "
                       f"Available keys (sample): {list(sd.keys())[:5]}")
    return sd[key]


def get_optional(sd: dict, key: str, fallback_shape=None) -> torch.Tensor:
    """Return tensor or zero tensor with fallback_shape."""
    if key in sd:
        return sd[key]
    if fallback_shape is not None:
        print(f"  Warning: '{key}' missing; using zeros {fallback_shape}")
        return torch.zeros(fallback_shape)
    return None


# ----------------------------------------------------------------
# MBCONV BLOCK WEIGHTS
# Weight stream: [expand_w, expand_bn_s, expand_bn_b,
#                  dw_w, dw_bn_s, dw_bn_b,
#                  proj_w, proj_bn_s, proj_bn_b]
# ----------------------------------------------------------------
def mbconv_weights(sd: dict, prefix: str, expand: int) -> list:
    """Extract MBConv block weights as flat numpy arrays in HLS order."""
    parts = []
    if expand > 1:
        parts.append(get(sd, f'{prefix}.expand.conv.weight').numpy())
        s, b = fuse_conv_bn(
            get(sd, f'{prefix}.expand.conv.weight'),
            get(sd, f'{prefix}.expand.bn.weight'),
            get(sd, f'{prefix}.expand.bn.bias'),
            get(sd, f'{prefix}.expand.bn.running_mean'),
            get(sd, f'{prefix}.expand.bn.running_var'))
        parts += [s, b]

    parts.append(get(sd, f'{prefix}.depthwise.conv.weight').numpy())
    s, b = fuse_conv_bn(
        get(sd, f'{prefix}.depthwise.conv.weight'),
        get(sd, f'{prefix}.depthwise.bn.weight'),
        get(sd, f'{prefix}.depthwise.bn.bias'),
        get(sd, f'{prefix}.depthwise.bn.running_mean'),
        get(sd, f'{prefix}.depthwise.bn.running_var'))
    parts += [s, b]

    parts.append(get(sd, f'{prefix}.project.conv.weight').numpy())
    s, b = fuse_conv_bn(
        get(sd, f'{prefix}.project.conv.weight'),
        get(sd, f'{prefix}.project.bn.weight'),
        get(sd, f'{prefix}.project.bn.bias'),
        get(sd, f'{prefix}.project.bn.running_mean'),
        get(sd, f'{prefix}.project.bn.running_var'))
    parts += [s, b]
    return parts


def conv_bn_weights(sd: dict, prefix: str) -> list:
    """Conv + BN weight pair → [w_flat, bn_scale, bn_bias]."""
    s, b = fuse_conv_bn(
        get(sd, f'{prefix}.weight'),
        get(sd, f'{prefix[:-4]}bn.weight'),
        get(sd, f'{prefix[:-4]}bn.bias'),
        get(sd, f'{prefix[:-4]}bn.running_mean'),
        get(sd, f'{prefix[:-4]}bn.running_var'))
    return [get(sd, f'{prefix}.weight').numpy(), s, b]


# ----------------------------------------------------------------
# EXPORT BACKBONE
# ----------------------------------------------------------------
def export_backbone(sd: dict) -> np.ndarray:
    """Extract and fuse all backbone weights in HLS consumption order.

    Layout per MBConv sub-layer: [conv_w, bn_scale, bn_bias]
      BN fused: scale = gamma/sqrt(var+eps),  bias = beta - mean*scale
      All backbone convs have bias=False, so the fusion formula is exact.

    Layout per MobileViT block transformer layer:
      [norm1_w, norm1_b,
       Wq, bq, Wk, bk, Wv, bv,   ← QKV split from fused (3*dim,dim) matrix
       proj_w, proj_b,
       norm2_w, norm2_b,
       fc1_w, fc1_b, fc2_w, fc2_b]
      LayerNorm gamma/beta are passed as-is; the HLS kernel computes mean/var
      at runtime (see fpga_utils.h:layer_norm_seq).
    """
    print("  Extracting backbone weights...")
    bp = 'backbone.model'
    parts = []

    def fuse(conv_key, bn_prefix):
        w = get(sd, f'{conv_key}.weight').numpy()
        s, b = fuse_conv_bn(
            get(sd, f'{conv_key}.weight'),
            get(sd, f'{bn_prefix}.weight'),
            get(sd, f'{bn_prefix}.bias'),
            get(sd, f'{bn_prefix}.running_mean'),
            get(sd, f'{bn_prefix}.running_var'))
        return w, s, b

    def add_mbconv(prefix):
        for sub in ['conv1_1x1', 'conv2_kxk', 'conv3_1x1']:
            w, s, b = fuse(f'{prefix}.{sub}.conv', f'{prefix}.{sub}.bn')
            parts.extend([w, s, b])
        print(f"    MBConv {prefix}: OK")

    def add_mobilevit(prefix):
        w, s, b = fuse(f'{prefix}.conv_kxk.conv', f'{prefix}.conv_kxk.bn')
        parts.extend([w, s, b])
        parts.append(get(sd, f'{prefix}.conv_1x1.weight').numpy())
        # Transformer blocks come BEFORE the block-level norm in the weight stream.
        # C code reads: [conv_1x1_w][transformer×depth][norm_w,norm_b][conv_proj][conv_fusion]
        lyr = 0
        while f'{prefix}.transformer.{lyr}.norm1.weight' in sd:
            tp = f'{prefix}.transformer.{lyr}'
            parts.extend([get(sd, f'{tp}.norm1.weight').numpy(),
                          get(sd, f'{tp}.norm1.bias').numpy()])
            qkv_w = get(sd, f'{tp}.attn.qkv.weight').numpy()
            qkv_b = get(sd, f'{tp}.attn.qkv.bias').numpy()
            d = qkv_w.shape[1]
            for i in range(3):
                parts.extend([qkv_w[i*d:(i+1)*d], qkv_b[i*d:(i+1)*d]])
            parts.extend([get(sd, f'{tp}.attn.proj.weight').numpy(),
                          get(sd, f'{tp}.attn.proj.bias').numpy()])
            parts.extend([get(sd, f'{tp}.norm2.weight').numpy(),
                          get(sd, f'{tp}.norm2.bias').numpy()])
            parts.extend([get(sd, f'{tp}.mlp.fc1.weight').numpy(),
                          get(sd, f'{tp}.mlp.fc1.bias').numpy()])
            parts.extend([get(sd, f'{tp}.mlp.fc2.weight').numpy(),
                          get(sd, f'{tp}.mlp.fc2.bias').numpy()])
            lyr += 1
        # Block-level LayerNorm (applied after all transformer blocks, before fold)
        parts.extend([get(sd, f'{prefix}.norm.weight').numpy(),
                      get(sd, f'{prefix}.norm.bias').numpy()])
        w, s, b = fuse(f'{prefix}.conv_proj.conv', f'{prefix}.conv_proj.bn')
        parts.extend([w, s, b])
        w, s, b = fuse(f'{prefix}.conv_fusion.conv', f'{prefix}.conv_fusion.bn')
        parts.extend([w, s, b])
        print(f"    MobileViT {prefix}: OK ({lyr} transformer layers)")

    w, s, b = fuse(f'{bp}.stem.conv', f'{bp}.stem.bn')
    parts.extend([w, s, b])
    print(f"    Stem: OK  shape={w.shape}")

    add_mbconv(f'{bp}.stages_0.0')
    for i in range(3):
        add_mbconv(f'{bp}.stages_1.{i}')
    add_mbconv(f'{bp}.stages_2.0')
    add_mobilevit(f'{bp}.stages_2.1')
    add_mbconv(f'{bp}.stages_3.0')
    add_mobilevit(f'{bp}.stages_3.1')
    add_mbconv(f'{bp}.stages_4.0')
    add_mobilevit(f'{bp}.stages_4.1')

    w, s, b = fuse(f'{bp}.final_conv.conv', f'{bp}.final_conv.bn')
    parts.extend([w, s, b])
    print(f"    Final conv: OK  shape={w.shape}")

    print(f"  Backbone: {len(parts)} tensors")
    return np.concatenate([p.flatten().astype(np.float32) for p in parts])


# ----------------------------------------------------------------
# EXPORT FPN
# ----------------------------------------------------------------
def export_fpn(sd: dict) -> np.ndarray:
    """Extract FPN neck weights in HLS order.

    MMDetection FPN (no norm_cfg) uses plain Conv2d(bias=True).
    HLS reads layout [weights, bias] — no scale vector.
    """
    print("  Extracting FPN weights...")
    parts = []
    FPN_IN_CHS = [64, 96, 128, 640]

    # Lateral 1x1 convolutions  layout: [w, bias]
    for i, in_ch in enumerate(FPN_IN_CHS):
        lat_key = f'neck.lateral_convs.{i}.conv'
        w = get(sd, f'{lat_key}.weight').numpy()
        b = get(sd, f'{lat_key}.bias').numpy()
        parts += [w, b]
        print(f"    FPN lateral_conv{i}: {w.shape} → OK")

    # Output 3x3 convolutions (P2→P5)  layout: [w, bias]
    for i in range(4):
        out_key = f'neck.fpn_convs.{i}.conv'
        w = get(sd, f'{out_key}.weight').numpy()
        b = get(sd, f'{out_key}.bias').numpy()
        parts += [w, b]
        print(f"    FPN output_conv{i}: {w.shape} → OK")

    return np.concatenate([p.flatten().astype(np.float32) for p in parts])


# ----------------------------------------------------------------
# EXPORT HEAD
# ----------------------------------------------------------------
def export_head(sd: dict) -> dict:
    """
    Returns dict with keys:
      cls_conv_w, reg_conv_w, cls_pred_w, cls_pred_b, reg_pred_w, reg_pred_b
    all as float32 numpy arrays.
    """
    print("  Extracting RetinaNet head weights...")
    head = {}
    NUM_STACKED = 4

    def stacked_conv_weights(branch: str) -> np.ndarray:
        parts = []
        for i in range(NUM_STACKED):
            w_key = f'bbox_head.{branch}_convs.{i}.conv.weight'
            b_key = f'bbox_head.{branch}_convs.{i}.conv.bias'
            w = get(sd, w_key).numpy()
            # No GroupNorm in this model — HLS expects [conv_w, scale=ones, bias=conv.bias]
            scale = np.ones(w.shape[0], dtype=np.float32)
            bias  = get(sd, b_key).numpy().astype(np.float32)
            parts += [w, scale, bias]
            print(f"    Head {branch}_conv{i}: {w.shape} → OK (plain conv+bias, no GN)")
        return np.concatenate([p.flatten().astype(np.float32) for p in parts])

    head['cls_conv_w'] = stacked_conv_weights('cls')
    head['reg_conv_w'] = stacked_conv_weights('reg')

    # Prediction convs
    head['cls_pred_w'] = get(sd, 'bbox_head.retina_cls.weight').numpy().flatten().astype(np.float32)
    head['cls_pred_b'] = get(sd, 'bbox_head.retina_cls.bias').numpy().flatten().astype(np.float32)
    head['reg_pred_w'] = get(sd, 'bbox_head.retina_reg.weight').numpy().flatten().astype(np.float32)
    head['reg_pred_b'] = get(sd, 'bbox_head.retina_reg.bias').numpy().flatten().astype(np.float32)
    print(f"    Head cls_pred: {head['cls_pred_w'].shape}")
    print(f"    Head reg_pred: {head['reg_pred_w'].shape}")

    return head


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Export HTDet weights for FPGA')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to MMDetection .pth checkpoint')
    ap.add_argument('--outdir', default='fpga_weights',
                    help='Output directory for binary weight files')
    ap.add_argument('--list-keys', action='store_true',
                    help='Print all state_dict keys (useful for debugging naming)')
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    print(f"  State dict: {len(sd)} tensors")

    if args.list_keys:
        print("\nAll keys:")
        for k in sorted(sd.keys()):
            print(f"  {k}  {tuple(sd[k].shape)}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    sizes = {}

    # ---- Backbone ----
    backbone_arr = export_backbone(sd)
    n = save_bin(os.path.join(args.outdir, 'backbone_w.bin'), backbone_arr)
    sizes['backbone_w'] = n
    print(f"  backbone_w.bin: {n/1e6:.2f} MB  ({backbone_arr.size} float32 elements)\n")

    # ---- FPN ----
    fpn_arr = export_fpn(sd)
    n = save_bin(os.path.join(args.outdir, 'fpn_w.bin'), fpn_arr)
    sizes['fpn_w'] = n
    print(f"  fpn_w.bin: {n/1e6:.2f} MB  ({fpn_arr.size} float32 elements)\n")

    # ---- Head ----
    head = export_head(sd)
    for key, arr in head.items():
        n = save_bin(os.path.join(args.outdir, f'{key}.bin'), arr)
        sizes[key] = n
        print(f"  {key}.bin: {n/1e6:.3f} MB  ({arr.size} float32 elements)")

    # ---- Write sizes.txt (values are byte counts: nbytes = n_elements * 4) ----
    with open(os.path.join(args.outdir, 'sizes.txt'), 'w') as f:
        for k, v in sizes.items():
            f.write(f'{k}={v}\n')
    print(f"\n  sizes.txt written to {args.outdir}/sizes.txt  (values = bytes; divide by 4 for element count)")
    print("\nDone. FPGA loader must read float32 (4 bytes/element). Update BACKBONE_W_ELEMS etc. in testbench.cpp with sizes.txt values / 4.")


if __name__ == '__main__':
    main()
