#!/usr/bin/env python3
"""
export_ptq_meta.py  —  Generate float32 meta binary files for HTDet PTQ INT8 FPGA.

Reads ptq_results_192/ptq_int8_weights/scales.json and generates:
  ptq_results_192/backbone_meta_float.bin   — backbone BN eff_scale, eff_bias, transformer weights
  ptq_results_192/fpn_meta_float.bin        — FPN biases (lateral + output)
  ptq_results_192/cls_conv_meta_float.bin   — cls stacked conv: [w_dequant_scale, bias] × 4 layers
  ptq_results_192/reg_conv_meta_float.bin   — reg stacked conv: [w_dequant_scale, bias] × 4 layers
  ptq_results_192/cls_pred_scale_float.bin  — cls pred per-channel dequant scale
  ptq_results_192/cls_pred_bias_float.bin   — cls pred bias
  ptq_results_192/reg_pred_scale_float.bin  — reg pred per-channel dequant scale
  ptq_results_192/reg_pred_bias_float.bin   — reg pred bias

The backbone_meta binary follows the EXACT traversal order of mobilevit_backbone.cpp:
  1. Stem conv: eff_scale[16], eff_bias[16]
  2. Stage 0 MBConv(16→32): expand_eff_scale[64], eff_bias[64], dw_eff_scale[64], eff_bias[64], proj_eff_scale[32], eff_bias[32]
  3. Stage 1 MBConv(32→64, s=2): expand_eff_scale[128], eff_bias[128], dw_eff_scale[128], eff_bias[128], proj_eff_scale[64], eff_bias[64]
  4. Stage 1 MBConv(64→64, s=1) × 2: same pattern with expand=256
  5. Stage 2 MBConv(64→96, s=2): expand_eff_scale[256], eff_bias[256], dw[256], proj_eff_scale[96], eff_bias[96]
  6. Stage 2 MobileViTBlock:
     - local conv3x3: eff_scale[96], eff_bias[96]
     - proj 1x1: (no BN)
     - S2 transformer depth=2: [LN1_w[144], LN1_b[144], Wq[144*144], bq[144], Wk..., Wv..., Wo..., LN2_w, LN2_b, W1[288*144], b1[288], W2[144*288], b2[144]]
     - final LN: norm_w[144], norm_b[144]
     - back proj 1x1 BN: eff_scale[96], eff_bias[96]
     - fusion conv3x3 BN: eff_scale[96], eff_bias[96]
  7. Stage 3 MBConv(96→128, s=2) + MobileViTBlock (S3: d=192, depth=4)
  8. Stage 4 MBConv(128→160, s=2) + MobileViTBlock (S4: d=240, depth=3)
  9. Final expansion Conv1x1(160→640): eff_scale[640], eff_bias[640]

NOTE: In scales.json the per-layer scales are already eff_scale = w_dequant_scale
(which equals max(abs(w[oc,...])) / 127.0 per output channel, then multiplied by
the BN scale gamma/sqrt(var+eps) to get the effective fused BN scale).
The bn_bias key contains the effective BN bias (beta - mean * gamma/sqrt(var+eps)).

For transformer blocks, the weights come from the original PyTorch model state_dict
(since transformers are NOT quantized — only conv layers are INT8 in W8A32 PTQ).
We need to load the original model to extract transformer weights.

If the original model is not available, this script can still generate the conv
meta files from scales.json alone, and leave the transformer portion as zeros
with a warning. The transformer section in backbone_meta will then need to be
populated by the user.

Usage:
  cd /workspace/ckarfa/htdet/htdet_project/mmdetection
  python3 fpga_ptq_192/export_ptq_meta.py
"""

import os
import sys
import json
import struct
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALES_JSON = os.path.join(BASE_DIR, "ptq_results_192_ep47", "ptq_int8_weights", "scales.json")
OUT_DIR     = os.path.join(BASE_DIR, "ptq_results_192_ep47")

# Model architecture constants (must match fpga_types.h)
STEM_CH         = 16
STAGE0_CH       = 32
C1_CH           = 64
C2_CH           = 96
C3_CH           = 128
C4_CH           = 640
STAGE4_PRE_CH   = 160
MVIT_S2_DIM     = 144
MVIT_S2_DEPTH   = 2
MVIT_S3_DIM     = 192
MVIT_S3_DEPTH   = 4
MVIT_S4_DIM     = 240
MVIT_S4_DEPTH   = 3
FPN_OUT_CH      = 192
HEAD_FEAT_CH    = 192
HEAD_STACKED    = 4
ANCHORS_PER_LOC = 9
NUM_CLASSES     = 4

# ============================================================
# HELPERS
# ============================================================
def write_float_bin(path, arr):
    """Write a flat float32 numpy array to binary file."""
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(path)
    print(f"  Written {len(arr)} float32 elements → {path}")
    print(f"    min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")

def get_scales(sc, key):
    """Get per-channel scale list from scales dict, or zeros if missing."""
    if key in sc:
        return np.array(sc[key], dtype=np.float32)
    print(f"  WARNING: key '{key}' not found in scales, using zeros")
    return None

def get_bn_bias(sc, key):
    """Get per-channel BN bias from scales dict."""
    bias_key = key + ".bn_bias"
    if bias_key in sc:
        return np.array(sc[bias_key], dtype=np.float32)
    print(f"  WARNING: key '{bias_key}' not found in scales, using zeros")
    return None

def append_mbconv_meta(meta_list, sc, prefix, in_ch, out_ch, expand):
    """
    Append MBConv meta for one block.
    Expand conv (if expand>1), depthwise conv, project conv.
    Each conv contributes: eff_scale[out_ch] + eff_bias[out_ch].
    """
    hid = in_ch * expand

    # Expand conv (conv1_1x1)
    if expand > 1:
        k = f"{prefix}.conv1_1x1.conv"
        s = get_scales(sc, k)
        b = get_bn_bias(sc, k)
        if s is None: s = np.zeros(hid, dtype=np.float32)
        if b is None: b = np.zeros(hid, dtype=np.float32)
        assert len(s) == hid, f"expand scale mismatch: {len(s)} vs {hid} for {k}"
        meta_list.extend(s.tolist())
        meta_list.extend(b.tolist())

    # Depthwise conv (conv2_kxk)
    k = f"{prefix}.conv2_kxk.conv"
    s = get_scales(sc, k)
    b = get_bn_bias(sc, k)
    if s is None: s = np.zeros(hid, dtype=np.float32)
    if b is None: b = np.zeros(hid, dtype=np.float32)
    assert len(s) == hid, f"dw scale mismatch: {len(s)} vs {hid} for {k}"
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())

    # Project conv (conv3_1x1)
    k = f"{prefix}.conv3_1x1.conv"
    s = get_scales(sc, k)
    b = get_bn_bias(sc, k)
    if s is None: s = np.zeros(out_ch, dtype=np.float32)
    if b is None: b = np.zeros(out_ch, dtype=np.float32)
    assert len(s) == out_ch, f"proj scale mismatch: {len(s)} vs {out_ch} for {k}"
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())


def append_transformer_block_meta(meta_list, model_sd, block_key, dim):
    """
    Append transformer block float weights.
    Layout per block (matches transformer_block_sX in mobilevit_backbone.cpp):
      LN1_w[dim], LN1_b[dim],
      Wq[dim*dim], bq[dim],
      Wk[dim*dim], bk[dim],
      Wv[dim*dim], bv[dim],
      Wo[dim*dim], bo[dim],
      LN2_w[dim], LN2_b[dim],
      W1[2*dim*dim], b1[2*dim],
      W2[dim*2*dim], b2[dim]
    Total: dim+dim + (dim*dim+dim)*4 + dim+dim + (2*dim)*dim+(2*dim) + dim*(2*dim)+dim
    """
    hidden = dim * 2

    def get_w(name, expected_size):
        if model_sd is not None and name in model_sd:
            w = model_sd[name].cpu().numpy().astype(np.float32).flatten()
            assert len(w) == expected_size, f"size mismatch {name}: {len(w)} vs {expected_size}"
            return w
        return np.zeros(expected_size, dtype=np.float32)

    # LN1
    meta_list.extend(get_w(f"{block_key}.norm1.weight", dim).tolist())
    meta_list.extend(get_w(f"{block_key}.norm1.bias",   dim).tolist())
    # MHSA: Wq, bq, Wk, bk, Wv, bv, Wo, bo
    meta_list.extend(get_w(f"{block_key}.attn.qkv.weight", dim*dim*3).reshape(3, dim*dim)[:1].flatten().tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.weight" in model_sd
                     else np.zeros(dim*dim).tolist())  # Wq
    meta_list.extend(get_w(f"{block_key}.attn.qkv.bias", dim*3)[:dim].tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.bias" in model_sd
                     else np.zeros(dim).tolist())  # bq
    meta_list.extend(get_w(f"{block_key}.attn.qkv.weight", dim*dim*3).reshape(3, dim*dim)[1:2].flatten().tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.weight" in model_sd
                     else np.zeros(dim*dim).tolist())  # Wk
    meta_list.extend(get_w(f"{block_key}.attn.qkv.bias", dim*3)[dim:2*dim].tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.bias" in model_sd
                     else np.zeros(dim).tolist())  # bk
    meta_list.extend(get_w(f"{block_key}.attn.qkv.weight", dim*dim*3).reshape(3, dim*dim)[2:3].flatten().tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.weight" in model_sd
                     else np.zeros(dim*dim).tolist())  # Wv
    meta_list.extend(get_w(f"{block_key}.attn.qkv.bias", dim*3)[2*dim:].tolist()
                     if model_sd is not None and f"{block_key}.attn.qkv.bias" in model_sd
                     else np.zeros(dim).tolist())  # bv
    meta_list.extend(get_w(f"{block_key}.attn.proj.weight", dim*dim).tolist())   # Wo
    meta_list.extend(get_w(f"{block_key}.attn.proj.bias",   dim).tolist())        # bo
    # LN2
    meta_list.extend(get_w(f"{block_key}.norm2.weight", dim).tolist())
    meta_list.extend(get_w(f"{block_key}.norm2.bias",   dim).tolist())
    # MLP: W1[hidden*dim], b1[hidden], W2[dim*hidden], b2[dim]
    meta_list.extend(get_w(f"{block_key}.mlp.fc1.weight", hidden*dim).tolist())
    meta_list.extend(get_w(f"{block_key}.mlp.fc1.bias",   hidden).tolist())
    meta_list.extend(get_w(f"{block_key}.mlp.fc2.weight", dim*hidden).tolist())
    meta_list.extend(get_w(f"{block_key}.mlp.fc2.bias",   dim).tolist())


def append_mvit_block_meta(meta_list, sc, model_sd, stage_prefix, conv_prefix,
                            in_ch, d, depth, tb_prefix):
    """
    Append MobileViT block meta.
    Layout (matches mobilevit_block_sX in mobilevit_backbone.cpp):
      1. local conv3x3 BN: eff_scale[in_ch], eff_bias[in_ch]
      2. proj 1x1: (no BN, no meta)
      3. transformer blocks: depth × transformer_block_meta(d)
      4. final LN: norm_w[d], norm_b[d]
      5. back proj 1x1 BN: eff_scale[in_ch], eff_bias[in_ch]
      6. fusion conv3x3 BN: eff_scale[in_ch], eff_bias[in_ch]
    """
    # 1. local 3x3 conv BN (conv_kxk)
    k = f"{conv_prefix}.conv_kxk.conv"
    s = get_scales(sc, k)
    b = get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())

    # 2. proj 1x1 (conv_1x1): no BN, skip
    # (weights go to coff, no meta contribution)

    # 3. transformer blocks
    for lyr in range(depth):
        block_key = f"{tb_prefix}.{lyr}"
        append_transformer_block_meta(meta_list, model_sd, block_key, d)

    # 4. final LN
    if model_sd is not None:
        stage_num = conv_prefix.split('stages_')[1].split('.')[0]
        ln_key = f"backbone.model.stages_{stage_num}.1.norm"
        if ln_key + ".weight" in model_sd:
            meta_list.extend(model_sd[ln_key + ".weight"].cpu().numpy().astype(np.float32).flatten().tolist())
            meta_list.extend(model_sd[ln_key + ".bias"].cpu().numpy().astype(np.float32).flatten().tolist())
        else:
            print(f"  WARNING: LN key {ln_key} not found, using zeros")
            meta_list.extend(np.zeros(d, dtype=np.float32).tolist())
            meta_list.extend(np.zeros(d, dtype=np.float32).tolist())
    else:
        meta_list.extend(np.zeros(d, dtype=np.float32).tolist())
        meta_list.extend(np.zeros(d, dtype=np.float32).tolist())

    # 5. back proj 1x1 BN (conv_proj)
    k = f"{conv_prefix}.conv_proj.conv"
    s = get_scales(sc, k)
    b = get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())

    # 6. fusion conv3x3 BN (conv_fusion)
    k = f"{conv_prefix}.conv_fusion.conv"
    s = get_scales(sc, k)
    b = get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())


# ============================================================
# LOAD MODEL STATE DICT (for transformer weights)
# ============================================================
def try_load_model_sd():
    """Try to load model state dict for transformer weights. Returns None if not available."""
    try:
        import torch
        # Try common checkpoint paths
        ckpt_paths = [
            os.path.join(BASE_DIR, "work_dirs/htdet_low_gflops_192/epoch_47.pth"),
            os.path.join(BASE_DIR, "work_dirs/htdet_low_gflops_192/latest.pth"),
            os.path.join(BASE_DIR, "ptq_results_192_ep47/model_fp32.pth"),
        ]
        import glob
        for pat in ckpt_paths:
            paths = glob.glob(pat)
            if paths:
                ckpt_path = sorted(paths)[-1]
                print(f"  Loading model from: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                sd = ckpt.get('state_dict', ckpt)
                # Strip 'backbone.' prefix if needed, or keep as-is
                return sd
    except Exception as e:
        print(f"  Could not load model state dict: {e}")
    return None


def extract_transformer_sd(sd, stage_idx):
    """Extract transformer-related keys for a given stage."""
    if sd is None:
        return None
    # Try to find keys matching transformer layers in this stage
    prefix = f"backbone.model.stages_{stage_idx}.1.transformer"
    result = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            # Rekey to relative path: backbone.model.stages_X.1.transformer.N.*  → N.*
            rel = k[len(f"backbone.model.stages_{stage_idx}.1.transformer."):]
            result[rel] = v
    if result:
        # Return a dict with full keys for append_transformer_block_meta
        full = {}
        for k, v in result.items():
            full[f"backbone.model.stages_{stage_idx}.1.transformer.{k}"] = v
        return full
    return None


# ============================================================
# MAIN EXPORT FUNCTIONS
# ============================================================

def export_backbone_meta(sc_backbone, sd):
    """Export backbone meta binary following exact mobilevit_backbone.cpp traversal order."""
    print("\n=== Exporting backbone_meta_float.bin ===")
    meta = []

    # ---- STEM: Conv3x3(3→16, s=2) + BN ----
    k = "backbone.model.stem.conv"
    s = get_scales(sc_backbone, k)
    b = get_bn_bias(sc_backbone, k)
    if s is None: s = np.zeros(STEM_CH, dtype=np.float32)
    if b is None: b = np.zeros(STEM_CH, dtype=np.float32)
    meta.extend(s.tolist())
    meta.extend(b.tolist())
    print(f"  Stem: {STEM_CH*2} floats")

    # ---- STAGE 0: MBConv(16→32, expand=4, s=1) ----
    # stages_0.0: expand 16→64, dw 64, proj 64→32
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_0.0",
                       in_ch=STEM_CH, out_ch=STAGE0_CH, expand=4)
    print(f"  Stage0 MBConv: {(64+64+64+64+32+32)} floats")

    # ---- STAGE 1: MBConv(32→64, s=2) + 2×MBConv(64→64, s=1) ----
    # stages_1.0: expand 32→128, dw 128, proj 128→64
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_1.0",
                       in_ch=STAGE0_CH, out_ch=C1_CH, expand=4)
    print(f"  Stage1 MBConv0: {(128+128+128+128+64+64)} floats")
    # stages_1.1: expand 64→256, dw 256, proj 256→64
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_1.1",
                       in_ch=C1_CH, out_ch=C1_CH, expand=4)
    print(f"  Stage1 MBConv1: {(256+256+256+256+64+64)} floats")
    # stages_1.2: expand 64→256, dw 256, proj 256→64
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_1.2",
                       in_ch=C1_CH, out_ch=C1_CH, expand=4)
    print(f"  Stage1 MBConv2: {(256+256+256+256+64+64)} floats")

    # ---- STAGE 2: MBConv(64→96, s=2) + MobileViTBlock(S2) ----
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_2.0",
                       in_ch=C1_CH, out_ch=C2_CH, expand=4)
    print(f"  Stage2 MBConv: {(256+256+256+256+96+96)} floats")

    # MobileViT S2
    d = MVIT_S2_DIM
    tb_prefix_base = "backbone.model.stages_2.1.transformer"
    # Build full sd subset for transformer blocks
    tb_sd = {}
    if sd is not None:
        for lyr in range(MVIT_S2_DEPTH):
            for k_sd, v in sd.items():
                blk_key = f"backbone.model.stages_2.1.transformer.{lyr}"
                if k_sd.startswith(blk_key):
                    tb_sd[k_sd] = v
    tb_sd_use = tb_sd if tb_sd else None

    # MobileViT block meta
    append_mvit_block_meta(meta, sc_backbone, tb_sd_use,
                           stage_prefix=f"backbone.model.stages_2",
                           conv_prefix=f"backbone.model.stages_2.1",
                           in_ch=C2_CH, d=d, depth=MVIT_S2_DEPTH,
                           tb_prefix=f"backbone.model.stages_2.1.transformer")
    print(f"  Stage2 MViT block: floats so far = {len(meta)}")

    # ---- STAGE 3: MBConv(96→128, s=2) + MobileViTBlock(S3) ----
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_3.0",
                       in_ch=C2_CH, out_ch=C3_CH, expand=4)
    print(f"  Stage3 MBConv: floats so far = {len(meta)}")

    d = MVIT_S3_DIM
    tb_sd3 = {}
    if sd is not None:
        for lyr in range(MVIT_S3_DEPTH):
            for k_sd, v in sd.items():
                blk_key = f"backbone.model.stages_3.1.transformer.{lyr}"
                if k_sd.startswith(blk_key):
                    tb_sd3[k_sd] = v
    tb_sd3_use = tb_sd3 if tb_sd3 else None

    append_mvit_block_meta(meta, sc_backbone, tb_sd3_use,
                           stage_prefix=f"backbone.model.stages_3",
                           conv_prefix=f"backbone.model.stages_3.1",
                           in_ch=C3_CH, d=d, depth=MVIT_S3_DEPTH,
                           tb_prefix=f"backbone.model.stages_3.1.transformer")
    print(f"  Stage3 MViT block: floats so far = {len(meta)}")

    # ---- STAGE 4: MBConv(128→160, s=2) + MobileViTBlock(S4) ----
    append_mbconv_meta(meta, sc_backbone, "backbone.model.stages_4.0",
                       in_ch=C3_CH, out_ch=STAGE4_PRE_CH, expand=4)
    print(f"  Stage4 MBConv: floats so far = {len(meta)}")

    d = MVIT_S4_DIM
    tb_sd4 = {}
    if sd is not None:
        for lyr in range(MVIT_S4_DEPTH):
            for k_sd, v in sd.items():
                blk_key = f"backbone.model.stages_4.1.transformer.{lyr}"
                if k_sd.startswith(blk_key):
                    tb_sd4[k_sd] = v
    tb_sd4_use = tb_sd4 if tb_sd4 else None

    append_mvit_block_meta(meta, sc_backbone, tb_sd4_use,
                           stage_prefix=f"backbone.model.stages_4",
                           conv_prefix=f"backbone.model.stages_4.1",
                           in_ch=STAGE4_PRE_CH, d=d, depth=MVIT_S4_DEPTH,
                           tb_prefix=f"backbone.model.stages_4.1.transformer")
    print(f"  Stage4 MViT block: floats so far = {len(meta)}")

    # ---- Final expansion Conv1x1(160→640) + BN + SiLU ----
    k = "backbone.model.final_conv.conv"
    s = get_scales(sc_backbone, k)
    b = get_bn_bias(sc_backbone, k)
    if s is None: s = np.zeros(C4_CH, dtype=np.float32)
    if b is None: b = np.zeros(C4_CH, dtype=np.float32)
    meta.extend(s.tolist())
    meta.extend(b.tolist())
    print(f"  Final expansion conv: {C4_CH*2} floats")
    print(f"  Total backbone_meta elements: {len(meta)}")

    out_path = os.path.join(OUT_DIR, "backbone_meta_float.bin")
    write_float_bin(out_path, meta)
    return len(meta)


def export_fpn_meta(sc_fpn):
    """
    Export FPN meta binary.
    Layout (matches fpn_neck.cpp traversal):
      lateral conv biases: lat1_b[192], lat2_b[192], lat3_b[192], lat4_b[192]
      output conv biases: out1_b[192], out2_b[192], out3_b[192], out4_b[192]
    Total: 8 × 192 = 1536 floats
    """
    print("\n=== Exporting fpn_meta_float.bin ===")
    meta = []

    for i in range(4):
        k = f"neck.lateral_convs.{i}.conv"
        b = get_bn_bias(sc_fpn, k)
        if b is None: b = np.zeros(FPN_OUT_CH, dtype=np.float32)
        assert len(b) == FPN_OUT_CH, f"lateral bias {i}: {len(b)} vs {FPN_OUT_CH}"
        meta.extend(b.tolist())
        print(f"  lateral_conv{i} bias: {len(b)} floats")

    for i in range(4):
        k = f"neck.fpn_convs.{i}.conv"
        b = get_bn_bias(sc_fpn, k)
        if b is None: b = np.zeros(FPN_OUT_CH, dtype=np.float32)
        assert len(b) == FPN_OUT_CH, f"fpn_conv bias {i}: {len(b)} vs {FPN_OUT_CH}"
        meta.extend(b.tolist())
        print(f"  fpn_conv{i} bias: {len(b)} floats")

    print(f"  Total fpn_meta elements: {len(meta)}")
    out_path = os.path.join(OUT_DIR, "fpn_meta_float.bin")
    write_float_bin(out_path, meta)
    return len(meta)


def export_head_conv_meta(sc_head, prefix, out_fname):
    """
    Export head stacked conv meta binary.
    Layout per layer: w_dequant_scale[192], bias[192]
    Total: 4 × (192 + 192) = 1536 floats
    """
    print(f"\n=== Exporting {out_fname} ===")
    meta = []

    for i in range(HEAD_STACKED):
        k_scale = f"{prefix}.{i}.conv"
        k_bias  = f"{prefix}.{i}.conv"

        s = get_scales(sc_head, k_scale)
        b = get_bn_bias(sc_head, k_scale)
        if s is None: s = np.zeros(HEAD_FEAT_CH, dtype=np.float32)
        if b is None: b = np.zeros(HEAD_FEAT_CH, dtype=np.float32)
        assert len(s) == HEAD_FEAT_CH, f"scale len mismatch layer {i}: {len(s)}"
        assert len(b) == HEAD_FEAT_CH, f"bias len mismatch layer {i}: {len(b)}"

        meta.extend(s.tolist())  # w_dequant_scale first
        meta.extend(b.tolist())  # then bias
        print(f"  layer {i}: scale[{len(s)}] + bias[{len(b)}]")

    print(f"  Total {out_fname} elements: {len(meta)}")
    out_path = os.path.join(OUT_DIR, out_fname)
    write_float_bin(out_path, meta)
    return len(meta)


def export_pred_meta(sc_pred, key_prefix, scale_fname, bias_fname, out_ch):
    """
    Export prediction conv scale and bias binaries.
    scale: w_dequant_scale[out_ch] (from scales.json <key>.conv)
    bias: eff_bias[out_ch] (from scales.json <key>.conv.bn_bias)
    """
    print(f"\n=== Exporting {scale_fname} and {bias_fname} ===")

    s = get_scales(sc_pred, key_prefix)
    b = get_bn_bias(sc_pred, key_prefix)
    if s is None: s = np.zeros(out_ch, dtype=np.float32)
    if b is None: b = np.zeros(out_ch, dtype=np.float32)
    assert len(s) == out_ch, f"pred scale mismatch: {len(s)} vs {out_ch}"
    assert len(b) == out_ch, f"pred bias mismatch: {len(b)} vs {out_ch}"

    write_float_bin(os.path.join(OUT_DIR, scale_fname), s)
    write_float_bin(os.path.join(OUT_DIR, bias_fname),  b)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    print("=" * 60)
    print("HTDet PTQ Meta Export")
    print(f"Scales JSON: {SCALES_JSON}")
    print(f"Output dir:  {OUT_DIR}")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load scales
    with open(SCALES_JSON) as f:
        scales = json.load(f)

    sc_backbone = scales['backbone']
    sc_fpn      = scales['fpn']
    sc_cls_conv = scales['cls_conv']
    sc_reg_conv = scales['reg_conv']
    sc_cls_pred = scales['cls_pred']
    sc_reg_pred = scales['reg_pred']

    # Try to load model state dict for transformer weights
    print("\nAttempting to load model state dict for transformer weights...")
    sd = try_load_model_sd()
    if sd is None:
        print("  Model state dict not available — transformer weights will be zeros.")
        print("  To get correct transformer weights, ensure the model checkpoint is")
        print("  accessible at one of the expected paths (see try_load_model_sd).")
        print("  Alternatively, export transformer weights from your training script")
        print("  and add them to backbone_meta_float.bin at the appropriate offsets.")

    # Export backbone meta
    n_backbone = export_backbone_meta(sc_backbone, sd)

    # Export FPN meta
    n_fpn = export_fpn_meta(sc_fpn)

    # Export cls/reg conv meta
    export_head_conv_meta(sc_cls_conv, "bbox_head.cls_convs", "cls_conv_meta_float.bin")
    export_head_conv_meta(sc_reg_conv, "bbox_head.reg_convs", "reg_conv_meta_float.bin")

    # Export pred conv meta
    cls_out_ch = ANCHORS_PER_LOC * NUM_CLASSES   # 36
    reg_out_ch = ANCHORS_PER_LOC * 4              # 36
    export_pred_meta(sc_cls_pred, "bbox_head.retina_cls",
                     "cls_pred_scale_float.bin", "cls_pred_bias_float.bin", cls_out_ch)
    export_pred_meta(sc_reg_pred, "bbox_head.retina_reg",
                     "reg_pred_scale_float.bin", "reg_pred_bias_float.bin", reg_out_ch)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  backbone_meta: {n_backbone} float elements (expected ~2926768)")
    print(f"  fpn_meta:      {n_fpn} float elements (expected 1536)")
    print("\nNote: Testbench expects these files in ptq_results_192_ep47/")
    print("If backbone_meta is smaller than expected, transformer weights are zeros.")
    print("Run this script after loading the model checkpoint for full accuracy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
