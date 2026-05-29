#!/usr/bin/env python3
"""
run_export_meta.py — Full PTQ meta export with all fixes applied.

Patches applied to export_ptq_meta.main():
  1. patched_load_sd: returns pre-loaded full state dict (no checkpoint path search)
  2. patched_append_mvit_v2: adds proj_1x1_scale[d] after local conv3x3 BN in each
     MobileViT block AND uses full state dict for LN keys (backbone.model.stages_X.1.norm)
     New backbone_meta layout: 2926768 + 144 + 192 + 240 = 2927344 floats
  3. patched_export_fpn_meta: exports [scale[192], bias[192]] per conv → 3072 floats
     (was 1536, missing per-channel dequant scale)

Run from mmdetection/ root:
  PYTHONPATH=. python3 fpga_ptq_192/run_export_meta.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT     = os.path.join(BASE_DIR, "work_dirs/htdet_low_gflops_192/latest.pth")

import torch
print(f"Loading checkpoint: {CKPT}")
_ckpt = torch.load(CKPT, map_location='cpu')
_FULL_SD = _ckpt.get('state_dict', _ckpt)
print(f"State dict loaded: {len(_FULL_SD)} keys")

import fpga_ptq_192.export_ptq_meta as _m

# ── Patch 1: try_load_model_sd returns the pre-loaded full state dict ─────────
def patched_load_sd():
    return _FULL_SD

_m.try_load_model_sd = patched_load_sd


# ── Patch 2: append_mvit_block_meta — insert proj_1x1_scale[d] + full SD LN ──
def patched_append_mvit_v2(meta_list, sc, model_sd, stage_prefix, conv_prefix,
                            in_ch, d, depth, tb_prefix):
    """
    Updated MobileViT block meta layout (matches mobilevit_backbone.cpp):
      1. local conv3x3 BN: eff_scale[in_ch], eff_bias[in_ch]
      2. proj 1x1 dequant scale[d]   ← NEW (was skipped)
      3. transformer blocks: depth × transformer_block_meta(d)
      4. final LN: norm_w[d], norm_b[d]  (from _FULL_SD)
      5. back proj 1x1 BN: eff_scale[in_ch], eff_bias[in_ch]
      6. fusion conv3x3 BN: eff_scale[in_ch], eff_bias[in_ch]
    """
    # 1. local conv3x3 BN
    k = f"{conv_prefix}.conv_kxk.conv"
    s = _m.get_scales(sc, k)
    b = _m.get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())

    # 2. proj 1x1 dequant scale[d]
    # key has no .conv suffix (conv_1x1 is a plain Linear, not Conv+BN)
    pk = f"{conv_prefix}.conv_1x1"
    ps = _m.get_scales(sc, pk)
    if ps is None: ps = np.zeros(d, dtype=np.float32)
    assert len(ps) == d, f"proj_1x1 scale {conv_prefix}: got {len(ps)}, expected {d}"
    meta_list.extend(ps.tolist())
    print(f"  proj_1x1 {conv_prefix}: scale[{d}] mean={np.mean(ps):.5f}")

    # 3. transformer blocks
    for lyr in range(depth):
        block_key = f"{tb_prefix}.{lyr}"
        _m.append_transformer_block_meta(meta_list, _FULL_SD, block_key, d)

    # 4. final LN — use full state dict to find top-level norm keys
    stage_num = conv_prefix.split('stages_')[1].split('.')[0]
    ln_key = f"backbone.model.stages_{stage_num}.1.norm"
    if ln_key + ".weight" in _FULL_SD:
        meta_list.extend(_FULL_SD[ln_key + ".weight"].cpu().numpy().astype(np.float32).flatten().tolist())
        meta_list.extend(_FULL_SD[ln_key + ".bias"].cpu().numpy().astype(np.float32).flatten().tolist())
        print(f"  LN {ln_key}: found in full SD")
    else:
        print(f"  WARNING: LN key {ln_key} not found in full SD, using zeros")
        meta_list.extend(np.zeros(d, dtype=np.float32).tolist())
        meta_list.extend(np.zeros(d, dtype=np.float32).tolist())

    # 5. back proj 1x1 BN
    k = f"{conv_prefix}.conv_proj.conv"
    s = _m.get_scales(sc, k)
    b = _m.get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())

    # 6. fusion conv3x3 BN
    k = f"{conv_prefix}.conv_fusion.conv"
    s = _m.get_scales(sc, k)
    b = _m.get_bn_bias(sc, k)
    if s is None: s = np.zeros(in_ch, dtype=np.float32)
    if b is None: b = np.zeros(in_ch, dtype=np.float32)
    meta_list.extend(s.tolist())
    meta_list.extend(b.tolist())


_m.append_mvit_block_meta = patched_append_mvit_v2


# ── Patch 3: export_fpn_meta — emit [scale[192], bias[192]] per conv → 3072 ──
def patched_export_fpn_meta(sc_fpn):
    FPN_OUT_CH = 192
    print("\n=== Exporting fpn_meta_float.bin (scale+bias, 3072 floats) ===")
    meta = []

    for i in range(4):
        k = f"neck.lateral_convs.{i}.conv"
        s = _m.get_scales(sc_fpn, k)
        b = _m.get_bn_bias(sc_fpn, k)
        if s is None: s = np.zeros(FPN_OUT_CH, dtype=np.float32)
        if b is None: b = np.zeros(FPN_OUT_CH, dtype=np.float32)
        assert len(s) == FPN_OUT_CH, f"lateral_conv{i} scale len {len(s)}"
        assert len(b) == FPN_OUT_CH, f"lateral_conv{i} bias len {len(b)}"
        meta.extend(s.tolist())
        meta.extend(b.tolist())
        print(f"  lateral_conv{i}: scale[{len(s)}] + bias[{len(b)}]")

    for i in range(4):
        k = f"neck.fpn_convs.{i}.conv"
        s = _m.get_scales(sc_fpn, k)
        b = _m.get_bn_bias(sc_fpn, k)
        if s is None: s = np.zeros(FPN_OUT_CH, dtype=np.float32)
        if b is None: b = np.zeros(FPN_OUT_CH, dtype=np.float32)
        assert len(s) == FPN_OUT_CH, f"fpn_conv{i} scale len {len(s)}"
        assert len(b) == FPN_OUT_CH, f"fpn_conv{i} bias len {len(b)}"
        meta.extend(s.tolist())
        meta.extend(b.tolist())
        print(f"  fpn_conv{i}: scale[{len(s)}] + bias[{len(b)}]")

    print(f"  Total fpn_meta elements: {len(meta)}")
    out_path = os.path.join(_m.OUT_DIR, "fpn_meta_float.bin")
    _m.write_float_bin(out_path, meta)
    return len(meta)


_m.export_fpn_meta = patched_export_fpn_meta


if __name__ == "__main__":
    _m.main()
