#!/usr/bin/env python3
"""
gen_csim_ref.py
Python-side reference for retina_head_top CSIM validation.

Modes
-----
--syn [--h H] [--w W]
    Use the SAME synthetic sin/cos feature that the C++ testbench generates.
    Requires only the PTQ weight files (no FPN feature map needed).
    This is the primary validation mode for Phase 1 (P6, 10×10).
    Run:  python3 gen_csim_ref.py --syn --h 10 --w 10

--level {p2|p3|p4|p5|p6}
    Use the 192-ch FPN feature saved by the fpga_ptq_192 CSIM
    (requires ../csim_validation/<level>_192ch.bin).
    Not available by default — generate by adding a dump to fpga_ptq_192/htdet_top.cpp.

Usage examples
--------------
    cd /path/to/mmdetection
    python3 fpga_retina_head/gen_csim_ref.py --syn --h 10 --w 10
    python3 fpga_retina_head/gen_csim_ref.py --syn --h 20 --w 20   # P5
    python3 fpga_retina_head/gen_csim_ref.py --syn --h 40 --w 40   # P4
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
CSIM_DIR   = os.path.join(BASE_DIR, "csim_validation")
PTQ_DIR    = os.path.join(BASE_DIR, "ptq_results_192_ep47")

HEAD_FEAT_CH    = 192
NUM_CLASSES     = 4
ANCHOR_SCALES_N = 3
ANCHOR_RATIOS_N = 3
ANCHORS_PER_LOC = ANCHOR_SCALES_N * ANCHOR_RATIOS_N   # 9
CLS_OUT_CH      = ANCHORS_PER_LOC * NUM_CLASSES        # 36
REG_OUT_CH      = ANCHORS_PER_LOC * 4                  # 36


# ── data loaders ─────────────────────────────────────────────────────────────
def load_float_bin(path, n):
    data = np.fromfile(path, dtype=np.float32)
    assert len(data) == n, f"{path}: expected {n} floats, got {len(data)}"
    return data


def load_int8_bin(path, n):
    data = np.fromfile(path, dtype=np.int8)
    assert len(data) == n, f"{path}: expected {n} int8, got {len(data)}"
    return data


# ── synthetic feature (must EXACTLY match fill_syn_feat in testbench) ─────────
def gen_syn_feat(ch, H, W):
    c = np.arange(ch, dtype=np.float32)
    h = np.arange(H,  dtype=np.float32)
    w = np.arange(W,  dtype=np.float32)
    C, HH, WW = np.meshgrid(c, h, w, indexing="ij")
    return (np.sin(C * 0.05) * np.cos(HH * 0.2 + WW * 0.3) * 2.0).astype(np.float32)


# ── forward pass helpers ──────────────────────────────────────────────────────
def conv3x3_int8(feat, w_int8, w_scale, w_bias, out_ch):
    """3×3 conv with int8 weights + per-channel dequant-scale + bias.  No activation."""
    C_in   = feat.shape[0]
    feat_t = torch.from_numpy(feat).unsqueeze(0).float()          # [1, C_in, H, W]
    w_t    = torch.from_numpy(w_int8.astype(np.float32)
                              .reshape(out_ch, C_in, 3, 3))
    bias_t = torch.zeros(out_ch)
    raw    = F.conv2d(feat_t, w_t, bias_t, padding=1).squeeze(0)  # [out_ch, H, W]
    scale_t = torch.from_numpy(w_scale.astype(np.float32)).view(out_ch, 1, 1)
    bias_t2 = torch.from_numpy(w_bias .astype(np.float32)).view(out_ch, 1, 1)
    return (raw * scale_t + bias_t2).numpy()


def stacked_convs_py(feat, w_int8_flat, w_meta_flat, n_layers=4):
    """4 stacked conv3x3 + dequant + bias + ReLU."""
    C_in   = feat.shape[0]
    S_conv = C_in * C_in * 9
    S_meta = C_in * 2
    x = feat.copy()
    for l in range(n_layers):
        lw   = w_int8_flat[l*S_conv : (l+1)*S_conv].reshape(C_in, C_in, 9)
        lm   = w_meta_flat[l*S_meta : (l+1)*S_meta]
        sc   = lm[:C_in]
        bias = lm[C_in:]
        y    = conv3x3_int8(x, lw, sc, bias, C_in)
        x    = np.maximum(y, 0.0)   # ReLU
    return x


def pred_conv_py(feat, w_int8_flat, w_scale, w_bias, out_ch):
    """Final prediction conv3x3, no activation."""
    C_in = feat.shape[0]
    lw   = w_int8_flat.reshape(out_ch, C_in, 9)
    return conv3x3_int8(feat, lw, w_scale, w_bias, out_ch)


# ── statistics (same format as C++ testbench) ─────────────────────────────────
def print_stats(label, arr):
    a = arr.ravel().astype(np.float64)
    thr_logit = -1.386   # sigmoid^{-1}(0.20)
    above     = int(np.sum(a >= thr_logit)) if "logit" in label.lower() else None
    print(f"  {label}")
    print(f"    n={len(a)}  min={float(a.min()):.6f}  max={float(a.max()):.6f}"
          f"  mean={float(a.mean()):.6f}  nonzero={int(np.count_nonzero(a))}")
    print(f"    first5={[round(float(v),7) for v in a[:5]]}")
    if above is not None:
        print(f"    anchors above score_thr=0.20: {above} / {len(a)}")


# ── load all retina head weights ──────────────────────────────────────────────
def load_head_weights():
    N_stacked_w = HEAD_FEAT_CH * HEAD_FEAT_CH * 9 * 4   # 1 327 104
    N_stacked_m = HEAD_FEAT_CH * 2 * 4                   # 1 536
    N_pred_w    = CLS_OUT_CH * HEAD_FEAT_CH * 9          # 62 208

    ws = {}
    ws["cls_conv_w"] = load_int8_bin (os.path.join(PTQ_DIR, "ptq_int8_weights/cls_conv_int8.bin"), N_stacked_w)
    ws["cls_conv_m"] = load_float_bin(os.path.join(PTQ_DIR, "cls_conv_meta_float.bin"),             N_stacked_m)
    ws["reg_conv_w"] = load_int8_bin (os.path.join(PTQ_DIR, "ptq_int8_weights/reg_conv_int8.bin"), N_stacked_w)
    ws["reg_conv_m"] = load_float_bin(os.path.join(PTQ_DIR, "reg_conv_meta_float.bin"),             N_stacked_m)
    ws["cls_pred_w"] = load_int8_bin (os.path.join(PTQ_DIR, "ptq_int8_weights/cls_pred_int8.bin"), N_pred_w)
    ws["cls_pred_s"] = load_float_bin(os.path.join(PTQ_DIR, "cls_pred_scale_float.bin"),            CLS_OUT_CH)
    ws["cls_pred_b"] = load_float_bin(os.path.join(PTQ_DIR, "cls_pred_bias_float.bin"),             CLS_OUT_CH)
    ws["reg_pred_w"] = load_int8_bin (os.path.join(PTQ_DIR, "ptq_int8_weights/reg_pred_int8.bin"), N_pred_w)
    ws["reg_pred_s"] = load_float_bin(os.path.join(PTQ_DIR, "reg_pred_scale_float.bin"),            REG_OUT_CH)
    ws["reg_pred_b"] = load_float_bin(os.path.join(PTQ_DIR, "reg_pred_bias_float.bin"),             REG_OUT_CH)
    return ws


# ── run the head forward pass ─────────────────────────────────────────────────
def run_head(feat, ws):
    cls_feat   = stacked_convs_py(feat, ws["cls_conv_w"], ws["cls_conv_m"])
    cls_logits = pred_conv_py(cls_feat, ws["cls_pred_w"], ws["cls_pred_s"], ws["cls_pred_b"], CLS_OUT_CH)
    reg_feat   = stacked_convs_py(feat, ws["reg_conv_w"], ws["reg_conv_m"])
    reg_deltas = pred_conv_py(reg_feat, ws["reg_pred_w"], ws["reg_pred_s"], ws["reg_pred_b"], REG_OUT_CH)
    return cls_logits, reg_deltas


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--syn",   action="store_true",
                        help="Use synthetic sin/cos feature (default mode)")
    parser.add_argument("--level", choices=["p2","p3","p4","p5","p6"],
                        help="Load 192-ch FPN feature from csim_validation/<level>_192ch.bin")
    parser.add_argument("--h",     type=int, default=10, help="TEST_H (default 10 = P6)")
    parser.add_argument("--w",     type=int, default=10, help="TEST_W (default 10 = P6)")
    args = parser.parse_args()

    if not args.syn and not args.level:
        args.syn = True   # default to synthetic

    H, W = args.h, args.w

    print(f"\n=== gen_csim_ref.py  H={H}  W={W} ===")

    ws = load_head_weights()

    if args.syn:
        print(f"\n  Mode: synthetic feature  sin(c*0.05)*cos(h*0.2+w*0.3)*2.0")
        feat = gen_syn_feat(HEAD_FEAT_CH, H, W)
        tag  = f"syn_{H}x{W}"
    else:
        feat_file = os.path.join(CSIM_DIR, f"{args.level}_192ch.bin")
        if not os.path.isfile(feat_file):
            sys.exit(f"ERROR: {feat_file} not found.\n"
                     f"       Dump 192-ch FPN features from fpga_ptq_192 CSIM first.")
        feat = np.fromfile(feat_file, dtype=np.float32).reshape(HEAD_FEAT_CH, H, W)
        tag  = args.level
        print(f"  feat {tag}: shape={feat.shape}  min={feat.min():.4f}  max={feat.max():.4f}")

    cls_logits, reg_deltas = run_head(feat, ws)

    print(f"\n  === Python reference statistics  [{tag}] ===")
    print_stats("cls_logits", cls_logits)
    print_stats("reg_deltas", reg_deltas)

    # Save reference binaries for optional binary-level comparison
    out_cls = os.path.join(CSIM_DIR, f"cls_logits_{tag}_python.bin")
    out_reg = os.path.join(CSIM_DIR, f"reg_deltas_{tag}_python.bin")
    cls_logits.astype(np.float32).tofile(out_cls)
    reg_deltas.astype(np.float32).tofile(out_reg)
    print(f"\n  Saved: {out_cls}")
    print(f"  Saved: {out_reg}")
    print(f"\n  Compare these stats against the C++ testbench Test 2 output.\n")


if __name__ == "__main__":
    main()
