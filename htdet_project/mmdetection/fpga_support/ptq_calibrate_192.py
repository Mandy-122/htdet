#!/usr/bin/env python3
"""
ptq_calibrate_192.py
--------------------
Post-Training Quantization (PTQ) for HTDet 192-channel variant.

Identical to ptq_calibrate.py except:
  CONFIG     → configs/htdet/htdet_gpu_low_gflops_192.py
  CHECKPOINT → work_dirs/htdet_low_gflops_192/latest.pth
  outdir     → ptq_results_192/

Usage:
  python fpga_support/ptq_calibrate_192.py \
      --num-cal 200 \
      --num-det 20  \
      --outdir  ptq_results_192
"""

# ── patch the two constants before importing shared logic ─────────────────
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# Override module-level constants BEFORE the shared code sets them
import fpga_support.ptq_calibrate as _ptq

_ptq.CONFIG     = str(ROOT / 'configs/htdet/htdet_gpu_low_gflops_192.py')
_ptq.CHECKPOINT = str(ROOT / 'work_dirs/htdet_low_gflops_192/latest.pth')

# Re-expose everything from the shared module so this file acts as a drop-in
from fpga_support.ptq_calibrate import *   # noqa: F401,F403
import fpga_support.ptq_calibrate as _ptq

import argparse

def main():
    ap = argparse.ArgumentParser(description='PTQ calibration for HTDet 192-ch model')
    ap.add_argument('--checkpoint', default=_ptq.CHECKPOINT)
    ap.add_argument('--val-dir',    default=str(ROOT / 'data/urpc/val2018/images'))
    ap.add_argument('--num-cal',    type=int, default=200,
                    help='Number of calibration images')
    ap.add_argument('--num-det',    type=int, default=20,
                    help='Images for float32 vs W8A32 detection comparison')
    ap.add_argument('--outdir',     default=str(ROOT / 'ptq_results_192'))
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--save-checkpoint', action='store_true',
                    help='Save fake-quantized .pth to <outdir>/ptq_model.pth')
    args = ap.parse_args()

    import random, torch
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nHTDet PTQ Calibration — 192-ch model")
    print(f"  Config     : {_ptq.CONFIG}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Val dir    : {args.val_dir}")
    print(f"  Cal images : {args.num_cal}")
    print(f"  Device     : {device}")

    val_dir  = Path(args.val_dir)
    all_imgs = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))
    random.shuffle(all_imgs)
    if not all_imgs:
        print(f"ERROR: no images found in {val_dir}"); return
    print(f"  Found {len(all_imgs)} images in val dir")

    print("\nLoading model...")
    model = _ptq.load_model(args.checkpoint, device)
    conv_layers = _ptq.collect_conv_layers(model)
    print(f"  Conv2d layers to quantize: {len(conv_layers)}")

    cal_imgs = all_imgs[:args.num_cal]
    collectors = _ptq.calibrate(model, cal_imgs, args.num_cal, device)

    import numpy as np
    print("\n  Computing quantization scales...")
    weight_scales = {}
    for name, mod in conv_layers.items():
        bn = _ptq.find_paired_bn(model, name)
        w_fused = _ptq.get_fused_weight(mod, bn)
        weight_scales[name] = _ptq.compute_weight_scale_perchannel(w_fused)

    layer_results = []
    for name, mod in conv_layers.items():
        bn      = _ptq.find_paired_bn(model, name)
        w_fused = _ptq.get_fused_weight(mod, bn)
        scale   = weight_scales[name]
        w_q     = _ptq.quantize_dequantize_weight(w_fused, scale)

        col       = collectors.get(name)
        act_min   = col.min_val  if col else None
        act_max   = col.max_val  if col else None
        act_scale = _ptq.compute_act_scale_pertensor(act_min, act_max) if col else None

        sq = _ptq.sqnr(w_fused, w_q)
        cs = _ptq.cosine_sim(w_fused, w_q)

        layer_results.append({
            'name':       name,
            'w_shape':    list(mod.weight.shape),
            'w_scale_max': float(scale.max()),
            'w_scale_min': float(scale.min()),
            'act_min':    float(act_min)   if act_min   is not None else None,
            'act_max':    float(act_max)   if act_max   is not None else None,
            'act_scale':  float(act_scale) if act_scale is not None else None,
            'sqnr_db':    sq,
            'cosine_sim': cs,
            'has_bn':     bn is not None,
        })

    det_imgs = all_imgs[:args.num_det]
    print(f"\n  Float32 detections ({args.num_det} images)...")
    float_counts = _ptq.run_detection(model, det_imgs, args.num_det, device)

    print(f"  W8A32 detections ({args.num_det} images)...")
    with _ptq.fake_quantize_weights(model):
        quant_counts = _ptq.run_detection(model, det_imgs, args.num_det, device)

    int8_outdir = Path(args.outdir) / 'ptq_int8_weights'
    print(f"\n  Exporting INT8 weights → {int8_outdir}")
    _ptq.export_int8_weights(model, layer_results, int8_outdir)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import json
    scales_json = {
        r['name']: {'w_scale': [r['w_scale_min'], r['w_scale_max']],
                    'act_scale': r['act_scale'],
                    'act_range': [r['act_min'], r['act_max']]}
        for r in layer_results
    }
    with open(outdir / 'ptq_scales.json', 'w') as f:
        json.dump(scales_json, f, indent=2)

    _ptq.print_report(layer_results, float_counts, quant_counts, str(outdir))

    if args.save_checkpoint:
        _ptq.save_ptq_checkpoint(model, args.checkpoint,
                                  str(outdir / 'ptq_model.pth'))

    print("\nDone. Outputs in:", args.outdir)


if __name__ == '__main__':
    main()
