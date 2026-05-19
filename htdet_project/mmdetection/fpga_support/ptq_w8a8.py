#!/usr/bin/env python3
"""
ptq_w8a8.py
-----------
W8A8 Post-Training Quantization for HTDet.

Extends W8A32 (from ptq_calibrate.py) by also quantizing activations to INT8.

Strategy
--------
  Conv2d weights    : INT8 per-channel symmetric  (same as W8A32)
  Conv2d activations: INT8 per-tensor symmetric   (NEW in W8A8)
  Transformer Linear: FP32  (attention is range-sensitive)
  BatchNorm         : FP32  (runtime statistics)

Mixed precision
---------------
  Two early depthwise conv layers have extreme activation ranges (>196):
    backbone.model.stages_0.0.conv2_kxk  (ActMax=232)
    backbone.model.stages_1.0.conv2_kxk  (ActMax=196)
  These are kept with FP32 activations to avoid precision collapse.
  All other 51 Conv2d layers have activations quantized to INT8.

Outputs  (written to --outdir, default w8a8/ptq_results/)
---------
  ptq_w8a8_scales.json    — per-layer weight + activation scale factors
  ptq_w8a8_report.txt     — SQNR, CosSim, mAP comparison table
  ptq_w8a8_model.pth      — saved PTQ checkpoint (if --save-checkpoint)
  ptq_int8_weights/       — INT8 weight binaries + scales.json (with act scales)

Usage
-----
  python fpga_support/ptq_w8a8.py \\
      --num-cal 200 --num-det 50 --outdir w8a8/ptq_results --save-checkpoint
"""

import argparse, os, sys, json, time, random, copy
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

# ── constants ────────────────────────────────────────────────────────────────
CONFIG     = str(ROOT / 'configs/htdet/htdet_gpu.py')
CHECKPOINT = str(ROOT / 'work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth')

INPUT_H = INPUT_W = 640
SCORE_THR = 0.20
MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']

INT8_MAX =  127
INT8_MIN = -128
EPS      = 1e-8

# These two layers have activation ranges >196 — keep activations in FP32
FP32_ACT_LAYERS = {
    'backbone.model.stages_0.0.conv2_kxk.conv',
    'backbone.model.stages_1.0.conv2_kxk.conv',
}


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint, device):
    cfg = Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device).eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
def preprocess(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None, None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    scale   = min(INPUT_H / h, INPUT_W / w)
    new_h, new_w = round(h * scale), round(w * scale)
    img_res  = cv2.resize(img_rgb, (new_w, new_h)).astype(np.float32)
    img_norm = (img_res - MEAN) / STD
    img_pad  = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
    img_pad[:new_h, :new_w] = img_norm
    img_chw  = img_pad.transpose(2, 0, 1).astype(np.float32)
    return img_chw, h, w, scale, new_h, new_w

def make_tensor(img_chw, device):
    return torch.from_numpy(img_chw[np.newaxis]).float().to(device)

def make_meta(img_path, scale, new_h, new_w):
    return {'img_shape': (new_h, new_w, 3), 'ori_shape': (new_h, new_w, 3),
            'pad_shape': (INPUT_H, INPUT_W, 3),
            'scale_factor': np.array([scale]*4, dtype=np.float32),
            'flip': False, 'flip_direction': None, 'filename': str(img_path)}


# ════════════════════════════════════════════════════════════════════════════
# QUANTIZATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_weight_scale_perchannel(w):
    w_flat  = w.view(w.shape[0], -1)
    max_abs = w_flat.abs().max(dim=1).values.clamp(min=EPS)
    return max_abs / INT8_MAX

def quantize_dequantize_weight(w, scale):
    s   = scale.view([-1] + [1] * (w.dim() - 1))
    w_q = (w / s).round().clamp(INT8_MIN, INT8_MAX)
    return w_q * s

def compute_act_scale(min_val, max_val):
    max_abs = max(abs(float(min_val)), abs(float(max_val)), EPS)
    return max_abs / INT8_MAX

def fake_quantize_act(x, scale):
    return (x / scale).round().clamp(INT8_MIN, INT8_MAX) * scale

def sqnr(x, xq):
    noise  = (x - xq).pow(2).mean()
    signal = x.pow(2).mean()
    return float('inf') if noise < EPS else float(10 * torch.log10(signal / noise))

def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)))

def quantize_weight_int8(w_fused, scale):
    s   = scale.view([-1] + [1] * (w_fused.dim() - 1))
    w_q = (w_fused / s).round().clamp(INT8_MIN, INT8_MAX)
    return w_q.cpu().numpy().astype(np.int8)


# ════════════════════════════════════════════════════════════════════════════
# BN HELPERS
# ════════════════════════════════════════════════════════════════════════════
def find_paired_bn(model, conv_name):
    parts = conv_name.rsplit('.', 1)
    if len(parts) < 2 or parts[1] != 'conv':
        return None
    bn_path = parts[0] + '.bn'
    try:
        mod = model
        for p in bn_path.split('.'):
            mod = getattr(mod, p)
        return mod if isinstance(mod, nn.BatchNorm2d) else None
    except AttributeError:
        return None

def get_fused_weight(conv, bn=None):
    w = conv.weight.detach().float()
    if bn is None:
        return w
    scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    return w * scale.view([-1] + [1] * (w.dim() - 1))

def get_fused_bias(conv, bn=None):
    if bn is None:
        return conv.bias.detach().cpu().float().numpy() if conv.bias is not None else None
    gamma = bn.weight.detach().float()
    beta  = bn.bias.detach().float()
    mean  = bn.running_mean.detach().float()
    var   = bn.running_var.detach().float()
    scale = gamma / torch.sqrt(var + bn.eps)
    return (beta - mean * scale).cpu().numpy()


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ════════════════════════════════════════════════════════════════════════════
class ActivationCollector:
    def __init__(self):
        self.min_val =  float('inf')
        self.max_val = -float('inf')

    def hook(self, module, inp, output):
        with torch.no_grad():
            o = output.detach().float()
            self.min_val = min(self.min_val, float(o.min()))
            self.max_val = max(self.max_val, float(o.max()))

def calibrate(model, image_paths, num_cal, device):
    collectors, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            col = ActivationCollector()
            collectors[name] = col
            hooks.append(mod.register_forward_hook(col.hook))

    print(f"\n  Calibrating on {min(num_cal, len(image_paths))} images...")
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(image_paths[:num_cal]):
            img_chw, _, _, scale, new_h, new_w = preprocess(p)
            if img_chw is None:
                continue
            try:
                model.simple_test(make_tensor(img_chw, device),
                                  [make_meta(p, scale, new_h, new_w)],
                                  rescale=False)
            except Exception as e:
                print(f"    [warn] {Path(p).name}: {e}")
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{num_cal}  ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()
    print(f"  Calibration done in {time.time()-t0:.0f}s")
    return collectors


# ════════════════════════════════════════════════════════════════════════════
# W8A8 FAKE-QUANTIZE CONTEXT MANAGER
# ════════════════════════════════════════════════════════════════════════════
@contextmanager
def fake_quantize_w8a8(model, collectors):
    """
    Fake-quantize both weights (per-channel INT8) AND activations (per-tensor INT8).
    Layers in FP32_ACT_LAYERS keep float activations (mixed precision).
    Transformer Linear layers are untouched.
    """
    originals = {}
    hooks     = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue

        # --- weight fake-quant (raw conv weight, BN stays active) ---
        w_raw = mod.weight.data.float()
        scale = compute_weight_scale_perchannel(w_raw)
        w_q   = quantize_dequantize_weight(w_raw, scale)
        originals[name] = mod.weight.data.clone()
        mod.weight.data  = w_q.to(mod.weight.dtype)

        # --- activation fake-quant hook ---
        if name in FP32_ACT_LAYERS:
            continue  # mixed precision: keep FP32 activations for outlier DW layers

        col = collectors.get(name)
        if col and col.max_val > -float('inf'):
            act_s = compute_act_scale(col.min_val, col.max_val)
            act_s_t = torch.tensor(act_s, dtype=torch.float32)

            def make_hook(s):
                def hook(m, inp, out):
                    return fake_quantize_act(out.float(), s).to(out.dtype)
                return hook

            hooks.append(mod.register_forward_hook(make_hook(act_s_t)))

    try:
        yield
    finally:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) and name in originals:
                mod.weight.data = originals[name]
        for h in hooks:
            h.remove()


# ════════════════════════════════════════════════════════════════════════════
# DETECTION RUNNER
# ════════════════════════════════════════════════════════════════════════════
def run_detection(model, image_paths, n_images, device):
    counts = []
    with torch.no_grad():
        for p in image_paths[:n_images]:
            img_chw, _, _, scale, new_h, new_w = preprocess(p)
            if img_chw is None:
                continue
            results = model.simple_test(make_tensor(img_chw, device),
                                        [make_meta(p, scale, new_h, new_w)],
                                        rescale=False)
            total = sum(
                int((bboxes[:, 4] >= SCORE_THR).sum())
                for bboxes in results[0]
                if bboxes is not None and len(bboxes)
            )
            counts.append(total)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# INT8 WEIGHT EXPORT
# ════════════════════════════════════════════════════════════════════════════
def export_int8_weights(model, collectors, outdir):
    outdir = Path(outdir)
    all_mods   = dict(model.named_modules())
    scales_out = {}

    def get_w_int8_and_scale(lname, mod):
        bn       = find_paired_bn(model, lname)
        w_fused  = get_fused_weight(mod, bn)
        scale    = compute_weight_scale_perchannel(w_fused)
        w_int8   = quantize_weight_int8(w_fused, scale)
        return w_int8, scale.cpu().numpy()

    def save_section(section_name, layer_names):
        parts_int8  = []
        parts_scale = {}
        for lname in layer_names:
            mod = all_mods.get(lname)
            if mod is None:
                continue
            w_int8, scale = get_w_int8_and_scale(lname, mod)
            parts_int8.append(w_int8.flatten())
            parts_scale[lname] = scale.tolist()

            bn   = find_paired_bn(model, lname)
            bias = get_fused_bias(mod, bn)
            if bias is not None:
                parts_scale[lname + '.bn_bias'] = bias.tolist()

            col = collectors.get(lname)
            if col and col.max_val > -float('inf'):
                act_s = compute_act_scale(col.min_val, col.max_val)
                parts_scale[lname + '.act'] = act_s
                parts_scale[lname + '.act_range'] = [float(col.min_val), float(col.max_val)]
                parts_scale[lname + '.fp32_act'] = lname in FP32_ACT_LAYERS

        blob = np.concatenate(parts_int8)
        path = outdir / f'{section_name}_int8.bin'
        blob.tofile(str(path))
        scales_out[section_name] = parts_scale
        print(f"  {path.name}: {len(blob):,} bytes  ({len(parts_int8)} layers)")

    backbone_layers = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d) and n.startswith('backbone')]
    fpn_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d) and n.startswith('neck')]
    cls_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d) and 'cls_convs' in n]
    reg_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d) and 'reg_convs' in n]
    cls_pred        = [n for n in all_mods if n == 'bbox_head.retina_cls']
    reg_pred        = [n for n in all_mods if n == 'bbox_head.retina_reg']

    save_section('backbone', backbone_layers)
    save_section('fpn',      fpn_layers)
    save_section('cls_conv', cls_layers)
    save_section('reg_conv', reg_layers)
    save_section('cls_pred', cls_pred)
    save_section('reg_pred', reg_pred)

    scales_path = outdir / 'scales.json'
    with open(scales_path, 'w') as f:
        json.dump(scales_out, f, indent=2)
    print(f"  scales.json: {scales_path}")
    return scales_out


# ════════════════════════════════════════════════════════════════════════════
# LAYER-WISE ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def analyse_layers(model, collectors):
    results = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        bn      = find_paired_bn(model, name)
        w_fused = get_fused_weight(mod, bn)
        scale   = compute_weight_scale_perchannel(w_fused)
        w_q     = quantize_dequantize_weight(w_fused, scale)
        sq      = sqnr(w_fused, w_q)
        cs      = cosine_sim(w_fused, w_q)
        col     = collectors.get(name)
        act_min = float(col.min_val) if col else None
        act_max = float(col.max_val) if col else None
        act_s   = compute_act_scale(act_min, act_max) if col else None
        results.append({
            'name': name, 'w_shape': list(mod.weight.shape),
            'sqnr_db': sq, 'cosine_sim': cs,
            'act_min': act_min, 'act_max': act_max, 'act_scale': act_s,
            'has_bn': bn is not None,
            'fp32_act': name in FP32_ACT_LAYERS,
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE
# ════════════════════════════════════════════════════════════════════════════
def save_w8a8_checkpoint(model, src_path, out_path):
    print(f"\n  Building W8A8 state dict...")
    state_dict = copy.deepcopy(model.state_dict())
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        w_raw = mod.weight.detach().float()
        scale = compute_weight_scale_perchannel(w_raw)
        w_q   = quantize_dequantize_weight(w_raw, scale)
        key = name + '.weight'
        if key in state_dict:
            state_dict[key] = w_q.to(mod.weight.dtype)
            n += 1
    print(f"  Quantized {n} Conv2d weight tensors.")
    orig = torch.load(src_path, map_location='cpu')
    meta = orig.get('meta', {})
    meta['ptq'] = 'W8A8'
    meta['ptq_note'] = 'INT8 weights + INT8 activations (mixed: 2 DW layers FP32 act)'
    torch.save({'state_dict': state_dict, 'meta': meta}, out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Saved W8A8 checkpoint → {out_path}  ({size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════════
def print_report(layer_results, float_counts, w8a32_counts, w8a8_counts, outdir):
    sep = '=' * 110
    lines = [sep,
             '  HTDet PTQ Report  —  W8A8 Simulation (INT8 weights + INT8 activations, 2 DW layers FP32 act)',
             sep]
    lines.append(f"\n  {'Layer':<65} {'Shape':<22} {'SQNR':>7}  {'CosSim':>7}  "
                 f"{'ActMax':>8}  {'ActScale':>9}  {'ActMode':>8}  BN")
    lines.append('  ' + '-' * 108)
    for r in layer_results:
        sq_s   = f"{r['sqnr_db']:7.1f}"   if r['sqnr_db'] is not None else '    N/A'
        cs_s   = f"{r['cosine_sim']:7.4f}" if r['cosine_sim'] is not None else '    N/A'
        am_s   = f"{max(abs(r['act_min']), abs(r['act_max'])):8.2f}" if r['act_max'] else '     N/A'
        as_s   = f"{r['act_scale']:9.5f}" if r['act_scale'] else '      N/A'
        mode   = 'FP32-act' if r['fp32_act'] else 'INT8-act'
        bn_s   = 'Y' if r['has_bn'] else ' '
        lines.append(f"  {r['name']:<65} {str(r['w_shape']):<22} {sq_s}  {cs_s}  "
                     f"{am_s}  {as_s}  {mode:>8}  {bn_s}")

    lines += [f"\n{sep}", '  Detection comparison  (score >= 0.20)', sep,
              f"  {'#':<4}  {'Float32':>8}  {'W8A32':>8}  {'W8A8':>8}  {'ΔW8A32':>8}  {'ΔW8A8':>8}",
              '  ' + '-' * 48]
    tf = tq32 = tq8 = 0
    for i, (f, q32, q8) in enumerate(zip(float_counts, w8a32_counts, w8a8_counts)):
        lines.append(f"  {i:<4}  {f:>8}  {q32:>8}  {q8:>8}  {q32-f:>+8}  {q8-f:>+8}")
        tf += f; tq32 += q32; tq8 += q8
    lines += ['  ' + '-' * 48,
              f"  {'TOT':<4}  {tf:>8}  {tq32:>8}  {tq8:>8}  {tq32-tf:>+8}  {tq8-tf:>+8}",
              f"\n  W8A32 detection delta: {100*(tq32-tf)/max(tf,1):+.1f}%",
              f"  W8A8  detection delta: {100*(tq8-tf)/max(tf,1):+.1f}%"]

    valid  = [r for r in layer_results if r['sqnr_db'] is not None]
    finite = [r['sqnr_db'] for r in valid if r['sqnr_db'] != float('inf')]
    cos_v  = [r['cosine_sim'] for r in valid if r['cosine_sim'] is not None]
    lines += [f"\n  Weight quantization summary ({len(valid)} Conv2d layers):",
              f"    SQNR  min:{min(finite):6.1f}  mean:{np.mean(finite):6.1f}  max:{max(finite):6.1f} dB",
              f"    CoSim min:{min(cos_v):.4f}  mean:{np.mean(cos_v):.4f}  max:{max(cos_v):.4f}",
              f"\n  Activation quantization:",
              f"    INT8-act layers : {sum(1 for r in layer_results if not r['fp32_act'])}",
              f"    FP32-act layers : {sum(1 for r in layer_results if r['fp32_act'])}  "
              f"(outlier DW convs with ActMax>196)"]
    lines.append(f"\n{sep}\n")
    report_str = '\n'.join(lines)
    print(report_str)
    rpath = Path(outdir) / 'ptq_w8a8_report.txt'
    with open(rpath, 'w') as f:
        f.write(report_str)
    print(f"  Report saved: {rpath}")
    return tf, tq32, tq8


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--val-dir',   default=str(ROOT / 'data/urpc/val2018/images'))
    ap.add_argument('--num-cal',   type=int, default=200)
    ap.add_argument('--num-det',   type=int, default=50,
                    help='Images for float32 vs W8A32 vs W8A8 detection comparison')
    ap.add_argument('--outdir',    default=str(ROOT / 'w8a8/ptq_results'))
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--save-checkpoint', action='store_true')
    args = ap.parse_args()

    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nHTDet W8A8 PTQ")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Val dir    : {args.val_dir}")
    print(f"  Cal images : {args.num_cal}")
    print(f"  Device     : {device}")
    print(f"  FP32-act layers (mixed precision): {sorted(FP32_ACT_LAYERS)}")

    val_dir  = Path(args.val_dir)
    all_imgs = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))
    random.shuffle(all_imgs)
    print(f"  Found {len(all_imgs)} images")

    print("\nLoading model...")
    model = load_model(args.checkpoint, device)
    n_conv = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Conv2d))
    print(f"  Conv2d layers: {n_conv}")

    # Calibrate
    collectors = calibrate(model, all_imgs[:args.num_cal], args.num_cal, device)

    # Layer error analysis
    print("\n  Analysing per-layer quantization error...")
    layer_results = analyse_layers(model, collectors)

    # Detection comparison: float32 vs W8A32 vs W8A8
    det_imgs = all_imgs[:args.num_det]
    print(f"\n  Float32 detections ({args.num_det} images)...")
    float_counts = run_detection(model, det_imgs, args.num_det, device)

    # Re-use W8A32 scales from existing run if available, else simulate
    w8a32_scales_path = ROOT / 'ptq_results/ptq_scales.json'
    print(f"  W8A32 detections ({args.num_det} images)...")
    from fpga_support.ptq_calibrate import (
        fake_quantize_weights as fake_quantize_w8a32
    )
    with fake_quantize_w8a32(model):
        w8a32_counts = run_detection(model, det_imgs, args.num_det, device)

    print(f"  W8A8 detections ({args.num_det} images)...")
    with fake_quantize_w8a8(model, collectors):
        w8a8_counts = run_detection(model, det_imgs, args.num_det, device)

    # Export INT8 weights
    int8_dir = outdir / 'ptq_int8_weights'
    int8_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Exporting INT8 weights → {int8_dir}")
    export_int8_weights(model, collectors, int8_dir)

    # Save scales JSON
    scales_json = {
        r['name']: {
            'w_scale':   [float(compute_weight_scale_perchannel(
                              get_fused_weight(dict(model.named_modules())[r['name']],
                                              find_paired_bn(model, r['name']))).min()),
                          float(compute_weight_scale_perchannel(
                              get_fused_weight(dict(model.named_modules())[r['name']],
                                              find_paired_bn(model, r['name']))).max())],
            'act_scale': r['act_scale'],
            'act_range': [r['act_min'], r['act_max']],
            'fp32_act':  r['fp32_act'],
            'has_bn':    r['has_bn'],
        }
        for r in layer_results
    }
    with open(outdir / 'ptq_w8a8_scales.json', 'w') as f:
        json.dump(scales_json, f, indent=2)

    # Report
    tf, tq32, tq8 = print_report(
        layer_results, float_counts, w8a32_counts, w8a8_counts, outdir)

    print(f"\nAll outputs in: {outdir}/")
    print(f"  ptq_w8a8_scales.json   — per-layer weight + activation scales")
    print(f"  ptq_w8a8_report.txt    — full layer analysis + detection comparison")
    print(f"  ptq_int8_weights/      — INT8 binaries + scales.json")

    if args.save_checkpoint:
        save_w8a8_checkpoint(model, args.checkpoint, outdir / 'ptq_w8a8_model.pth')

    return tf, tq32, tq8


if __name__ == '__main__':
    main()
