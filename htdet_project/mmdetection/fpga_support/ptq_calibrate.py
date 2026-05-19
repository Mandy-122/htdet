#!/usr/bin/env python3
"""
ptq_calibrate.py
----------------
Post-Training Quantization (PTQ) for HTDet.

Strategy
--------
  Conv2d layers (backbone MBConv, FPN, head):
      Weights  — INT8 per-channel symmetric
      Activations — INT8 per-tensor symmetric  (collected via calibration)

  Transformer Linear layers (attn.qkv / attn.proj / mlp.fc*):
      Kept FP32 — attention Q·Kᵀ and softmax have large/dynamic range.

  LayerNorm / BatchNorm:
      Always FP32 (runtime statistics).

BN fusion
---------
  Backbone Conv2d layers are followed by BN (fused at HLS export time).
  For accurate weight quantization the BN must be merged into the conv
  weight before computing the INT8 scale:
      w_fused[c] = w[c] * gamma[c] / sqrt(var[c] + eps)

Outputs (written to --outdir, default ptq_results/)
--------
  ptq_scales.json        — per-layer scale factors (weight + activation)
  ptq_report.txt         — per-layer SQNR and cosine similarity
  ptq_int8_weights/      — INT8 weight binaries (same layout as fpga_support/export_weights.py)
      backbone_w_int8.bin, fpn_w_int8.bin, cls_conv_w_int8.bin, reg_conv_w_int8.bin,
      cls_pred_w_int8.bin, reg_pred_w_int8.bin
      scales.json         — all scale vectors for the HLS loader

Usage
-----
  python fpga_support/ptq_calibrate.py \\
      --checkpoint work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \\
      --val-dir    data/urpc/val2018/images \\
      --num-cal    200 \\
      --outdir     ptq_results
"""

import argparse, os, sys, json, time, random
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

INT8_MAX = 127       # symmetric INT8: range [-128, 127], scale from 127
INT8_MIN = -128
EPS      = 1e-8


# ════════════════════════════════════════════════════════════════════════════
# 1.  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint, device):
    cfg = Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device).eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# 2.  IMAGE PREPROCESSING  (same as detect_python.py)
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
# 3.  LAYER CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
TRANSFORMER_KEYWORDS = ('.attn.qkv', '.attn.proj', '.mlp.fc')

def is_transformer_linear(name):
    return any(kw in name for kw in TRANSFORMER_KEYWORDS)

def collect_conv_layers(model):
    """Return dict {name: module} for all Conv2d layers to be quantized."""
    result = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            result[name] = mod
    return result


# ════════════════════════════════════════════════════════════════════════════
# 4.  BN FUSION  (mirrors export_weights.py fuse_conv_bn)
# ════════════════════════════════════════════════════════════════════════════
def find_paired_bn(model, conv_name):
    """
    For 'backbone.model.stages_0.0.conv1_1x1.conv' look for
    'backbone.model.stages_0.0.conv1_1x1.bn'.
    Returns the BN module or None.
    """
    parts = conv_name.rsplit('.', 1)
    if len(parts) < 2 or parts[1] != 'conv':
        return None
    bn_path = parts[0] + '.bn'
    try:
        mod = model
        for p in bn_path.split('.'):
            mod = getattr(mod, p)
        if isinstance(mod, nn.BatchNorm2d):
            return mod
    except AttributeError:
        pass
    return None


def get_fused_weight(conv, bn=None):
    """
    Return the BN-fused effective weight tensor [out_ch, ...].
    If no BN, returns conv.weight directly.
    """
    w = conv.weight.detach().float()
    if bn is None:
        return w
    gamma = bn.weight.detach().float()
    var   = bn.running_var.detach().float()
    scale = gamma / torch.sqrt(var + bn.eps)           # [out_ch]
    shape = [-1] + [1] * (w.dim() - 1)
    return w * scale.view(shape)


def get_fused_bias(conv, bn=None):
    """
    Return the BN-fused bias vector [out_ch] as a numpy float32 array.
    Formula: beta - mean * (gamma / sqrt(var + eps))
    If no BN and no conv bias, returns None.
    """
    if bn is None:
        if conv.bias is not None:
            return conv.bias.detach().cpu().float().numpy()
        return None
    gamma = bn.weight.detach().float()
    beta  = bn.bias.detach().float()
    mean  = bn.running_mean.detach().float()
    var   = bn.running_var.detach().float()
    scale = gamma / torch.sqrt(var + bn.eps)
    return (beta - mean * scale).cpu().numpy()


# ════════════════════════════════════════════════════════════════════════════
# 5.  QUANTIZATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compute_weight_scale_perchannel(w_fused):
    """
    Symmetric per-channel INT8 scale.
    w_fused: [out_ch, ...] → scale: [out_ch]
    """
    w_flat = w_fused.view(w_fused.shape[0], -1)
    max_abs = w_flat.abs().max(dim=1).values.clamp(min=EPS)
    return max_abs / INT8_MAX


def quantize_dequantize_weight(w_fused, scale):
    """Simulate INT8 weight quantization (fake-quant for accuracy check)."""
    shape = [-1] + [1] * (w_fused.dim() - 1)
    s = scale.view(shape)
    w_q = (w_fused / s).round().clamp(INT8_MIN, INT8_MAX)
    return w_q * s                       # dequantized float


def compute_act_scale_pertensor(min_val, max_val):
    """
    Symmetric per-tensor INT8 scale from calibrated range.
    Post-ReLU activations: min_val ≥ 0, so max_abs = max_val.
    Pre-ReLU or FPN/head: use max(|min|, |max|).
    """
    max_abs = max(abs(float(min_val)), abs(float(max_val)), EPS)
    return max_abs / INT8_MAX


def fake_quantize_act(x, scale):
    """Simulate INT8 activation quantization."""
    return (x / scale).round().clamp(INT8_MIN, INT8_MAX) * scale


def sqnr(x_float, x_quant):
    """Signal-to-Quantization-Noise Ratio in dB."""
    noise = (x_float - x_quant).pow(2).mean()
    signal = x_float.pow(2).mean()
    if noise < EPS:
        return float('inf')
    return float(10 * torch.log10(signal / noise))


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)))


# ════════════════════════════════════════════════════════════════════════════
# 6.  CALIBRATION — collect activation ranges
# ════════════════════════════════════════════════════════════════════════════
class ActivationCollector:
    """Forward hook that tracks running min/max of a layer's output."""
    def __init__(self):
        self.min_val =  float('inf')
        self.max_val = -float('inf')
        self.n_samples = 0

    def hook(self, module, inp, output):
        with torch.no_grad():
            o = output.detach().float()
            self.min_val = min(self.min_val, float(o.min()))
            self.max_val = max(self.max_val, float(o.max()))
            self.n_samples += 1


def calibrate(model, image_paths, num_cal, device):
    """
    Run num_cal images through the model and collect per-Conv2d output ranges.
    Returns {layer_name: ActivationCollector}.
    """
    collectors = {}
    hooks      = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            col = ActivationCollector()
            collectors[name] = col
            hooks.append(mod.register_forward_hook(col.hook))

    paths = image_paths[:num_cal]
    print(f"\n  Calibrating on {len(paths)} images...")
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            img_chw, _, _, scale, new_h, new_w = preprocess(p)
            if img_chw is None:
                continue
            img_t = make_tensor(img_chw, device)
            meta  = make_meta(p, scale, new_h, new_w)
            try:
                model.simple_test(img_t, [meta], rescale=False)
            except Exception as e:
                print(f"    [warn] {p.name}: {e}")
                continue
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)")

    for h in hooks:
        h.remove()
    print(f"  Calibration done in {time.time()-t0:.0f}s")
    return collectors


# ════════════════════════════════════════════════════════════════════════════
# 7.  WEIGHT QUANTIZATION SIMULATION (W8A32)
# ════════════════════════════════════════════════════════════════════════════
@contextmanager
def fake_quantize_weights(model):
    """
    Temporarily replace every Conv2d's raw weight with its fake-quantized
    (dequantized INT8) version, using per-channel scale from the RAW weight.
    BN layers remain active and operate on the quantized conv output — this
    correctly models what happens in hardware when only the conv weights are
    quantized (BN fused at export, not during PTQ simulation).
    Restores originals on exit.
    """
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            w_raw = mod.weight.data.float()
            scale = compute_weight_scale_perchannel(w_raw)
            w_q   = quantize_dequantize_weight(w_raw, scale)
            originals[name] = mod.weight.data.clone()
            mod.weight.data  = w_q.to(mod.weight.dtype)
    try:
        yield
    finally:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) and name in originals:
                mod.weight.data = originals[name]


# ════════════════════════════════════════════════════════════════════════════
# 8.  LAYER-WISE QUANTIZATION ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def analyse_layer_error(model, conv_layers, collectors, image_paths, n_analyse, device):
    """
    For each Conv2d, compare float32 vs fake-quantized (W8A32) output
    on n_analyse images. Returns list of dicts with SQNR / cosine_sim.
    """
    results = []
    paths   = image_paths[:n_analyse]

    for name, mod in conv_layers.items():
        bn      = find_paired_bn(model, name)
        w_fused = get_fused_weight(mod, bn)
        w_scale = compute_weight_scale_perchannel(w_fused)
        w_q     = quantize_dequantize_weight(w_fused, w_scale)

        col = collectors.get(name)
        act_scale = None
        if col:
            act_scale = compute_act_scale_pertensor(col.min_val, col.max_val)

        # Run one image through original and fake-quantized layer
        sqnr_vals = []
        cos_vals  = []
        for p in paths[:5]:    # 5 images is enough for per-layer stats
            img_chw, *_ = preprocess(p)
            if img_chw is None:
                continue
            # Capture float output
            float_out = []
            quant_out = []
            def hook_float(m, inp, out):
                float_out.append(out.detach().cpu())
            def hook_quant(m, inp, out):
                quant_out.append(out.detach().cpu())

            h1 = mod.register_forward_hook(hook_float)
            img_t = make_tensor(img_chw, device)
            with torch.no_grad():
                model.backbone(img_t) if 'backbone' in name else None
            h1.remove()

            # Replace weight temporarily
            orig_w = mod.weight.data.clone()
            mod.weight.data = w_q.to(mod.weight.dtype)
            h2 = mod.register_forward_hook(hook_quant)
            with torch.no_grad():
                model.backbone(img_t) if 'backbone' in name else None
            h2.remove()
            mod.weight.data = orig_w

            if float_out and quant_out:
                sqnr_vals.append(sqnr(float_out[0], quant_out[0]))
                cos_vals.append(cosine_sim(float_out[0], quant_out[0]))

        results.append({
            'name':       name,
            'w_shape':    list(mod.weight.shape),
            'w_scale_max': float(w_scale.max()),
            'w_scale_min': float(w_scale.min()),
            'act_min':    float(col.min_val) if col else None,
            'act_max':    float(col.max_val) if col else None,
            'act_scale':  float(act_scale)   if act_scale else None,
            'sqnr_db':    float(np.mean(sqnr_vals)) if sqnr_vals else None,
            'cosine_sim': float(np.mean(cos_vals))  if cos_vals  else None,
            'has_bn':     bn is not None,
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# 9.  DETECTION COMPARISON  (float32 vs W8A32 fake-quant)
# ════════════════════════════════════════════════════════════════════════════
def run_detection(model, image_paths, n_images, device):
    """Returns list of per-image detection counts at SCORE_THR."""
    counts = []
    with torch.no_grad():
        for p in image_paths[:n_images]:
            img_chw, _, _, scale, new_h, new_w = preprocess(p)
            if img_chw is None:
                continue
            img_t = make_tensor(img_chw, device)
            meta  = make_meta(p, scale, new_h, new_w)
            results = model.simple_test(img_t, [meta], rescale=False)
            total = sum(
                int((bboxes[:, 4] >= SCORE_THR).sum())
                for bboxes in results[0]
                if bboxes is not None and len(bboxes)
            )
            counts.append(total)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# 10.  INT8 WEIGHT EXPORT  (same layout as export_weights.py)
# ════════════════════════════════════════════════════════════════════════════
def quantize_weight_int8(w_fused, scale):
    """Return numpy int8 array, quantized per-channel symmetric."""
    shape = [-1] + [1] * (w_fused.dim() - 1)
    s = scale.view(shape)
    w_q = (w_fused / s).round().clamp(INT8_MIN, INT8_MAX)
    return w_q.cpu().numpy().astype(np.int8)


def export_int8_weights(model, layer_results, outdir):
    """
    Export INT8 backbone / FPN / head weights.
    Each layer: [int8_weights (packed), scale_vector (float32)]
    Saves one .bin per section (backbone, fpn, cls_conv, reg_conv, cls_pred, reg_pred)
    plus a companion scales.json.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result_map = {r['name']: r for r in layer_results}
    scales_out = {}

    def get_w_int8_and_scale(name, mod):
        r = result_map.get(name, {})
        bn = find_paired_bn(model, name)
        w_fused = get_fused_weight(mod, bn)
        if 'w_scale_max' in r:
            scale = compute_weight_scale_perchannel(w_fused)
        else:
            scale = compute_weight_scale_perchannel(w_fused)
        w_int8 = quantize_weight_int8(w_fused, scale)
        return w_int8, scale.cpu().numpy()

    def save_section(section_name, layer_names_ordered):
        parts_int8  = []
        parts_scale = {}
        for lname in layer_names_ordered:
            mod = dict(model.named_modules()).get(lname)
            if mod is None:
                continue
            w_int8, scale = get_w_int8_and_scale(lname, mod)
            parts_int8.append(w_int8.flatten())
            parts_scale[lname] = scale.tolist()
            # BN-fused bias (needed by testbench to reconstruct full inference)
            bn = find_paired_bn(model, lname)
            bias = get_fused_bias(mod, bn)
            if bias is not None:
                parts_scale[lname + '.bn_bias'] = bias.tolist()
            # activation scale
            r = result_map.get(lname, {})
            if r.get('act_scale'):
                parts_scale[lname + '.act'] = r['act_scale']

        blob = np.concatenate(parts_int8)
        path = outdir / f'{section_name}_int8.bin'
        blob.tofile(str(path))
        scales_out[section_name] = parts_scale
        print(f"  {path.name}: {len(blob):,} bytes  ({len(parts_int8)} layers)")

    # Collect layer names in HLS order (mirrors export_weights.py)
    all_mods = dict(model.named_modules())

    backbone_layers = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d)
                       and n.startswith('backbone')]
    fpn_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d)
                       and n.startswith('neck')]
    cls_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d)
                       and 'cls_convs' in n]
    reg_layers      = [n for n in all_mods if isinstance(all_mods[n], nn.Conv2d)
                       and 'reg_convs' in n]
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
# 11.  REPORT
# ════════════════════════════════════════════════════════════════════════════
def print_report(layer_results, float_counts, quant_counts, outdir):
    lines = []
    sep   = '=' * 105

    lines.append(sep)
    lines.append('  HTDet PTQ Report  —  W8A32 Simulation (INT8 weights, FP32 activations)')
    lines.append(sep)
    lines.append(f"\n  {'Layer':<65} {'Shape':<22} {'SQNR(dB)':>9}  {'CosSim':>7}  "
                 f"{'ActMin':>8}  {'ActMax':>8}  {'ActScale':>9}  BN")
    lines.append('  ' + '-' * 101)

    for r in layer_results:
        shape_str = str(r['w_shape'])
        sqnr_s  = f"{r['sqnr_db']:8.1f}"   if r['sqnr_db']    is not None else '      N/A'
        cos_s   = f"{r['cosine_sim']:7.4f}" if r['cosine_sim'] is not None else '    N/A'
        amin_s  = f"{r['act_min']:8.2f}"    if r['act_min']    is not None else '     N/A'
        amax_s  = f"{r['act_max']:8.2f}"    if r['act_max']    is not None else '     N/A'
        ascl_s  = f"{r['act_scale']:9.5f}"  if r['act_scale']  is not None else '      N/A'
        bn_s    = 'Y' if r['has_bn'] else ' '
        lines.append(f"  {r['name']:<65} {shape_str:<22} {sqnr_s}  {cos_s}  "
                     f"{amin_s}  {amax_s}  {ascl_s}  {bn_s}")

    # Detection comparison
    lines.append(f"\n{sep}")
    lines.append('  Detection comparison  (score >= 0.20)')
    lines.append(sep)
    lines.append(f"  {'#':<4}  {'Float32':>8}  {'W8A32':>8}  {'Diff':>6}")
    lines.append('  ' + '-' * 30)
    total_f = total_q = 0
    for i, (f, q) in enumerate(zip(float_counts, quant_counts)):
        diff = q - f
        lines.append(f"  {i:<4}  {f:>8}  {q:>8}  {diff:>+6}")
        total_f += f; total_q += q
    lines.append('  ' + '-' * 30)
    lines.append(f"  {'TOT':<4}  {total_f:>8}  {total_q:>8}  {total_q-total_f:>+6}")
    diff_pct = 100*(total_q - total_f)/max(total_f, 1)
    lines.append(f"\n  Overall detection delta: {diff_pct:+.1f}%")

    # Summary stats
    valid = [r for r in layer_results if r['sqnr_db'] is not None]
    if valid:
        sqnr_vals = [r['sqnr_db'] for r in valid if r['sqnr_db'] != float('inf')]
        cos_vals  = [r['cosine_sim'] for r in valid if r['cosine_sim'] is not None]
        lines.append(f"\n  Weight quantization summary ({len(valid)} Conv2d layers):")
        lines.append(f"    SQNR  — min: {min(sqnr_vals):6.1f} dB   "
                     f"mean: {np.mean(sqnr_vals):6.1f} dB   "
                     f"max: {max(sqnr_vals):6.1f} dB")
        lines.append(f"    CoSim — min: {min(cos_vals):.4f}   "
                     f"mean: {np.mean(cos_vals):.4f}   "
                     f"max: {max(cos_vals):.4f}")

        low_sqnr  = [r['name'] for r in valid if r['sqnr_db'] is not None and r['sqnr_db'] < 30]
        low_cosim = [r['name'] for r in valid if r['cosine_sim'] is not None and r['cosine_sim'] < 0.99]
        if low_sqnr:
            lines.append(f"\n  Sensitive layers (SQNR < 30 dB):")
            for n in low_sqnr:
                lines.append(f"    {n}")
        if low_cosim:
            lines.append(f"\n  Sensitive layers (CosSim < 0.99):")
            for n in low_cosim:
                lines.append(f"    {n}")

    lines.append(f"\n{sep}\n")
    report_str = '\n'.join(lines)
    print(report_str)

    report_path = Path(outdir) / 'ptq_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"  Report saved: {report_path}")


# ════════════════════════════════════════════════════════════════════════════
# 12.  PTQ CHECKPOINT SAVE
# ════════════════════════════════════════════════════════════════════════════
def save_ptq_checkpoint(model, src_checkpoint_path, out_path):
    """
    Save a PTQ-quantized PyTorch checkpoint (.pth) compatible with MMDetection.

    Conv2d weights are replaced with their fake-quantized (dequantized INT8)
    equivalents using per-channel symmetric scaling on the RAW conv weight
    (BN remains active in the model, so we quantize raw weights — not BN-fused
    ones — to keep the inference graph identical to the float baseline).

    All other parameters (BN, transformer Linear, biases) are kept as float32.

    The saved file is a standard MMDetection checkpoint dict:
        {'state_dict': ..., 'meta': {'ptq': 'W8A32', ...}}

    Args:
        model:               The loaded float32 MMDetection model (eval mode).
        src_checkpoint_path: Path to the original .pth (used only to copy meta).
        out_path:            Destination path for the PTQ checkpoint.
    """
    import copy

    print(f"\n  Building PTQ state dict (fake-quantized Conv2d weights)...")
    state_dict = copy.deepcopy(model.state_dict())

    n_quantized = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        w_raw = mod.weight.detach().float()
        scale = compute_weight_scale_perchannel(w_raw)
        w_q   = quantize_dequantize_weight(w_raw, scale)
        # Write back as the same dtype the original weight used
        key = name + '.weight'
        if key in state_dict:
            state_dict[key] = w_q.to(mod.weight.dtype)
            n_quantized += 1

    print(f"  Quantized {n_quantized} Conv2d weight tensors.")

    # Load original checkpoint meta (epoch, optimizer state, etc.) if present
    orig = torch.load(src_checkpoint_path, map_location='cpu')
    meta = orig.get('meta', {})
    meta['ptq'] = 'W8A32'
    meta['ptq_note'] = ('INT8 per-channel symmetric weights (fake-quantized), '
                        'FP32 activations and BN')

    ckpt = {'state_dict': state_dict, 'meta': meta}
    torch.save(ckpt, out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Saved PTQ checkpoint → {out_path}  ({size_mb:.1f} MB)")
    print(f"  Evaluate with:")
    print(f"    python tools/test.py configs/htdet/htdet_gpu.py {out_path} "
          f"--eval bbox")


# ════════════════════════════════════════════════════════════════════════════
# 13.  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--val-dir',    default=str(ROOT / 'data/urpc/val2018/images'))
    ap.add_argument('--num-cal',    type=int, default=200,
                    help='Number of calibration images')
    ap.add_argument('--num-det',    type=int, default=20,
                    help='Images for float32 vs W8A32 detection comparison')
    ap.add_argument('--outdir',     default=str(ROOT / 'ptq_results'))
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--save-checkpoint', action='store_true',
                    help='Save a PTQ .pth checkpoint with fake-quantized weights '
                         '(saved to <outdir>/ptq_model.pth; does not affect other outputs)')
    args = ap.parse_args()

    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nHTDet PTQ Calibration")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Val dir    : {args.val_dir}")
    print(f"  Cal images : {args.num_cal}")
    print(f"  Device     : {device}")

    # ── load images ───────────────────────────────────────────────────────
    val_dir = Path(args.val_dir)
    all_imgs = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))
    random.shuffle(all_imgs)
    if not all_imgs:
        print(f"ERROR: no images found in {val_dir}"); return
    print(f"  Found {len(all_imgs)} images in val dir")

    # ── load model ────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)
    conv_layers = collect_conv_layers(model)
    print(f"  Conv2d layers to quantize: {len(conv_layers)}")

    # ── calibration ───────────────────────────────────────────────────────
    cal_imgs = all_imgs[:args.num_cal]
    collectors = calibrate(model, cal_imgs, args.num_cal, device)

    # ── compute weight + activation scales ────────────────────────────────
    print("\n  Computing quantization scales...")
    weight_scales = {}
    for name, mod in conv_layers.items():
        bn = find_paired_bn(model, name)
        w_fused = get_fused_weight(mod, bn)
        weight_scales[name] = compute_weight_scale_perchannel(w_fused)

    # ── layer-wise error analysis ─────────────────────────────────────────
    print("\n  Running per-layer error analysis (W8A32)...")
    # Simplified: compute weight quantization error stats without re-running
    # full model (too slow for all layers); use weight-level metrics instead.
    layer_results = []
    for name, mod in conv_layers.items():
        bn      = find_paired_bn(model, name)
        w_fused = get_fused_weight(mod, bn)
        scale   = weight_scales[name]
        w_q     = quantize_dequantize_weight(w_fused, scale)

        col     = collectors.get(name)
        act_min = col.min_val if col else None
        act_max = col.max_val if col else None
        act_scale = compute_act_scale_pertensor(act_min, act_max) if col else None

        # Weight-level metrics (fast, no re-running the model)
        sq  = sqnr(w_fused, w_q)
        cs  = cosine_sim(w_fused, w_q)

        layer_results.append({
            'name':       name,
            'w_shape':    list(mod.weight.shape),
            'w_scale_max': float(scale.max()),
            'w_scale_min': float(scale.min()),
            'act_min':    float(act_min)   if act_min is not None else None,
            'act_max':    float(act_max)   if act_max is not None else None,
            'act_scale':  float(act_scale) if act_scale is not None else None,
            'sqnr_db':    sq,
            'cosine_sim': cs,
            'has_bn':     bn is not None,
        })

    # ── detection comparison  (float32 vs W8A32) ─────────────────────────
    det_imgs = all_imgs[:args.num_det]
    print(f"\n  Float32 detections ({args.num_det} images)...")
    float_counts = run_detection(model, det_imgs, args.num_det, device)

    print(f"  W8A32 detections ({args.num_det} images)...")
    with fake_quantize_weights(model):
        quant_counts = run_detection(model, det_imgs, args.num_det, device)

    # ── export INT8 weights ───────────────────────────────────────────────
    int8_outdir = Path(args.outdir) / 'ptq_int8_weights'
    print(f"\n  Exporting INT8 weights → {int8_outdir}")
    scales_out = export_int8_weights(model, layer_results, int8_outdir)

    # ── save ptq_scales.json ─────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scales_json = {
        r['name']: {
            'w_scale':   [r['w_scale_min'], r['w_scale_max']],
            'act_scale': r['act_scale'],
            'act_range': [r['act_min'], r['act_max']],
            'has_bn':    r['has_bn'],
        }
        for r in layer_results
    }
    with open(outdir / 'ptq_scales.json', 'w') as f:
        json.dump(scales_json, f, indent=2)

    # ── print + save report ───────────────────────────────────────────────
    print_report(layer_results, float_counts, quant_counts, args.outdir)
    print(f"\nAll outputs in: {outdir}/")
    print("  ptq_scales.json         — per-layer scale factors")
    print("  ptq_report.txt          — SQNR / cosine similarity table")
    print("  ptq_int8_weights/       — INT8 binary weights + scales.json")
    print("\nNext: inspect ptq_report.txt for sensitive layers,")
    print("      then update fpga_types.h to use ap_fixed<8,N> per layer.")

    # ── optional: save PTQ checkpoint ────────────────────────────────────────
    if args.save_checkpoint:
        ckpt_path = outdir / 'ptq_model.pth'
        save_ptq_checkpoint(model, args.checkpoint, ckpt_path)


if __name__ == '__main__':
    main()
