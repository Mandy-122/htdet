"""
export_fpn_weights.py
Exports FPN neck weights from the HTDet PTQ checkpoint into the binary format
expected by fpn_p5_top (Phase 1: P5+P6 only).

Usage (run from mmdetection/ root):
    python fpga_fpn_192/export_fpn_weights.py \
        --config configs/htdet/htdet_gpu.py \
        --checkpoint work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \
        --outdir weights \
        --image csim_validation/input_image.bin

Outputs (all in <outdir>/):
    fpn_p5_w_conv.bin  — 454,656 bytes (int8): lat4_w + out4_w
    fpn_p5_w_meta.bin  —   1,536 bytes (float32): lat4_b + out4_b
    c4_feat.bin        — 256,000 bytes (float32 CHW): backbone C4 output

Weight layout written:
    w_conv:
      [0 ..  122879]  lat4 lateral 1×1  int8  [192, 640]  row-major
      [122880..454655] out4 output 3×3   int8  [192, 192, 3, 3]  OIHW row-major
    w_meta:
      [0 .. 191]  lat4 bias  float32  [192]
      [192..383]  out4 bias  float32  [192]
"""

import argparse
import os
import sys
import numpy as np
import torch

# ---- add mmdetection to path ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model(config_path, checkpoint_path):
    from mmdet.apis import init_detector
    device = 'cpu'
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def quantize_to_int8(w: np.ndarray) -> np.ndarray:
    """Per-tensor symmetric int8 quantization (simple scale = max_abs / 127)."""
    max_abs = np.max(np.abs(w))
    if max_abs == 0:
        return np.zeros_like(w, dtype=np.int8)
    scale = max_abs / 127.0
    q = np.round(w / scale).clip(-128, 127).astype(np.int8)
    return q


def run_backbone(model, image_bin_path: str) -> dict:
    """
    Run the backbone and FPN on the test image.
    Returns a dict with C4 feature map and FPN outputs for reference.
    """
    img = np.fromfile(image_bin_path, dtype=np.float32)  # CHW 320×320
    img_tensor = torch.from_numpy(img).reshape(1, 3, 320, 320)

    feats = {}
    hooks = []

    def hook_factory(name):
        def hook(m, inp, out):
            feats[name] = out.detach().cpu().numpy()
        return hook

    # Hook backbone outputs (C1..C4)
    backbone = model.backbone
    for idx, stage_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
        if hasattr(backbone, stage_name):
            h = getattr(backbone, stage_name).register_forward_hook(
                hook_factory(f'C{idx+1}'))
            hooks.append(h)

    with torch.no_grad():
        _ = model.backbone(img_tensor)

    for h in hooks:
        h.remove()

    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/htdet/htdet_gpu.py')
    parser.add_argument('--checkpoint', default='work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth')
    parser.add_argument('--outdir',     default='weights')
    parser.add_argument('--image',      default='csim_validation/input_image.bin',
                        help='preprocessed input image .bin (float32 CHW 3×320×320)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = build_model(args.config, args.checkpoint)

    neck = model.neck   # mmdet FPN

    # ----------------------------------------------------------------
    # Extract FPN weights
    # ----------------------------------------------------------------
    # lateral_convs: list of Conv2d(in, out, 1)
    # fpn_convs:     list of Conv2d(out, out, 3, padding=1)

    lat_convs = neck.lateral_convs   # [0]=C1, [1]=C2, [2]=C3, [3]=C4
    fpn_convs = neck.fpn_convs       # [0]=P2, [1]=P3, [2]=P4, [3]=P5

    # lat4: index 3  (C4=640 → 192)
    lat4_w = lat_convs[3].weight.detach().cpu().numpy()   # [192, 640, 1, 1]
    lat4_b = lat_convs[3].bias.detach().cpu().numpy()     # [192]

    # out4: index 3  (192 → 192, 3×3)
    out4_w = fpn_convs[3].weight.detach().cpu().numpy()   # [192, 192, 3, 3]
    out4_b = fpn_convs[3].bias.detach().cpu().numpy()     # [192]

    # Reshape lateral to [OC, IC] (drop spatial 1×1 dims)
    lat4_w = lat4_w.reshape(lat4_w.shape[0], lat4_w.shape[1])  # [192, 640]

    # Quantize to int8
    lat4_w_q = quantize_to_int8(lat4_w)   # [192, 640]  row-major
    out4_w_q = quantize_to_int8(out4_w.reshape(192, 192*9)).reshape(192, 192, 3, 3)

    # Flatten to layout expected by C code:
    #   conv1x1_plain: weights[oc*in_ch + ic]   → lat4_w_q[oc, ic]  → row-major ✓
    #   fpn_conv3x3:   weights[(oc*192+ic)*9 + kh*3+kw]  → out4_w_q[oc, ic, kh, kw] → row-major ✓
    lat4_w_flat = lat4_w_q.flatten()          # 122,880 int8
    out4_w_flat = out4_w_q.flatten()           # 331,776 int8

    w_conv = np.concatenate([lat4_w_flat, out4_w_flat])
    w_meta = np.concatenate([lat4_b.astype(np.float32),
                              out4_b.astype(np.float32)])

    # ----------------------------------------------------------------
    # Extract C4 feature map
    # ----------------------------------------------------------------
    print("Running backbone to extract C4 feature map...")
    feats = run_backbone(model, args.image)
    c4_key = [k for k in feats if '4' in k]
    if not c4_key:
        print("WARNING: Could not hook C4 — dumping zero feature map.")
        c4 = np.zeros(640 * 10 * 10, dtype=np.float32)
    else:
        c4 = feats[c4_key[0]].squeeze(0)  # [640, 10, 10]
        if c4.shape != (640, 10, 10):
            print(f"WARNING: C4 shape {c4.shape}, expected (640,10,10) for 320×320 input")
        c4 = c4.flatten().astype(np.float32)

    # ----------------------------------------------------------------
    # Write output files
    # ----------------------------------------------------------------
    wc_path = os.path.join(args.outdir, 'fpn_p5_w_conv.bin')
    wm_path = os.path.join(args.outdir, 'fpn_p5_w_meta.bin')
    c4_path = os.path.join(args.outdir, 'c4_feat.bin')

    w_conv.tofile(wc_path)
    w_meta.tofile(wm_path)
    c4.tofile(c4_path)

    print(f"\nWrote:")
    print(f"  {wc_path}  ({w_conv.nbytes:,} B, int8,   lat4_w + out4_w)")
    print(f"  {wm_path}  ({w_meta.nbytes:,} B, float32, lat4_b + out4_b)")
    print(f"  {c4_path}   ({c4.nbytes:,} B, float32, CHW 640×10×10)")
    print(f"\nExpected by C testbench (Test 3):")
    print(f"  w_conv: {w_conv.nbytes:,} B  (FPN_P5_WCONV_ELEMS={192*640+192*192*9})")
    print(f"  w_meta: {w_meta.nbytes:,} B  (FPN_P5_WMETA_ELEMS={2*192} floats × 4)")


if __name__ == '__main__':
    main()
