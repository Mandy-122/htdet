#!/usr/bin/env python3
"""
run_multi_image.py
------------------
Runs Python + C-sim inference for multiple images and stores per-image results.

For each image:
  1. Preprocesses the image (keep-ratio resize + zero-pad to 640x640, ImageNet norm)
  2. Runs PyTorch inference  (model loaded ONCE for all images)
  3. Saves input_image.bin + python_detections.txt to csim_validation/
  4. Runs C-sim testbench   (reads input_image.bin, writes csim_detections.txt)
  5. Copies all outputs to  csim_validation/multi_image/<image_stem>/
  6. Runs IoU-based comparison and saves combined_detections.json + comparison_table.txt

Output layout:
  csim_validation/multi_image/
    <image_stem>/
      input_image.bin           (preprocessed CHW float32 for this image)
      python_detections.txt
      csim_detections.txt
      vis_python.jpg
      combined_detections.json
      comparison_table.txt
    summary.json                (aggregate stats across all images)

Usage (run from mmdetection/ root):
  python csim_validation/run_multi_image.py
  python csim_validation/run_multi_image.py --images img1.jpg img2.jpg img3.jpg

Runtime estimate: ~8-10 min per image (C-sim dominates). 5 images ≈ 45-50 min total.
"""

import sys
import os
import subprocess
import shutil
import json
import time
import argparse
from pathlib import Path

# ---- Paths ----
MMDET_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(MMDET_ROOT))

import cv2
import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

CONFIG     = str(MMDET_ROOT / 'configs/htdet/htdet_gpu.py')
CHECKPOINT = str(MMDET_ROOT / 'work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth')
VAL_DIR    = MMDET_ROOT / 'data/urpc/val2018/images'
CSIM_DIR   = MMDET_ROOT / 'csim_validation'
FPGA_DIR   = MMDET_ROOT / 'fpga_temp'
MULTI_DIR  = CSIM_DIR / 'multi_image'

INPUT_H   = 640
INPUT_W   = 640
SCORE_THR = 0.20
MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']
COLORS = {
    'holothurian': (0,   0,   255),
    'echinus':     (0,   255, 0),
    'scallop':     (255, 0,   0),
    'starfish':    (0,   255, 255),
}

# 5 images covering different cameras/scenes in val2018
DEFAULT_IMAGES = [
    'CHN083846_0291.jpg',    # CHN camera
    'G0024172_0636.jpg',     # GoPro G0024172
    'GOPR0293_10229.jpg',    # GoPro GOPR0293
    'YDXJ0001_10003.jpg',    # YDXJ camera
    'YN010001_1312.jpg',     # YN camera
]


# ================================================================
# Preprocessing  (matches training test pipeline)
# ================================================================
def preprocess(img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_orig = img_bgr.copy()
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w     = img_rgb.shape[:2]
    scale    = min(INPUT_H / h, INPUT_W / w)
    new_h    = round(h * scale)
    new_w    = round(w * scale)
    img_res  = cv2.resize(img_rgb, (new_w, new_h)).astype(np.float32)
    img_norm = (img_res - MEAN) / STD
    img_pad  = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
    img_pad[:new_h, :new_w] = img_norm
    img_chw  = img_pad.transpose(2, 0, 1).astype(np.float32)
    return img_chw, img_orig, scale, new_h, new_w


# ================================================================
# PyTorch inference
# ================================================================
def build_model_once():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading PyTorch model on {device}...")
    cfg = Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model.to(device)
    model.eval()
    print("  Model ready.\n")
    return model, device


def run_inference(model, img_chw, scale, new_h, new_w, device, img_path: Path):
    img_tensor = torch.from_numpy(img_chw[np.newaxis]).float().to(device)
    img_meta = {
        'img_shape'    : (new_h, new_w, 3),
        'ori_shape'    : (new_h, new_w, 3),
        'pad_shape'    : (INPUT_H, INPUT_W, 3),
        'scale_factor' : np.array([scale, scale, scale, scale], dtype=np.float32),
        'flip'         : False,
        'flip_direction': None,
        'filename'     : str(img_path),
    }
    with torch.no_grad():
        results = model.simple_test(img_tensor, [img_meta], rescale=False)
    detections = []
    for cls_id, bboxes in enumerate(results[0]):
        if bboxes is None or len(bboxes) == 0:
            continue
        for det in bboxes:
            x1, y1, x2, y2, score = det
            if score >= SCORE_THR:
                # Same format as load_detections_txt: (cls_id, score, x1, y1, x2, y2)
                detections.append((cls_id, float(score),
                                   float(x1), float(y1), float(x2), float(y2)))
    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


# ================================================================
# File I/O helpers
# ================================================================
def save_detections_txt(detections, path: Path):
    with open(path, 'w') as f:
        f.write(f'{len(detections)}\n')
        for cls_id, score, x1, y1, x2, y2 in detections:
            f.write(f'{cls_id} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')


def load_detections_txt(path: Path):
    dets = []
    if not path.exists():
        return dets
    with open(path) as f:
        n = int(f.readline().strip())
        for _ in range(n):
            p = f.readline().split()
            dets.append((int(p[0]), float(p[1]),
                         float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return dets


def visualise(img_orig, detections, out_path: Path, scale, new_h, new_w):
    vis = img_orig.copy()
    for cls_id, score, x1, y1, x2, y2 in detections:
        cls_name = CLASS_NAMES[cls_id]
        x1c = min(x1, new_w); x2c = min(x2, new_w)
        y1c = min(y1, new_h); y2c = min(y2, new_h)
        ix1 = int(x1c / scale); iy1 = int(y1c / scale)
        ix2 = int(x2c / scale); iy2 = int(y2c / scale)
        color = COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), color, 2)
        cv2.putText(vis, f'{cls_name} {score:.2f}', (ix1, max(iy1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# ================================================================
# Comparison + output
# ================================================================
def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare_detections(py_dets, csim_dets, iou_thr=0.5):
    matched_py   = [False] * len(py_dets)
    matched_csim = [False] * len(csim_dets)
    matches = []
    for i, pd in enumerate(py_dets):
        best_iou = 0.0; best_j = -1
        for j, cd in enumerate(csim_dets):
            if matched_csim[j] or pd[0] != cd[0]:
                continue
            iou = box_iou(pd[2:], cd[2:])
            if iou > best_iou:
                best_iou = iou; best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_py[i]        = True
            matched_csim[best_j] = True
            matches.append((i, best_j, best_iou))
    py_only   = [i for i, m in enumerate(matched_py)   if not m]
    csim_only = [j for j, m in enumerate(matched_csim) if not m]
    return matches, py_only, csim_only


def save_combined_json(image_name, py_dets, csim_dets, matches, py_only, csim_only, path: Path):
    def to_dict(d):
        return {'class': CLASS_NAMES[d[0]], 'score': round(d[1], 6),
                'box': [round(d[2], 2), round(d[3], 2), round(d[4], 2), round(d[5], 2)]}
    data = {
        'meta': {'image': image_name, 'score_thr': SCORE_THR},
        'python_detections': [to_dict(d) for d in py_dets],
        'csim_detections':   [to_dict(d) for d in csim_dets],
        'matched_pairs': [
            {'py_idx': i, 'csim_idx': j, 'iou': round(iou, 4),
             'python': to_dict(py_dets[i]),
             'csim':   to_dict(csim_dets[j]),
             'score_diff': round(abs(py_dets[i][1] - csim_dets[j][1]), 6)}
            for i, j, iou in matches
        ],
        'python_only': [to_dict(py_dets[i])   for i in py_only],
        'csim_only':   [to_dict(csim_dets[j]) for j in csim_only],
        'summary': {
            'python_count': len(py_dets),
            'csim_count':   len(csim_dets),
            'matched':      len(matches),
            'python_only':  len(py_only),
            'csim_only':    len(csim_only),
            'match_pct':    round(100 * len(matches) / max(len(py_dets), len(csim_dets), 1), 1),
        },
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_comparison_table(image_name, py_dets, csim_dets, matches, py_only, csim_only, path: Path):
    lines = []
    sep = '=' * 115
    lines.append(sep)
    lines.append(f'  Image   : {image_name}')
    lines.append(f'  Python  : {len(py_dets)} detections')
    lines.append(f'  C-sim   : {len(csim_dets)} detections')
    lines.append(f'  Matched : {len(matches)} pairs (IoU >= 0.5)')
    pct = 100 * len(matches) / max(len(py_dets), len(csim_dets), 1)
    lines.append(f'  Match % : {pct:.1f}%')
    lines.append(sep)

    if matches:
        hdr = (f"  {'#':<4} {'Class':<14} {'IoU':>6}  {'Py Score':>9}  "
               f"{'CSim Score':>10}  {'ΔScore':>8}  "
               f"{'Py box [x1,y1,x2,y2]':<30}  CSim box")
        lines.append(hdr)
        lines.append('  ' + '-' * 111)
        for i, j, iou in matches:
            pd = py_dets[i]; cd = csim_dets[j]
            ds = abs(pd[1] - cd[1])
            pb = f"[{pd[2]:.1f},{pd[3]:.1f},{pd[4]:.1f},{pd[5]:.1f}]"
            cb = f"[{cd[2]:.1f},{cd[3]:.1f},{cd[4]:.1f},{cd[5]:.1f}]"
            flag = '  <- large Δ' if ds > 0.1 else ''
            lines.append(f"  {i:<4} {CLASS_NAMES[pd[0]]:<14} {iou:6.3f}  "
                         f"{pd[1]:9.4f}  {cd[1]:10.4f}  {ds:8.4f}  {pb:<30}  {cb}{flag}")

    if py_only:
        lines.append('\n  Python-only (no C-sim match):')
        for i in py_only:
            d = py_dets[i]
            lines.append(f"    #{i:<3}  {CLASS_NAMES[d[0]]:<14}  score={d[1]:.4f}  "
                         f"[{d[2]:.1f},{d[3]:.1f},{d[4]:.1f},{d[5]:.1f}]")

    if csim_only:
        lines.append('\n  C-sim-only (no Python match):')
        for j in csim_only:
            d = csim_dets[j]
            lines.append(f"    #{j:<3}  {CLASS_NAMES[d[0]]:<14}  score={d[1]:.4f}  "
                         f"[{d[2]:.1f},{d[3]:.1f},{d[4]:.1f},{d[5]:.1f}]")

    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# ================================================================
# Per-image pipeline
# ================================================================
def process_image(image_name: str, model, device):
    img_path = VAL_DIR / image_name
    stem     = Path(image_name).stem
    out_dir  = MULTI_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{image_name}]")
    print(f"{'='*60}")

    # -- 1. Preprocess -------------------------------------------------
    t0 = time.time()
    img_chw, img_orig, scale, new_h, new_w = preprocess(img_path)
    print(f"  Preprocessed: valid={new_h}x{new_w}  scale={scale:.4f}")

    # -- 2. Write input_image.bin (testbench reads from csim_validation/) --
    input_bin = CSIM_DIR / 'input_image.bin'
    img_chw.flatten().astype(np.float32).tofile(str(input_bin))

    # -- 3. Python inference -------------------------------------------
    py_dets = run_inference(model, img_chw, scale, new_h, new_w, device, img_path)
    py_txt  = CSIM_DIR / 'python_detections.txt'
    save_detections_txt(py_dets, py_txt)
    print(f"  Python: {len(py_dets)} detections  (inference took {time.time()-t0:.1f}s)")

    # -- 4. Copy Python outputs to per-image dir -----------------------
    shutil.copy(py_txt,    out_dir / 'python_detections.txt')
    shutil.copy(input_bin, out_dir / 'input_image.bin')
    visualise(img_orig, py_dets, out_dir / 'vis_python.jpg', scale, new_h, new_w)
    print(f"  Saved python_detections.txt + vis_python.jpg")

    # -- 5. C-sim (no --dump: only runs htdet_inference, no extra passes) --
    csim_dst = out_dir / 'csim_detections.txt'
    csim_min = 0.0
    if csim_dst.exists():
        print(f"  C-sim already done (csim_detections.txt exists) — skipping.")
    else:
        print(f"  Running C-sim (~8-10 min)...")
        t_csim = time.time()
        result = subprocess.run(
            ['./testbench_csim',
             '../weights/',
             '../csim_validation/input_image.bin'],   # no --dump → skip backbone/FPN re-runs
            cwd=str(FPGA_DIR),
            text=True,
        )
        csim_min = (time.time() - t_csim) / 60.0
        print(f"  C-sim done: {csim_min:.1f} min  (return={result.returncode})")

        if result.returncode != 0:
            print(f"  ERROR: testbench returned {result.returncode} — skipping this image")
            return None

        # -- 6. Copy csim_detections.txt (written to fpga_temp/ CWD) ------
        csim_src = FPGA_DIR / 'csim_detections.txt'
        if not csim_src.exists():
            print(f"  ERROR: csim_detections.txt not found at {csim_src}")
            return None
        shutil.copy(csim_src, csim_dst)
        shutil.copy(csim_src, CSIM_DIR / 'csim_detections.txt')  # keep shared copy current

    # -- 7. Compare detections -----------------------------------------
    csim_dets = load_detections_txt(csim_dst)
    matches, py_only, csim_only = compare_detections(py_dets, csim_dets)
    total = max(len(py_dets), len(csim_dets), 1)
    pct   = 100 * len(matches) / total
    print(f"  Match: {len(matches)}/{total}  ({pct:.0f}%)  "
          f"py_only={len(py_only)}  csim_only={len(csim_only)}")

    # -- 8. Save comparison outputs ------------------------------------
    save_combined_json(
        image_name, py_dets, csim_dets, matches, py_only, csim_only,
        out_dir / 'combined_detections.json')
    save_comparison_table(
        image_name, py_dets, csim_dets, matches, py_only, csim_only,
        out_dir / 'comparison_table.txt')
    print(f"  Saved combined_detections.json + comparison_table.txt")

    return {
        'image':        image_name,
        'python_count': len(py_dets),
        'csim_count':   len(csim_dets),
        'matched':      len(matches),
        'python_only':  len(py_only),
        'csim_only':    len(csim_only),
        'match_pct':    round(pct, 1),
        'csim_time_min': round(csim_min, 1),
    }


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', default=DEFAULT_IMAGES,
                    help='Image filenames from data/urpc/val2018/images/')
    args = ap.parse_args()

    MULTI_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Multi-image C-sim comparison")
    print(f"  Images  : {len(args.images)}")
    print(f"  Output  : {MULTI_DIR}")
    print(f"  ETA     : ~{len(args.images) * 9} min (C-sim ~8-10 min/image)")
    print()

    model, device = build_model_once()

    all_results = []
    t_total = time.time()
    for i, img_name in enumerate(args.images, 1):
        print(f"\n[{i}/{len(args.images)}] {img_name}")
        r = process_image(img_name, model, device)
        if r:
            all_results.append(r)

    total_min = (time.time() - t_total) / 60.0

    # -- Aggregate summary --
    summary = {
        'images':      all_results,
        'total_images': len(all_results),
        'total_time_min': round(total_min, 1),
    }
    summary_path = MULTI_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("  MULTI-IMAGE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Image':<30}  {'Py':>4}  {'CSim':>5}  {'Match':>6}  {'%':>6}  {'Time':>6}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in all_results:
        print(f"  {r['image']:<30}  {r['python_count']:>4}  {r['csim_count']:>5}  "
              f"{r['matched']:>6}  {r['match_pct']:>5.1f}%  {r['csim_time_min']:>5.1f}m")
    print(f"\n  Total time: {total_min:.1f} min")
    print(f"  Results:    {MULTI_DIR}")
    print(f"  Summary:    {summary_path}")


if __name__ == '__main__':
    main()
