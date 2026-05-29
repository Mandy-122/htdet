# -*- coding: utf-8 -*-
"""
detect_python.py
----------------
Run HTDet object detection on one image using the PyTorch model,
then save the preprocessed image tensor for C simulation comparison.

Preprocessing matches the training test pipeline from urpc_detection.py:
  1. BGR→RGB
  2. Resize keeping aspect ratio so max(H,W) ≤ INPUT_H/W (= 640)
  3. Normalize with ImageNet mean/std
  4. Pad to INPUT_H × INPUT_W with zeros

Outputs (all in csim_validation/):
  input_image.txt       -- preprocessed CHW float32 values, one per line
  input_image.bin       -- same data as binary float32 (for testbench.cpp)
  python_detections.txt -- detection results (class_id score x1 y1 x2 y2)
  vis_output.jpg        -- visualised detections drawn on original image

Run from the csim_validation/ directory:
  cd csim_validation
  python detect_python.py
"""

import sys
import os

MMDET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MMDET_ROOT)

import cv2
import numpy as np
import torch

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

# ----------------------------------------------------------------
# Configuration — edit these paths if needed
# ----------------------------------------------------------------
CONFIG     = os.path.join(MMDET_ROOT, 'configs/htdet/htdet_gpu.py')
CHECKPOINT = os.path.join(MMDET_ROOT, 'work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth')
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MMDET_ROOT, 'data/urpc/val2018/images/CHN083846_0270.jpg')

# Must match fpga_types.h  INPUT_H / INPUT_W
INPUT_H = 640
INPUT_W = 640

# Match fpga_types.h  SCORE_THR
SCORE_THR = 0.05

# ImageNet normalisation — same as training pipeline (to_rgb=True)
MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)

CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------
# Step 1: Preprocess image — matches training test pipeline
#   Resize(keep_ratio, max_side=640) → Normalize → Pad(to 640×640)
# ----------------------------------------------------------------
def preprocess(img_path, H, W):
    """
    Load BGR image, convert to RGB, resize keeping aspect ratio so that
    max(h, w) ≤ H (= W = 640), normalize with ImageNet mean/std, then
    zero-pad to exactly H×W.

    Returns:
        img_chw   -- float32 ndarray [3, H, W]  (the FPGA input)
        img_orig  -- uint8  ndarray [h_orig, w_orig, 3] BGR (for vis)
        scale     -- float  scale applied during resize
        new_h     -- int    height after resize (before padding)
        new_w     -- int    width  after resize (before padding)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_orig = img_bgr.copy()
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h_orig, w_orig = img_rgb.shape[:2]
    scale  = min(H / h_orig, W / w_orig)
    new_h  = round(h_orig * scale)
    new_w  = round(w_orig * scale)

    img_res  = cv2.resize(img_rgb, (new_w, new_h)).astype(np.float32)  # HWC
    img_norm = (img_res - MEAN) / STD

    # Zero-pad to H×W (padded region has value 0 after norm, ~= background)
    img_pad = np.zeros((H, W, 3), dtype=np.float32)
    img_pad[:new_h, :new_w, :] = img_norm
    img_chw = img_pad.transpose(2, 0, 1)   # CHW

    return img_chw.astype(np.float32), img_orig, scale, new_h, new_w


# ----------------------------------------------------------------
# Step 2: Save image tensor for C simulation
# ----------------------------------------------------------------
def save_image_tensor(img_chw, txt_path, bin_path):
    flat = img_chw.flatten()   # C-order: channel × row × col

    # Text file: one float value per line (matches C sim read loop)
    with open(txt_path, 'w') as f:
        for v in flat:
            f.write(f'{v:.8f}\n')
    print(f"[save] {txt_path}  ({len(flat)} values)")

    # Binary float32 (for testbench.cpp  load_image_f32)
    flat.astype(np.float32).tofile(bin_path)
    print(f"[save] {bin_path}  ({flat.nbytes} bytes)")


# ----------------------------------------------------------------
# Step 3: Run inference through PyTorch model
# ----------------------------------------------------------------
def build_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def run_inference(model, img_chw, H, W, scale, new_h, new_w, device):
    """
    Feed a single preprocessed CHW float32 array through the model.
    img_meta reflects the actual resize+pad so MMDetection's decoder
    uses the correct anchor stride-to-pixel mapping.
    Returns list of (class_name, score, x1, y1, x2, y2, cls_id).
    Coordinates are in the H×W (640×640) preprocessed image space.
    """
    img_tensor = torch.from_numpy(img_chw[np.newaxis]).float().to(device)

    img_meta = {
        'img_shape'    : (new_h, new_w, 3),          # valid region after resize
        'ori_shape'    : (new_h, new_w, 3),
        'pad_shape'    : (H, W, 3),                   # full padded tensor shape
        'scale_factor' : np.array([scale, scale, scale, scale], dtype=np.float32),
        'flip'         : False,
        'flip_direction': None,
        'filename'     : IMAGE_PATH,
    }

    with torch.no_grad():
        # simple_test returns list[list[ndarray]]  shape [num_classes][N,5]
        results = model.simple_test(img_tensor, [img_meta], rescale=False)

    detections = []
    bbox_results = results[0]
    for cls_id, bboxes in enumerate(bbox_results):
        if bboxes is None or len(bboxes) == 0:
            continue
        for det in bboxes:
            x1, y1, x2, y2, score = det
            if score >= SCORE_THR:
                detections.append((CLASS_NAMES[cls_id], float(score),
                                   float(x1), float(y1), float(x2), float(y2),
                                   cls_id))
    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


# ----------------------------------------------------------------
# Step 4: Save detections
# ----------------------------------------------------------------
def save_detections(detections, path):
    with open(path, 'w') as f:
        f.write(f'{len(detections)}\n')
        for cls_name, score, x1, y1, x2, y2, cls_id in detections:
            f.write(f'{cls_id} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')

    print(f"\n[save] {path}")
    print(f"       {len(detections)} detection(s):")
    for cls_name, score, x1, y1, x2, y2, _ in detections:
        print(f"  {cls_name:<14}  score={score:.4f}  "
              f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


# ----------------------------------------------------------------
# Step 5: Visualise
# ----------------------------------------------------------------
COLORS = {
    'holothurian': (0, 0, 255),
    'echinus'    : (0, 255, 0),
    'scallop'    : (255, 0, 0),
    'starfish'   : (0, 255, 255),
}

def visualise(img_orig, detections, out_path, scale, new_h, new_w):
    """
    Draw detections on the original image.
    Detections are in the 640×640 padded space; undo the keep-ratio
    scale to map back to the original image coordinates.
    """
    h_orig, w_orig = img_orig.shape[:2]

    vis = img_orig.copy()
    for cls_name, score, x1, y1, x2, y2, _ in detections:
        # Clamp to valid region (exclude padding zone)
        x1 = min(x1, new_w); x2 = min(x2, new_w)
        y1 = min(y1, new_h); y2 = min(y2, new_h)
        # Scale back to original image space
        x1s = int(x1 / scale); y1s = int(y1 / scale)
        x2s = int(x2 / scale); y2s = int(y2 / scale)
        color = COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(vis, (x1s, y1s), (x2s, y2s), color, 2)
        label = f'{cls_name} {score:.2f}'
        cv2.putText(vis, label, (x1s, max(y1s - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"[save] {out_path}")


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Image : {IMAGE_PATH}")
    print(f"Input size: {INPUT_H}×{INPUT_W}  (keep_ratio resize + zero-pad)")
    print()

    # 1. Preprocess (keep_ratio + pad — matches training test pipeline)
    img_chw, img_orig, scale, new_h, new_w = preprocess(IMAGE_PATH, INPUT_H, INPUT_W)
    print(f"Preprocessed tensor: shape={img_chw.shape}  "
          f"valid_region={new_h}×{new_w}  scale={scale:.4f}")
    print(f"  value range: min={img_chw.min():.3f}  max={img_chw.max():.3f}")

    # 2. Save for C simulation
    save_image_tensor(
        img_chw,
        os.path.join(OUT_DIR, 'input_image.txt'),
        os.path.join(OUT_DIR, 'input_image.bin'),
    )

    # 3. Build model and infer
    print("\nLoading model ...")
    model = build_model(CONFIG, CHECKPOINT, device)
    print("Running inference ...")
    detections = run_inference(model, img_chw, INPUT_H, INPUT_W,
                               scale, new_h, new_w, device)

    # 4. Save detections
    save_detections(detections, os.path.join(OUT_DIR, 'python_detections.txt'))

    # 5. Visualise
    visualise(img_orig, detections,
              os.path.join(OUT_DIR, 'vis_output.jpg'),
              scale, new_h, new_w)

    print("\nDone. Files written to:", OUT_DIR)
    print("  input_image.txt       <- feed this to C simulation (one value/line)")
    print("  input_image.bin       <- binary version for testbench.cpp")
    print("  python_detections.txt <- reference detections for compare_results.py")
    print("  vis_output.jpg        <- visualised detections")
