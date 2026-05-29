"""
dump_head_p3.py
Dumps cls_feat, reg_feat (after stacked convs) and cls_logits, reg_deltas
(pred conv outputs) for the P3 FPN level.

Saves:
  csim_validation/reg_feat_p3_python.bin   float32 [256, 80, 80]
  csim_validation/reg_deltas_p3_python.bin float32 [36, 80, 80]
  csim_validation/cls_feat_p3_python.bin   float32 [256, 80, 80]
  csim_validation/cls_logits_p3_python.bin float32 [36, 80, 80]

Run from mmdetection/ root:
  python csim_validation/dump_head_p3.py
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

CFG_PATH  = 'configs/htdet/htdet_gpu.py'
CKPT_PATH = 'work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth'
IMAGE_BIN = 'csim_validation/input_image.bin'
OUT_DIR   = 'csim_validation'

print("Loading model...")
cfg = Config.fromfile(CFG_PATH)
cfg.model.pretrained = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT_PATH, map_location='cpu')
model.eval()

img = np.fromfile(IMAGE_BIN, dtype=np.float32)
assert img.size == 3*640*640
x = torch.from_numpy(img).view(1, 3, 640, 640)

# ---------- collect FPN features via hook ----------
fpn_feats = []
def fpn_hook(m, inp, out):
    for o in out:
        fpn_feats.append(o.detach().float().cpu())
model.neck.register_forward_hook(fpn_hook)

print("Running full forward pass...")
with torch.no_grad():
    model(return_loss=False, img=[x],
          img_metas=[[dict(img_shape=(640,640,3), ori_shape=(640,640,3),
                           pad_shape=(640,640,3), scale_factor=1.0,
                           flip=False, batch_input_shape=(640,640))]])

# P3 is index 1 in [P2, P3, P4, P5, P6]
p3 = fpn_feats[1]  # [1, 256, 80, 80]
print(f"P3 shape: {p3.shape}, first5: {p3.flatten()[:5].tolist()}")

# ---------- run head manually on P3 ----------
head = model.bbox_head

def run_stacked_convs(feat, conv_list):
    x = feat
    for conv_module in conv_list:
        x = conv_module(x)
    return x

with torch.no_grad():
    cls_feat = run_stacked_convs(p3, head.cls_convs)  # [1,256,80,80]
    reg_feat = run_stacked_convs(p3, head.reg_convs)  # [1,256,80,80]
    cls_logits = head.retina_cls(cls_feat)             # [1,36,80,80]
    reg_deltas = head.retina_reg(reg_feat)             # [1,36,80,80]

# squeeze batch dim, save as [C,H,W]
def save_arr(arr_t, name):
    arr = arr_t.squeeze(0).numpy().astype(np.float32)
    path = os.path.join(OUT_DIR, name)
    arr.tofile(path)
    print(f"  {name}: shape={arr.shape}  min={arr.min():.4f}  max={arr.max():.4f}"
          f"  mean={arr.mean():.4f}  first5={arr.flatten()[:5].tolist()}")

print("\nDumping arrays:")
save_arr(cls_feat,   'cls_feat_p3_python.bin')
save_arr(reg_feat,   'reg_feat_p3_python.bin')
save_arr(cls_logits, 'cls_logits_p3_python.bin')
save_arr(reg_deltas, 'reg_deltas_p3_python.bin')

# ---------- Also check score distribution at P3 ----------
scores_per_cls = torch.sigmoid(cls_logits)  # [1,36,80,80]
print(f"\nP3 cls scores (after sigmoid): min={scores_per_cls.min():.4f} "
      f"max={scores_per_cls.max():.4f} mean={scores_per_cls.mean():.4f}")
above_thr = (scores_per_cls > 0.05).sum().item()
print(f"  Locations with score > 0.05: {above_thr}")

print(f"\nP3 reg_deltas: min={reg_deltas.min():.4f} max={reg_deltas.max():.4f}"
      f" mean={reg_deltas.mean():.4f}")
print(f"  dh channel (ch 3,7,11,...): range for ch3 = "
      f"[{reg_deltas[0,3].min():.4f}, {reg_deltas[0,3].max():.4f}]")

# ---------- decode a few boxes to verify they look right ----------
from mmdet.core import build_anchor_generator, build_bbox_coder
anchor_gen = head.anchor_generator
stride_8_anchors = anchor_gen.single_level_grid_priors((80,80), 1, device='cpu')
print(f"\nP3 anchors shape: {stride_8_anchors.shape}")
print(f"First 9 anchors (stride=8): {stride_8_anchors[:9].numpy().tolist()}")

print("\nDone.")
