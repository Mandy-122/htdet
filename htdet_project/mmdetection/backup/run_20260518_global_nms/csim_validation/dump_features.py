"""
dump_features.py
Extracts C1-C4 backbone features (and optionally P2-P6 FPN features) from
the PyTorch model and saves them as float32 binary files for comparison with
the C-sim outputs.

Usage:
  python3 dump_features.py
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

CFG_PATH   = 'configs/htdet/htdet_gpu.py'
CKPT_PATH  = 'work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth'
IMAGE_BIN  = 'csim_validation/input_image.bin'
OUT_DIR    = 'csim_validation'

# ---------- load model ----------
print("Loading model...")
cfg = Config.fromfile(CFG_PATH)
cfg.model.pretrained = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT_PATH, map_location='cpu')
model.eval()

# ---------- load image ----------
img = np.fromfile(IMAGE_BIN, dtype=np.float32)   # 3×640×640, pre-normalised
assert img.size == 3*640*640, f"Expected 1228800 floats, got {img.size}"
x = torch.from_numpy(img).view(1, 3, 640, 640)

# ---------- hook into backbone stages ----------
feats = {}

def make_hook(name):
    def hook(module, inp, out):
        # out may be a tensor or tuple; take the tensor
        t = out[0] if isinstance(out, (list, tuple)) else out
        feats[name] = t.detach().float().cpu().numpy()
    return hook

timm_model = model.backbone.model

# TIMM FeatureListNet: forward() returns a list [C1, C2, C3, C4, ...] depending on out_indices
# We hook the forward of model.backbone to get the feature list directly.
backbone_feats = []
def backbone_hook(module, inp, out):
    # out is a tuple/list of feature maps (one per out_index)
    for i, o in enumerate(out):
        backbone_feats.append(o.detach().float().cpu().numpy())

model.backbone.register_forward_hook(backbone_hook)

fpn_feats = []
def fpn_hook(module, inp, out):
    for i, o in enumerate(out):
        fpn_feats.append(o.detach().float().cpu().numpy())

model.neck.register_forward_hook(fpn_hook)

# ---------- run forward ----------
print("Running forward pass...")
with torch.no_grad():
    model(return_loss=False, img=[x],
          img_metas=[[dict(
              img_shape=(640,640,3), ori_shape=(640,640,3),
              pad_shape=(640,640,3), scale_factor=1.0,
              flip=False, batch_input_shape=(640,640))]])

# ---------- save ----------
names_bb = ['c1', 'c2', 'c3', 'c4']
print("\nBackbone features:")
for i, arr in enumerate(backbone_feats):
    name = names_bb[i] if i < len(names_bb) else f'c{i+1}'
    path = os.path.join(OUT_DIR, f'{name}_python.bin')
    arr.squeeze(0).tofile(path)   # remove batch dim → [C,H,W]
    print(f"  {name}: shape={arr.shape}  first5={arr.flatten()[:5].tolist()}")
    print(f"  saved → {path}")

names_fpn = ['p2', 'p3', 'p4', 'p5', 'p6']
print("\nFPN features:")
for i, arr in enumerate(fpn_feats):
    name = names_fpn[i] if i < len(names_fpn) else f'p{i+2}'
    path = os.path.join(OUT_DIR, f'{name}_python.bin')
    arr.squeeze(0).tofile(path)
    print(f"  {name}: shape={arr.shape}  first5={arr.flatten()[:5].tolist()}")
    print(f"  saved → {path}")
