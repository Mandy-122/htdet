import torch
import torch.nn as nn
import torch_pruning as tp
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = 'configs/htdet/retinanet_mobilevit_pruning.py'
CHECKPOINT = 'work_dirs/pruned_mobilevit_retinanet/latest.pth'
SAVE_PATH = 'work_dirs/tp_pruned_model.pth'

PRUNE_RATIO = 0.3

# -----------------------------
# Build model
# -----------------------------
cfg = Config.fromfile(CONFIG)

model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

load_checkpoint(model, CHECKPOINT, map_location='cpu')
model.eval()

# -----------------------------
# Dummy input (VERY IMPORTANT)
# -----------------------------
example_inputs = torch.randn(1, 3, 640, 640)

# -----------------------------
# Ignore layers (IMPORTANT)
# -----------------------------
ignored_layers = []

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # skip final detection heads
        if m.out_channels == 36:  # retina cls/reg
            ignored_layers.append(m)

# -----------------------------
# Build pruner
# -----------------------------
imp = tp.importance.MagnitudeImportance(p=1)

pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=PRUNE_RATIO,
    ignored_layers=ignored_layers
)

# -----------------------------
# Prune
# -----------------------------
print("🔪 Pruning model...")
pruner.step()

# -----------------------------
# Save
# -----------------------------
torch.save({'state_dict': model.state_dict()}, SAVE_PATH)

print(f"\n✅ Torch-Pruned model saved at: {SAVE_PATH}")