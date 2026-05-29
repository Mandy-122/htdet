import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

# -----------------------------
# CONFIG
# -----------------------------
CONFIG_FILE = 'configs/htdet/retinanet_mobilevit_pruning.py'
CHECKPOINT = 'work_dirs/htdet_mobilevit_April21st_2/epoch_42.pth'
SAVE_PATH = 'work_dirs/pruned_model_epoch42.pth'

PRUNE_RATIO = 0.3   # 30% channels

# -----------------------------
# BUILD MODEL
# -----------------------------
cfg = Config.fromfile(CONFIG_FILE)

model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

# load weights
load_checkpoint(model, CHECKPOINT, map_location='cpu')

model.eval()

# -----------------------------
# PRUNING FUNCTION
# -----------------------------
def prune_conv_layer(layer, amount):
    if isinstance(layer, nn.Conv2d):
        # skip final detection heads (important!)
        return prune.ln_structured(
            layer,
            name="weight",
            amount=amount,
            n=1,
            dim=0  # prune output channels
        )

# -----------------------------
# APPLY PRUNING
# -----------------------------
for name, module in model.named_modules():
    
    # Skip classification & regression heads
    if 'retina_cls' in name or 'retina_reg' in name:
        continue

    if isinstance(module, nn.Conv2d):
        try:
            prune_conv_layer(module, PRUNE_RATIO)
            print(f"Pruned: {name}")
        except Exception as e:
            print(f"Skipped: {name} ({e})")

# -----------------------------
# MAKE PRUNING PERMANENT
# -----------------------------
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        try:
            prune.remove(module, 'weight')
        except:
            pass

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save({'state_dict': model.state_dict()}, SAVE_PATH)

print(f"\n✅ Pruned model saved at: {SAVE_PATH}")