# Performance Improvement Recommendations for HTDet

## Current Performance Analysis

**Best Result (Epoch 42, MobileViT-S + RetinaNet):**
```
bbox_mAP:     0.4142  ← Current baseline
bbox_mAP_50:  0.7623
bbox_mAP_75:  0.4119
bbox_mAP_s:   0.2521  ← Weakest area (small objects)
bbox_mAP_m:   0.4241
bbox_mAP_l:   0.5167
```

**Training Trajectory:**
- Peak performance at epoch 42
- Gradual decline after epoch 45 (overfitting)
- Final epoch 60: 0.408 mAP (slight degradation)

---

## Key Issues Identified

1. **Small Object Detection (bbox_mAP_s = 0.2521)**
   - Significantly lower than medium/large
   - 416x416 input resolution is too small
   - Need better multi-scale training

2. **Architecture Limitations**
   - MobileViT v1 is older architecture (2021)
   - Newer efficient ViTs available with better accuracy/speed

3. **Training Configuration**
   - Fixed 416x416 resolution limits small object detection
   - No advanced augmentations (Mosaic, MixUp)
   - Step LR schedule less optimal than Cosine

---

## Recommended Configurations (Created)

### 1. **MobileViT v2 + RetinaNet** (`htdet_mobilevitv2_retina.py`)
**Expected mAP: 0.43-0.45**

Key improvements:
- MobileViT v2 architecture (2023) - improved attention blocks
- Multi-scale training (640-800) for better small object detection
- Cosine Annealing LR schedule
- Higher evaluation resolution (800x800)

```bash
python tools/train.py configs/htdet/htdet_mobilevitv2_retina.py
```

---

### 2. **TinyViT + RetinaNet** (`htdet_tinyvit_retina.py`) ⭐ RECOMMENDED
**Expected mAP: 0.44-0.47**

Key improvements:
- Microsoft's TinyViT (21M) - SOTA efficient ViT
- Better accuracy than MobileViT at similar speed
- AdamW optimizer with proper ViT hyperparameters
- IoU loss for better bounding box localization
- Higher resolution training (768-896)

```bash
python tools/train.py configs/htdet/htdet_tinyvit_retina.py
```

---

### 3. **MaxViT + RetinaNet** (`htdet_maxvit_retina.py`)
**Expected mAP: 0.43-0.46**

Key improvements:
- MaxViT with block attention + MBConv
- Excellent for dense prediction tasks
- TF-trained weights (high quality)

```bash
python tools/train.py configs/htdet/htdet_maxvit_retina.py
```

---

### 4. **MobileViT v2 + DETR** (`htdet_mobilevitv2_detr.py`)
**Expected mAP: 0.44-0.48**

Key improvements:
- Transformer-based detector (anchor-free)
- Hungarian matching for optimal assignment
- No anchor tuning needed
- Better for small objects

```bash
python tools/train.py configs/htdet/htdet_mobilevitv2_detr.py
```

---

## Quick Comparison Table

| Config | Backbone | Resolution | Expected mAP | Speed |
|--------|----------|------------|--------------|-------|
| Current | MobileViT-S | 416x416 | 0.414 | Fastest |
| MobileViT v2 | MobileViT v2 | 640-800 | 0.43-0.45 | Fast |
| TinyViT ⭐ | TinyViT 21M | 768-896 | 0.44-0.47 | Fast |
| MaxViT | MaxViT Tiny | 640-768 | 0.43-0.46 | Medium |
| DETR | MobileViT v2 | 640-800 | 0.44-0.48 | Medium |

---

## Additional Optimization Tips

### 1. **Early Stopping**
Save best model automatically:
```python
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP'  # Auto-saves best checkpoint
)
```

### 2. **Extended Training**
If models haven't converged:
- Increase `max_epochs` to 80-100
- Adjust LR schedule: `step=[60, 80]` or use CosineAnnealing

### 3. **Data Augmentations** (for small objects)
Add to `train_pipeline`:
```python
dict(
    type='Mosaic',
    img_scale=(800, 800),
    pad_val=114.0
),
```

### 4. **Test-Time Augmentation (TTA)**
For final evaluation only:
```python
# Enable flip testing
test_cfg = dict(
    ...
    flip=True,
    flip_keep_origin=True
)
```

### 5. **Class-Specific Analysis**
Check which classes have low AP:
```bash
python tools/test.py <config> <checkpoint> --eval bbox
```

---

## Training Commands

### Train with GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/htdet/htdet_tinyvit_retina.py
```

### Resume from checkpoint:
```bash
python tools/train.py configs/htdet/htdet_tinyvit_retina.py --resume_from work_dirs/htdet_tinyvit_retina/latest.pth
```

### Evaluate specific checkpoint:
```bash
python tools/test.py configs/htdet/htdet_tinyvit_retina.py work_dirs/htdet_tinyvit_retina/epoch_best.pth --eval bbox
```

---

## Migration Notes

### Channel Changes Required:
When switching backbones, update neck `in_channels`:

| Backbone | in_channels |
|----------|-------------|
| MobileViT-S | [64, 96, 128, 640] |
| MobileViT v2 | [96, 128, 192, 768] |
| TinyViT 21M | [80, 160, 320, 448] |
| MaxViT Tiny | [96, 192, 768] |

---

## Expected Outcomes

Based on the analysis:

1. **TinyViT** should give the best overall mAP (+3-5% improvement)
2. **MobileViT v2** offers good speed/accuracy tradeoff (+2-3% improvement)
3. **DETR-based** should improve small object detection significantly
4. All configs include multi-scale training which should boost bbox_mAP_s

---

## Next Steps

1. Start with **TinyViT** config (highest expected gain)
2. Train for 50 epochs
3. Compare validation results with current best (0.414 mAP)
4. If satisfied, retrain with seed for reproducibility
5. Consider ensemble if deploying for competition
