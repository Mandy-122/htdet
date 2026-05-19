# Low GFLOPs Model - Actual Computational Reduction

## GFLOPs Comparison

| Model | Parameters | GFLOPs | Reduction |
|-------|------------|--------|-----------|
| **Original (htdet_gpu.py)** | 12.411 M | 198.876 | - |
| **Low GFLOPs (htdet_gpu_low_gflops.py)** | 6.899 M | 59.903 | **69.9%** |

## Changes Made

| Component | Original | New | Reduction |
|-----------|----------|-----|-----------|
| **FPN out_channels** | 256 | 128 | 50% |
| **Head feat_channels** | 256 | 128 | 50% |
| **Backbone** | MobileViT-S | MobileViT-S | Same |

## Why This Reduces GFLOPs

Unlike channel pruning (which zeros weights), this config **actually changes the architecture**:

### FPN (Feature Pyramid Network)
- **Before:** 5 levels × 256 channels = 1,280 feature maps
- **After:** 5 levels × 128 channels = 640 feature maps
- **Conv FLOPs:** Proportional to `Cin × Cout` - halving both gives 4× reduction

### RetinaHead
- **Before:** 4 stacked convs × 256 channels × 5 levels
- **After:** 4 stacked convs × 128 channels × 5 levels
- **FLOPs reduction:** ~4× for head computation

## Training

### From Scratch
```bash
python tools/train.py configs/htdet/htdet_gpu_low_gflops.py \
    --work-dir work_dirs/htdet_low_gflops
```

### Transfer Learning (Recommended)
Since backbone is the same, you can load pretrained weights:
```bash
python tools/train.py configs/htdet/htdet_gpu_low_gflops.py \
    --load-from work_dirs/htdet_mobilevit_April21st_2/epoch_42.pth \
    --work-dir work_dirs/htdet_low_gflops
```

Note: The FPN and head weights will be randomly initialized since their dimensions changed.

## Expected Accuracy

| Model | Expected mAP | Notes |
|-------|--------------|-------|
| Original | 0.414 | Baseline |
| Low GFLOPs (initial) | 0.30-0.35 | Head/FPN randomly initialized |
| Low GFLOPs (after FT) | 0.38-0.40 | After fine-tuning |

The smaller head/FPN may cause slight accuracy drop, but 128 channels is usually sufficient for most detection tasks.

## Speed Comparison

| Metric | Original | Low GFLOPs | Improvement |
|--------|----------|------------|-------------|
| GFLOPs | 198.9 | 59.9 | **3.3× faster** |
| Params | 12.4 M | 6.9 M | **1.8× smaller** |
| Inference (est.) | ~10 ms | ~3-4 ms | **~3× faster** |

*Inference time estimates depend on GPU and batch size*

## File Location

- **Config:** `configs/htdet/htdet_gpu_low_gflops.py`
- **Work dir:** `work_dirs/htdet_low_gflops/`

## Comparison with Channel Pruning

| Aspect | Channel Pruning | Low GFLOPs Config |
|--------|-----------------|-------------------|
| GFLOPs | Same (198.9) | Reduced (59.9) |
| Params (stored) | Same (12.4M) | Reduced (6.9M) |
| Active params | Reduced (~8.7M) | Reduced (6.9M) |
| Architecture | Same | Changed |
| Can load original checkpoint | Yes | Partial (backbone only) |
| Speedup | Only on sparse HW | On any HW |
| Accuracy impact | Low (with FT) | Low-Medium |

## When to Use Each

**Use Channel Pruning when:**
- You want to keep the same architecture
- You have sparse computation hardware
- You need gradual model compression

**Use Low GFLOPs Config when:**
- You need actual speedup on any hardware
- You're deploying to edge devices
- You can afford to train from scratch or fine-tune

## References

1. Lin, T.Y. et al. "Focal Loss for Dense Object Detection" (RetinaNet paper)
2. MobileViT: "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
