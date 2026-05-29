# How to Further Save GFLOPs and Memory

**Current state:** MobileViT-S + FPN(256) + RetinaHead(4 convs, 256ch)  
**Baseline:** 198.9 GFLOPs, 12.4M params, W8A32 PTQ done (−0.2% mAP)  
**W8A32 PTQ already achieved:** conv weight memory 49.7MB → 9.5MB (5.2×), latency ~2.5×

Techniques below are ordered from easiest to hardest.

---

## 1. W8A8 — Full INT8 (Quantize Activations Too)

**GFLOPs count: unchanged | Activation memory: 4× smaller | Latency: ~6–10× vs FP32**

Natural next step after W8A32. The `act_scale` values are already computed and
stored in `ptq_results/ptq_scales.json` from the calibration run. Only needs
activation fake-quantization added to `ptq_calibrate.py`.

**What changes over W8A32:**
- Activation feature maps shrink 4× → major BRAM savings on FPGA
- DSP48E2 primitives pack 4 INT8 MACs per DSP vs 3 DSPs for a single FP32 multiply → ~12× fewer DSPs per PE
- Estimated FPGA latency: ~50–80 ms vs ~200 ms for W8A32

**Caution:** Two early depthwise convolutions have activation ranges up to ±232
(stages_0.0.conv2_kxk, stages_1.0.conv2_kxk). These need `ap_fixed<8,9>` which
leaves only 1 fractional bit — very coarse. Keep these two layers in FP32
(mixed precision) to avoid accuracy loss.

**Expected mAP impact:** −0.5 to −1.5% additional drop on top of W8A32.

---

## 2. Reduce FPN + Head Channels: 256 → 128

**GFLOPs: 198.9 → 59.9 (69.9% reduction) | Params: 12.4M → 6.9M | Config exists**

Config already available: `configs/htdet/htdet_gpu_low_gflops.py`

FPN and RetinaNet head account for ~70% of total GFLOPs because they run 3×3
convolutions across 5 pyramid levels simultaneously. FLOPs scale with
Cin × Cout — halving both from 256 to 128 gives a 4× reduction in those layers.

| Model | GFLOPs | Params | Expected mAP |
|-------|--------|--------|--------------|
| Current (256ch) | 198.9 | 12.4M | 0.408 |
| Low GFLOPs (128ch) | **59.9** | **6.9M** | ~0.38–0.40 |

Backbone weights are reused unchanged (same architecture). Only FPN and head
need to be trained from scratch or fine-tuned (~10–20 epochs).

---

## 3. Structured Channel Pruning

**GFLOPs saved: ~20–40% on top of current | No architecture change | Script exists**

`tools/analysis_tools/channel_prune.py` is already in the project.
Removes entire output channels ranked by L1-norm (smallest magnitude = least useful).
Unlike unstructured pruning (zeroing individual weights), structured pruning gives
real speedup on any hardware because you are literally removing rows/columns from
weight matrices.

```bash
python tools/analysis_tools/channel_prune.py \
    configs/htdet/htdet_gpu.py \
    work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth \
    --prune-ratio 0.3 \
    --out-dir work_dirs/pruned_models
```

A 30% pruning ratio typically costs 0.5–1.5% mAP without fine-tuning,
and recovers to near-original with 5–10 epochs of fine-tuning.

---

## 4. Reduce Input Resolution

**GFLOPs: proportional to resolution² | No retraining needed to test**

GFLOPs scale with the square of input resolution — halving one side quarters
the spatial feature map sizes across all backbone stages and FPN levels.

| Input | GFLOPs (est.) | Memory | mAP impact |
|-------|--------------|--------|------------|
| 640×640 (current) | 198.9 | 1× | baseline |
| 512×512 | ~127 | 0.64× | −1 to −2% |
| 416×416 | ~84  | 0.42× | −2 to −4% |
| 320×320 | ~50  | 0.25× | −4 to −8% |

For underwater objects that tend to be medium-to-large in frame, 512×512 is
usually safe. Configs for lower resolutions already exist:
`htdet_gpu_low_gflops_192.py`, `htdet_gpu_low_gflops_224.py`.

---

## 5. Reduce Stacked Convolutions in Head: 4 → 2

**GFLOPs saved: ~50% of head compute | Simple one-line config change**

The RetinaNet head runs 4 stacked 3×3 convolutions per branch (cls and reg),
applied across all 5 FPN levels. Each conv is 256×256×3×3 — expensive.
Reducing to 2 stacked convs halves head GFLOPs with minimal accuracy loss
when the backbone is already well-pretrained (MobileViT-S already captures
rich features before the head).

```python
# In config:
stacked_convs=2,  # was 4
```

Expected mAP impact: −0.5 to −1.5%. No retraining needed (just re-evaluate),
or fine-tune 5 epochs to recover.

---

## 6. Knowledge Distillation

**GFLOPs: depends on student | Best accuracy recovery for aggressive compression**

Train a smaller student model (e.g. the 59.9 GFLOPs 128-channel config) using
the current full 198.9 GFLOPs model as a teacher. The student learns to mimic:
- The teacher's FPN feature maps (intermediate supervision)
- The teacher's classification logits (soft labels carry more information than hard GT labels)

This recovers 1–3% mAP compared to training the student from scratch — important
when you have already made large architectural reductions that cause accuracy gaps.

Most effort (requires a custom training loop with teacher-student loss) but gives
the best accuracy-efficiency trade-off for aggressive compression targets.

---

## Combined Strategy — Recommended for FPGA Deployment

| Step | Change | GFLOPs | Conv Weight Mem | Est. FPGA Latency | mAP impact |
|------|--------|--------|----------------|-------------------|------------|
| Baseline (float) | — | 198.9 | 49.7 MB | ~500 ms | 0.408 |
| Step 1 (done) | W8A32 PTQ | 198.9 | 9.5 MB | ~200 ms | 0.407 (−0.2%) |
| Step 2 | W8A8 | 198.9 | 9.5 MB | ~50–80 ms | ~0.40 (−0.5%) |
| Step 3 | 256→128 channels | **59.9** | **~2.4 MB** | ~15–25 ms | ~0.38–0.40 |
| Step 4 (optional) | 30% pruning | ~45 | ~1.8 MB | ~12 ms | ~0.37–0.39 |

**Final combined effect (Steps 1–3):**
- GFLOPs: 198.9 → ~60 (**3.3× reduction**)
- Conv weight memory: 49.7 MB → ~2.4 MB (**~20× reduction**)
- Estimated FPGA latency: ~500 ms → **~15–25 ms**
- Expected mAP: ~0.38–0.40 (acceptable for URPC real-time deployment)

---

## Decision Guide

| Goal | Best technique |
|------|---------------|
| Maximum accuracy preservation | W8A8 (step 2) — keeps architecture identical |
| Maximum GFLOPs reduction | Reduce FPN/head channels 256→128 (step 3) |
| No retraining allowed | W8A8 + reduce resolution |
| Smallest FPGA footprint | All steps combined (1–4) |
| Fast to implement | Step 2 (act_scale already in ptq_scales.json) |
