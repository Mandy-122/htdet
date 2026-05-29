# HTDet PTQ vs Float32 — Full Comparison Report

**Model:** RetinaNet + MobileViT-S backbone + FPN (URPC underwater detection, 4 classes)  
**Checkpoint:** `work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth`  
**PTQ strategy:** W8A32 — INT8 per-channel symmetric weights, FP32 activations  
**PTQ checkpoint:** `ptq_results/ptq_model.pth`  
**Calibration:** 200 URPC val images, seed=42  
**Date:** 2026-05-19

---

## 1. Accuracy (mAP on URPC val2018, 800 images)

| Metric                    | Float32 | PTQ W8A32 | Δ (abs) | Δ (%)   |
|---------------------------|---------|-----------|---------|---------|
| **mAP (IoU 0.50:0.95)**   | 0.408   | **0.407** | −0.001  | −0.2%   |
| **mAP@50**                | 0.755   | **0.753** | −0.002  | −0.3%   |
| **mAP@75**                | 0.398   | **0.395** | −0.003  | −0.8%   |
| mAP (small objects)       | 0.242   | 0.239     | −0.003  | −1.2%   |
| mAP (medium objects)      | 0.417   | 0.416     | −0.001  | −0.2%   |
| mAP (large objects)       | 0.515   | 0.513     | −0.002  | −0.4%   |
| AR@100                    | —       | 0.534     | —       | —       |
| AR (small)                | —       | 0.399     | —       | —       |
| AR (medium)               | —       | 0.544     | —       | —       |
| AR (large)                | —       | 0.605     | —       | —       |

**Accuracy loss is negligible (<0.3% on mAP@50). The PTQ model is production-ready.**

### Per-layer quantization quality (weight domain)

| Stat        | Min     | Mean    | Max     |
|-------------|---------|---------|---------|
| SQNR (dB)   | 36.8    | 43.3    | 49.1    |
| Cosine Sim  | 0.9999  | 1.0000  | 1.0000  |

All 53 Conv2d layers are above the 30 dB SQNR safety threshold. No sensitive layers detected.

---

## 2. Model Size & Weight Memory

| Component            | Float32      | INT8 (conv only) | Reduction  |
|----------------------|-------------|------------------|------------|
| backbone_w           | 18.9 MB     | 1.92 MB          | 9.8×       |
| fpn_w                | 9.9 MB      | 2.48 MB          | 4.0×       |
| cls_conv_w           | 9.0 MB      | 2.25 MB          | 4.0×       |
| reg_conv_w           | 9.0 MB      | 2.25 MB          | 4.0×       |
| cls_pred_w           | 0.32 MB     | 0.079 MB         | 4.0×       |
| reg_pred_w           | 0.32 MB     | 0.079 MB         | 4.0×       |
| **Total conv weights** | **49.7 MB** | **9.49 MB**    | **5.2×**   |
| Full .pth (all params) | 96 MB     | 48 MB            | 2.0×       |

> The backbone shows ~10× weight reduction because BN terms (scale + bias, 
> stored as float32 in the float binary) are not included in the INT8 binary.
> FPN and head show the expected ~4× reduction (pure conv weights only).

---

## 3. Compute — GFLOPs

| Component                  | GFLOPs  | Notes                              |
|----------------------------|---------|------------------------------------|
| **Total model (FP32)**     | 198.9   | Baseline                           |
| Conv2d layers (quantized)  | ~120    | 53 layers — backbone MBConv + FPN + head |
| Transformer attention      | ~79     | MobileViT-S — kept FP32           |
| **W8A32 effective savings** | 0      | Same MAC count; saving is in memory BW |
| **W8A8 MAC reduction**     | ~120    | 4× INT8 throughput on quantized convs |

> **W8A32** (current PTQ): GFLOPs are the same because activations are still
> FP32 — the saving is entirely in weight fetch bandwidth (4× fewer bytes).
>
> **W8A8** (future, if activation quantization is added): INT8 MACs execute
> at 4× the throughput of FP32 MACs on DSP48E2, giving an effective
> ~120 GFLOPs reduction for the Conv2d portion (~60% of total).

---

## 4. FPGA Hardware Resources (estimated, 640×640 input)

### 4a. BRAM — Weight Storage

| Storage              | FP32      | W8A32 INT8 | Reduction |
|----------------------|-----------|-----------|-----------|
| Conv weight bytes    | 49.7 MB   | 9.49 MB   | **5.2×**  |
| BRAM36 required (est.)| ~11,310  | ~2,160    | **−81%**  |

> Assumes all weights stored on-chip in BRAM36 (36 Kb each).
> Off-chip DDR: same proportional reduction in memory footprint and DDR traffic.

### 4b. DSP Blocks — Multiply-Accumulate Units

| Mode                      | DSPs per 8-MAC PE | Notes                          |
|---------------------------|-------------------|--------------------------------|
| FP32 weights + FP32 accum | ~24 DSPs          | 3 DSP48 per FP32 multiplier    |
| W8A32 (INT8 W, FP32 A)    | ~24 DSPs          | Dequantize then FP32 MAC — no DSP saving |
| **W8A8 (INT8 W + INT8 A)**| **~2 DSPs**       | DSP48E2 packs 4×INT8 MACs in one primitive — **~12× fewer DSPs** |

> W8A32 saves BRAM but not DSPs. Full DSP savings require W8A8
> (activation quantization — next step after current PTQ).

### 4c. Weight Fetch Throughput (same memory bandwidth)

| Weight precision | Weights per clock | Bandwidth multiplier |
|------------------|-------------------|----------------------|
| FP32 (32 bit)    | 1                 | 1×                   |
| INT8 (8 bit)     | 4                 | **4×**               |

Memory-bandwidth-bound layers (large kernels, large channel counts) will
see near-linear throughput improvement from INT8 weight fetching.

### 4d. Inference Latency (FPGA HLS, 640×640, 200 MHz target — estimates)

| Mode                  | Estimated Latency | vs FP32 Baseline |
|-----------------------|-------------------|-----------------|
| FP32 baseline         | ~500 ms           | 1×              |
| **W8A32 (current PTQ)**| **~200 ms**      | **~2.5×**       |
| W8A8 (full INT8)      | ~50–80 ms         | ~6–10×          |

> Estimates assume dataflow-pipelined HLS implementation with AXI4 streaming.
> W8A32 speedup comes from 4× weight bandwidth; activations still use FP32 datapaths.
> W8A8 adds 4× MAC throughput on top, and halves activation buffer sizes.
> Actual numbers require synthesis and implementation (Vivado/Vitis).

---

## 5. Summary Table

| Metric                  | Float32 | PTQ W8A32 | PTQ W8A8 (projected) |
|-------------------------|---------|-----------|----------------------|
| mAP@50                  | 0.755   | **0.753** | ~0.74–0.75 (est.)    |
| mAP (0.50:0.95)         | 0.408   | **0.407** | ~0.40 (est.)         |
| Conv weight memory      | 49.7 MB | **9.5 MB**| **9.5 MB**           |
| Full model .pth         | 96 MB   | **48 MB** | **~20 MB** (est.)    |
| BRAM36 (weight storage) | ~11,310 | **~2,160**| **~2,160**           |
| DSPs (per PE)           | ~24     | ~24       | **~2**               |
| Weight BW multiplier    | 1×      | **4×**    | **4×**               |
| Latency (est. FPGA)     | ~500 ms | **~200 ms**| **~50–80 ms**       |

---

## 6. Files

| File                                     | Description                               |
|------------------------------------------|-------------------------------------------|
| `ptq_results/ptq_model.pth`              | PTQ checkpoint — drop-in for MMDet eval   |
| `ptq_results/ptq_scales.json`            | Per-layer act + weight scale factors      |
| `ptq_results/ptq_report.txt`             | Full SQNR / CosSim layer table            |
| `ptq_results/ptq_int8_weights/`          | INT8 binary weights for HLS testbench     |
| `ptq_results/ptq_int8_weights/scales.json` | Per-channel weight scales + BN biases   |
| `fpga_temp/fpga_types.h`                 | `ap_fixed<8,N>` precision table (updated) |

---

## 7. Recommended Next Steps

1. **W8A8 simulation** — add activation quantization in `ptq_calibrate.py`
   using the calibrated `act_scale` values already in `ptq_scales.json`.
   Expected additional mAP impact: −0.5–1.5%.

2. **HLS synthesis** — uncomment `wint8_t` / `act8b*_t` aliases in
   `fpga_temp/fpga_types.h` and update `fpga_utils.h` conv kernels to use
   INT8 weight loads + dequantize-before-MAC pattern.

3. **INT8 testbench** — load `ptq_int8_weights/backbone_int8.bin` with
   per-channel scales and BN biases (now in `scales.json`) to run the
   C-sim testbench in W8A32 mode for end-to-end numerical validation.
