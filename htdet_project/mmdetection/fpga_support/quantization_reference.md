# Quantization & Data Types Reference — HTDet FPGA Project

## 1. Floating-Point Family

Format: `[sign | exponent | mantissa]` (IEEE 754)

| Type        | Bits | Exponent | Mantissa | Range          | Used In                          |
|-------------|------|----------|----------|----------------|----------------------------------|
| FP32        | 32   | 8        | 23       | ±3.4×10³⁸     | Training, C-sim reference        |
| FP16        | 16   | 5        | 10       | ±65504         | GPU inference (NVIDIA)           |
| BF16        | 16   | 8        | 7        | ±3.4×10³⁸     | TPU training, A100/H100          |
| FP8 E4M3    | 8    | 4        | 3        | ±448           | NVIDIA H100 (forward pass)       |
| FP8 E5M2    | 8    | 5        | 2        | ±57344         | NVIDIA H100 (gradients)          |
| FP4         | 4    | 2        | 1        | very limited   | Experimental (Blackwell GPUs)    |

**BF16 vs FP16:** Same 16 bits but BF16 keeps FP32's exponent width → wider range,
less precision. Better for training (avoids gradient overflow). FP16 has more mantissa
bits → better precision for small inference values.

---

## 2. Integer Family

| Type   | Bits | Range         | Used In                                      |
|--------|------|---------------|----------------------------------------------|
| UINT8  | 8    | 0 → 255       | Activations after ReLU (always positive)     |
| INT8   | 8    | -128 → 127    | Most common inference format (TensorRT, etc) |
| INT4   | 4    | -8 → 7        | LLM weight compression (GPTQ, AWQ)           |
| INT2   | 2    | -2 → 1        | Extreme compression — research only          |
| INT1   | 1    | {0, 1}        | Binary networks (BitNet) — research only     |

---

## 3. Fixed-Point — `ap_fixed<W, I>` (Xilinx HLS)

```
ap_fixed<W, I, RND_MODE, OVF_MODE>
         |  |
         |  └── Integer bits (includes sign bit)
         └───── Total word length in bits
Fractional bits = W - I
Step size       = 1 / 2^(W-I)
Range           = -2^(I-1)  to  +2^(I-1) - step
```

**Rounding modes:** AP_TRN (truncate), AP_RND (round to nearest)
**Overflow modes:** AP_WRAP (wraps around), AP_SAT (clamps to max/min — use this)

Examples from this project's PTQ table:

| ap_fixed spec      | Int bits | Frac bits | Range        | Step    | Best for              |
|--------------------|----------|-----------|--------------|---------|------------------------|
| ap_fixed<8,  2>    | 2        | 6         | -2 → +1.98  | 0.0156  | retina_reg output      |
| ap_fixed<8,  3>    | 3        | 5         | -4 → +3.97  | 0.031   | conv_proj, small FPN   |
| ap_fixed<8,  5>    | 5        | 3         | -16 → +15.9 | 0.125   | most backbone 1×1      |
| ap_fixed<8,  6>    | 6        | 2         | -32 → +31.8 | 0.25    | fusion, retina_cls     |
| ap_fixed<8,  9>    | 9        | -1        | -256 → +255 | 2.0     | DW outliers (*)        |
| ap_fixed<16, 8>    | 8        | 8         | -128 → +128 | 0.0039  | weights (W8A32 mode)   |

(*) Early DW convs (stages_0.0, stages_1.0) reach ActMax ~232 before BN.
    ap_fixed<8,9> leaves only 1 fractional bit — very coarse. Consider keeping
    these two layers in FP32 or using per-channel input clipping.

---

## 4. Mixed-Precision Schemes (Industry Standard)

| Scheme   | Weights | Activations | Notes                                          |
|----------|---------|-------------|------------------------------------------------|
| W32A32   | FP32    | FP32        | Full precision — current C-sim                 |
| W16A16   | FP16    | FP16        | GPU half-precision inference                   |
| W8A32    | INT8    | FP32        | **Validated for this project** (+3.3% delta)   |
| W8A8     | INT8    | INT8        | Target for FPGA synthesis (per-layer ap_fixed) |
| W4A16    | INT4    | FP16        | LLaMA-style LLM inference on GPU              |
| W4A8     | INT4    | INT8        | Aggressive — NVIDIA TensorRT-LLM               |
| W1A8     | 1-bit   | INT8        | BitNet — requires QAT from scratch             |

---

## 5. Hardware-Specific Contexts

- **NVIDIA GPU (Hopper H100+):** FP8 E4M3/E5M2 natively; TensorRT uses INT8 for older GPUs
- **Google TPU:** BF16 training, INT8 inference
- **Apple Neural Engine:** FP16 and INT8 via CoreML
- **ARM Ethos NPU:** INT8 symmetric/asymmetric, UINT8 activations
- **Xilinx FPGA (this project):** ap_fixed per-layer — you choose exact bit width per layer,
  unlike GPUs which have fixed MAC hardware. This is more flexible but requires manual
  calibration (done via fpga_support/ptq_calibrate.py).

---

## 6. HTDet Weight Sizes (12,425,192 total FP32 elements)

| File             | FP32   | Notes                        |
|------------------|--------|------------------------------|
| backbone_w.bin   | 19.75 MB | MobileViT-S backbone       |
| fpn_w.bin        | 10.40 MB | FPN lateral + fpn convs    |
| cls_conv_w.bin   | 9.45 MB  | RetinaNet cls stacked convs|
| reg_conv_w.bin   | 9.45 MB  | RetinaNet reg stacked convs|
| cls_pred_w.bin   | 0.33 MB  | Classification predictor   |
| reg_pred_w.bin   | 0.33 MB  | Regression predictor       |
| **TOTAL**        | **49.70 MB** |                        |

### Weights-only size at each precision level

| Precision      | Size     | Reduction |
|----------------|----------|-----------|
| FP32 (baseline)| 49.70 MB | 1×        |
| FP16 / BF16    | 24.85 MB | 2×        |
| INT8 (W8)      | 12.43 MB | 4×        |
| INT4 (W4)      | 6.21 MB  | 8×        |
| INT2           | 3.11 MB  | 16×       |
| INT1           | 1.55 MB  | 32×       |

---

## 7. Recommendation for This Project

See section below for full analysis → `quantization_recommendation.md`
(or read the project analysis in the conversation / fpga_types.h comments)
