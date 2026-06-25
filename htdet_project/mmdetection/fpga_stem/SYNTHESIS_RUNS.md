# stem Synthesis Run History

**Module**: `stem_top` — MobileViT-S input stem  
**Config**: Conv3×3(3→16, stride=2, 320×320) + BN + SiLU  
**Device**: xczu28dr-ffvg1517-2-e (ZynqUltraScale+), Clock: 5 ns (200 MHz)  
**Layout**: CHW throughout  
**Input**: [3, 320, 320] float → **Output**: [16, 160, 160] float

---

## Key Parameters

| Parameter | Value |
|---|---|
| in_ch | 3 (INPUT_C) |
| out_ch | 16 (STEM_CH) |
| stride | 2 |
| Kernel | 3×3 |
| Spatial in | 320×320 |
| Spatial out | 160×160 |
| MACs | 16 × 160 × 160 × 3 × 9 = 11.1M |
| w_conv | 432 int8 |
| w_meta | 32 float (BN scale + bias) |

## Resource / Latency Estimates

- Only 3 input channels — IC loop unrolled fully (27 MACs per output pixel)
- Innermost PIPELINE: `OC × OH × OW` = 409,600 iterations
- Expected II=1, latency ≈ 409,600 cycles ≈ **2 ms @ 200 MHz**
- Very low resource usage (smallest module in the backbone)

## Synthesis Runs

| Date | Settings | Total cycles | @ 200 MHz | DSP | LUT | Notes |
|---|---|---|---|---|---|---|
| pending | baseline | TBD | TBD | TBD | TBD | First run |

---

## Pending Optimizations

| Optimization | Expected gain | Status |
|---|---|---|
| Baseline synthesis | Characterise II and latency | Planned |
| clock relaxation if needed | — | If timing fails |

---

## File Map

| File | Description |
|---|---|
| `fpga_utils.h` | CHW conv primitives (ptq_192 version) |
| `stem_top.cpp` | Top-level: calls conv3x3_bn_silu |
| `stem_top.h` | Constants: STEM_WCONV_ELEMS, STEM_WMETA_ELEMS |
| `fpga_types.h` | Shared data types |
| `testbench_stem.cpp` | C-sim testbench |
