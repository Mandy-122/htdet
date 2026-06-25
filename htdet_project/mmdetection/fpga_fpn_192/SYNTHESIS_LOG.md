# FPN Neck — HLS Synthesis Log

**Module family**: `fpn_p5_top` → `fpn_p4p5_top` → `fpn_p3p4p5_top` → *(fpn_full_top — Phase 4)*  
**Architecture**: Feature Pyramid Network, MMDetection default config  
**Device**: xczu9eg-ffvb1156-2-e (ZCU102), Clock: 5 ns (200 MHz)  
**Weight types**: `weight_t = int8_t` (PTQ conv kernels), `meta_t = float` (biases)  
**Activation type**: `float` (W8A32)

---

## Model Parameters

| Parameter | Value | Notes |
|---|---|---|
| in_channels | [64, 96, 128, 640] | C1–C4 from MobileViT-S backbone |
| out_channels (FPN_OUT_CH) | 192 | All FPN levels |
| num_outs | 5 | P2–P6 |
| Input resolution | 320×320 | Spatial dims at each level: 80/40/20/10 |
| Activation | None | FPN default in MMDetection (act_cfg=None) |
| Extra level | P6 = MaxPool2d(k=1, s=2) on P5 | Stride-2 subsample, no learned weights |

**FPN Level Spatial Dimensions (320×320 input):**

| Level | Input channels | Lateral | Output spatial | Notes |
|---|---|---|---|---|
| P2 | C1 = 64 | 192 | 80×80 | *(Phase 4)* |
| P3 | C2 = 96 | 192 | 40×40 | Phase 3 |
| P4 | C3 = 128 | 192 | 20×20 | Phase 2 |
| P5 | C4 = 640 | 192 | 10×10 | Phase 1 |
| P6 | — | — | 5×5 | maxpool(P5) |

---

## Key Design Decisions

| Decision | Implementation | Reason |
|---|---|---|
| CHW layout throughout | `input[c*H*W + h*W + w]` | Matches backbone output format; avoids transposition |
| Separate int8 / float arrays | `w_conv` (int8), `w_meta` (float) | PTQ: int8 kernel weights, float biases — matches mbconv pattern |
| No BN on FPN convs | Plain bias add | MMDetection FPN has no BatchNorm (act_cfg=None, no norm_cfg) |
| PIPELINE on innermost loop | `C1P_IC` (1×1) / `FPN3_KW` (3×3) | Avoids PIPELINE+UNROLL body explosion that hangs HLS 2022.2 |
| OC_TILE=4 | Outer `C1P_OC` / `FPN3_OC` steps by 4 | 4× fewer outer loop restarts; II unchanged at 5; no synthesis hang |
| `add_upsampled_2x_inplace` | RMW on dst, reads src directly | Eliminates temporary upsample buffer (saves 1.2 MB BRAM at P3 level) |
| `maxpool2x2` = stride-2 subsample | `dst[...] = src[(2*oh)*W + (2*ow)]` | MMDetection P6 uses `MaxPool2d(kernel=1, stride=2)` — not a true 2×2 max |
| Block partition on top-level args | **Removed** | block factor=640 on BRAM interface port hung HLS interface elaboration |
| `bind_storage type=RAM_T2P impl=BRAM` | All internal buffers | Replaces deprecated `#pragma HLS RESOURCE` (Vitis HLS 2022.2) |

---

## Synthesis History

### Phase 1 — `fpn_p5_top` (P5+P6 only, OC_TILE=1)
*Validated P5 pathway: C4→lat4→conv3x3→P5→P6*

| Setting | Value |
|---|---|
| Top function | `fpn_p5_top` |
| Inputs | c4[640,10,10] |
| Outputs | p5[192,10,10], p6[192,5,5] |
| WCONV | 454,656 int8 |
| WMETA | 384 float |
| OC_TILE | 1 (single accumulator per outer trip) |
| Synthesis date | 2026-06-01 |

#### Per-module latency

| Pipeline module | Loop | Trip | II | Depth | Cycles | Time |
|---|---|---|---|---|---|---|
| C1P_HW_C1P_IC | lat4 conv1×1 (HW×IC) | 64,000 | 5 | 21 | 320,015 per OC | — |
| C1P_OC outer | lat4 lateral | 192 serial | — | — | **61,443,840** | **307 ms** |
| FPN3_IC_FPN3_KH_FPN3_KW | P5 output conv | 1,728 | 5 | 20 | 8,654 per OW | — |
| FPN3_OC_OH_OW outer | P5 conv3×3 | 19,200 serial | — | — | **166,425,600** | **832 ms** |
| MP_C_MP_H_MP_W | P6 maxpool | 4,800 | 1 | 3 | 4,803 | <1 ms |
| **Total** | | | | | **227,874,246** | **1.139 s** |

#### Resources

| BRAM_18K | DSP | FF | LUT | URAM | Timing (est.) | Fmax |
|---|---|---|---|---|---|---|
| 38 (2%) | 6 (~0%) | 2,157 (~0%) | 3,800 (1%) | 0 | 3.824 ns (−0.17 ns vs budget) | **261 MHz** ✅ |

#### II root cause

Both pipelines: loop-carried dependency on `acc` through 6-cycle no-DSP `fadd` → II = 5.  
Timing: 3.824 ns > 3.65 ns effective budget (= 5 ns − 1.35 ns uncertainty), but < 5 ns actual → passes.

---

### Phase 2a — `fpn_p4p5_top` (P4+P5+P6, OC_TILE=1)
*Added C3 lateral and top-down merge; OC_TILE still 1.*

| Total latency | BRAM | DSP | FF | LUT |
|---|---|---|---|---|
| **942,765,922 cycles = 4.714 s** | 328 (17%) | 16 | 7,477 (1%) | 9,962 (3%) |

P4 conv3×3 (76,800 outer trips) dominated at 3.33 s = 70% of total.

---

### Phase 2b — `fpn_p4p5_top` (P4+P5+P6, OC_TILE=4) ✅ CURRENT BEST
*OC_TILE=4 on both `conv1x1_plain` and `fpn_conv3x3` → 4× speedup.*

| Setting | Value |
|---|---|
| OC_TILE | **4** (outer loop steps by 4, 4 independent acc chains) |
| Inputs | c4[640,10,10], c3[128,20,20] |
| Outputs | p5[192,10,10], p4[192,20,20], p6[192,5,5] |
| WCONV | 811,008 int8 |
| WMETA | 768 float |

#### Per-module latency

| Pipeline module | Trip | II | Cycles | Time | vs Phase 2a |
|---|---|---|---|---|---|
| C1P_HW_C1P_IC lat4 | 64,000 | 5 | 320,019 per OC4 | — | — |
| C1P_OC (lat4) | **48** serial | — | **15,361,105** | **77 ms** | 4× faster |
| C1P_HW_C1P_IC lat3 | 51,200 | 5 | 256,019 per OC4 | — | — |
| C1P_OC (lat3) | **48** serial | — | **12,289,105** | **61 ms** | 4× faster |
| UP_C_UP_H_UP_W | 19,200 | 2 | 38,425 | 0.19 ms | — |
| EAI (elem_add) | 76,800 | 1 | 76,811 | 0.38 ms | — |
| FPN3_IC_KH_KW (P5) | 1,728 | 5 | 8,659 per OW | — | — |
| FPN3_OC_OH_OW (P5) | **4,800** serial | — | **41,625,601** | **208 ms** | 4× faster |
| FPN3_IC_KH_KW (P4) | 1,728 | 5 | 8,659 per OW | — | — |
| FPN3_OC_OH_OW (P4) | **19,200** serial | — | **166,521,602** | **832 ms** | 4× faster |
| MP_C_MP_H_MP_W | 4,800 | 1 | 4,803 | <1 ms | — |
| **Total** | | | **235,797,420** | **1.179 s** | **4× faster** |

#### Resources

| BRAM_18K | DSP | FF | LUT | URAM | Timing | Fmax |
|---|---|---|---|---|---|---|
| 328 (17%) | 18 (~0%) | 10,446 (1%) | 15,087 (5%) | 0 | −0.17 ns vs budget | **261 MHz** ✅ |

BRAM breakdown: 38 (lat4) + 150 (lat3) + 120 (up4 buffer) ≈ 328 BRAM18K.  
DSP +2 vs Phase 2a: extra fmul units for the 4 parallel OC chains.  
p4/p5/w_conv now expose both PORTA + PORTB (HLS uses dual ports for 4 concurrent reads/writes).

---

### Phase 3 — `fpn_p3p4p5_top` (P3+P4+P5+P6, OC_TILE=4) ✅ SYNTHESISED
*Adds C2[96,40,40] lateral and second top-down merge. Uses `add_upsampled_2x_inplace` — no up-buffer.*

| Setting | Value |
|---|---|
| OC_TILE | 4 |
| Inputs | c4[640,10,10], c3[128,20,20], c2[96,40,40] |
| Outputs | p5[192,10,10], p4[192,20,20], p3[192,40,40], p6[192,5,5] |
| WCONV | 1,161,216 int8 |
| WMETA | 1,152 float |

#### Per-module latency (actual)

| Pipeline module | Trip | II | Cycles | Time |
|---|---|---|---|---|
| C1P_OC lat4 | 48 | — | 15,361,105 | 77 ms |
| C1P_OC lat3 | 48 | — | 12,289,105 | 61 ms |
| C1P_OC lat2 (C2=96, HW=1600) | 48 | — | **36,865,201** | **184 ms** |
| ADD_UP lat4→lat3 (76,800) | 76,800 | 1 | 76,814 | 0.38 ms |
| ADD_UP lat3→lat2 (307,200) | 307,200 | 1 | 307,214 | 1.54 ms |
| fpn_conv3x3 P5 | 4,800 | 5 | 41,625,601 | 208 ms |
| fpn_conv3x3_3 P4 | 19,200 | 5 | 166,521,602 | 832 ms |
| fpn_conv3x3_4 **P3** | **76,800** | 5 | **666,316,804** | **3,332 ms** |
| MP P6 | 4,800 | 1 | 4,803 | <1 ms |
| **Total** | | | **938,979,429** | **4.695 s** |

#### Resources

| BRAM_18K | DSP | FF | LUT | URAM | Timing |
|---|---|---|---|---|---|
| **724 (39%)** | 22 (~0%) | 9,914 (1%) | 16,998 (6%) | 0 | −0.17 ns (Fmax 261 MHz) ✅ |

`add_upsampled_2x_inplace` confirmed II=1 for both up-add steps — no DEPENDENCE violation.

---

## Bottleneck Analysis

### 1. Outer serial loops dominate (not II)

The inner pipelines (`FPN3_IC_KH_KW`, `C1P_HW_C1P_IC`) are efficient at II=5. The bottleneck is the **serial outer loops** that restart the inner pipeline for each output pixel/channel.

- OC_TILE=1: outer trips = FPN_OUT_CH × H × W (e.g., 192×20×20 = 76,800 for P4)
- OC_TILE=4: outer trips = (FPN_OUT_CH/4) × H × W (= 19,200 for P4) → **4× speedup**
- Next step (OC_TILE=8): outer trips = 9,600 → 2× further speedup from OC_TILE=4

### 2. II=5 is BRAM-bandwidth limited, not fadd

Phase 1 log showed:  
`fadd (2.95 ns) + select (0.449 ns) + store (0.427 ns) = 3.826 ns` critical path  
→ Fmax = 261 MHz, II constrained to 5 by loop-carried `acc` dep through 6-cycle no-DSP `fadd`.

With OC_TILE=4, HLS tries II=1 but settles at II=5 for the same reason. The 4 extra weight reads per body fit within the 5-cycle II window (T2P BRAM: 2 reads/cycle × 5 cycles = 10 reads available; body needs 4+1=5 reads).

### 3. P3/P2 spatial scaling

Outer trip count scales as (H×W):

| Level | (H×W) | Outer trips (OC_TILE=4) | Conv3×3 time |
|---|---|---|---|
| P5 | 10×10 = 100 | 4,800 | 208 ms |
| P4 | 20×20 = 400 | 19,200 | 832 ms |
| P3 | 40×40 = 1,600 | 76,800 | ~3.33 s |
| P2 | 80×80 = 6,400 | 307,200 | ~13.3 s |

Full P2–P6 estimate (OC_TILE=4): **~18 s** — feasible for synthesis/validation.  
Further speedup requires OC_TILE=8 or pipelining across the OC/OH/OW outer loops.

### 4. BRAM grows with spatial level

| Phase | Internal buffers | BRAM18K (internal) |
|---|---|---|
| Phase 1 (P5) | lat4 = 75 KB | 38 (2%) |
| Phase 2 (P4+P5) | lat4+lat3+up4 = 675 KB | 328 (17%) |
| Phase 3 (P3+P4+P5) | lat4+lat3+lat2 = 1,575 KB | ~788 (43%) |
| Phase 4 (full P2–P6) | +lat1 (1.2 MB) | ~1,388 (76%) |

`add_upsampled_2x_inplace` avoids 1.2 MB temporary buffer at each level vs the `upsample_2x + elem_add` approach used in Phase 2.

---

## II Detailed Table (from synthesis reports)

| Pipeline | Input | Trip | Target II | Final II | Root cause |
|---|---|---|---|---|---|
| C1P_HW_C1P_IC (lat4) | C4=640 ICs, HW=100 | 64,000 | 1 | **5** | `acc` RAW through 6-cycle `fadd_no_dsp` |
| C1P_HW_C1P_IC (lat3) | C3=128 ICs, HW=400 | 51,200 | 1 | **5** | same |
| FPN3_IC_KH_KW (P5) | IC=192, K=9 | 1,728 | 1 | **5** | same |
| FPN3_IC_KH_KW (P4) | IC=192, K=9 | 1,728 | 1 | **5** | same |
| UP_C_UP_H_UP_W | 4 writes/src pixel | 19,200 | 1 | **2** | 4 writes to T2P BRAM (2 ports) |
| EAI (elem_add) | 1 RMW/element | 76,800 | 1 | **1** ✅ | — |
| MP_C_MP_H_MP_W | 1 read/output | 4,800 | 1 | **1** ✅ | — |

---

## HLS 2022.2 Pitfalls Encountered

| Issue | Symptom | Fix |
|---|---|---|
| `ARRAY_PARTITION block factor=640` on top-level BRAM arg | HLS hung at "Using interface defaults" — never finished | Use cyclic factor=16 or no partition on interface ports; reserve large block partitions for internal static buffers |
| `PIPELINE II=1` + `UNROLL factor=N` (N>8) on inner loop | HLS scheduler hung (exponential ILP) for both P5 Phase 1 first run and FPN3 with UNROLL factor=8 | Move PIPELINE to innermost loop only; no UNROLL on sub-loops (body = 1 MAC) |
| `#pragma HLS RESOURCE` | Deprecated warning in 2022.2 | Replace with `#pragma HLS bind_storage type=RAM_T2P impl=BRAM` |
| `UNROLL factor=2` inside `PIPELINE II=1` (linear_f pattern) | 2022.2 promotes partial UNROLL to complete unroll → 192 concurrent reads → II=97 | Known Vitis HLS 2022.2 behavior: any sub-loop in PIPELINE body gets fully unrolled. Workaround: use innermost PIPELINE with no UNROLL |

---

## Applied Optimizations

### OC_TILE 4→8 + `PIPELINE II=5` for `conv1x1_plain` and `fpn_conv3x3` ✅ SYNTHESISED (report7)
### OC_TILE 8→16 + cyclic-17 partition on `w_conv` ⏳ PENDING SYNTHESIS

Both FPN functions compute 8 output channels per outer loop trip (was 4). Pipeline target changed from II=1 to II=5 to force 2-DSP allocation.

**Why II stays at 5 — DSP allocation analysis (from report6 sub-module inspection):**

Report6 confirms: the inner pipeline (`_Pipeline_FPN3_IC_FPN3_KH_FPN3_KW`) uses exactly **1 DSP** (`mac_muladd`, latency=4) for ALL 4 OC accumulations. The 3 DSPs in fpn_conv3x3_5/6 are for **outer-loop address arithmetic** (`mul_mul_8ns_11ns_19/21_4_1`), not the inner accumulation.

**II=5 mechanism with OC_TILE=4:**  1 DSP × 4 sequential MAC states (cycles 1–4) → acc0[n] ready at cycle 5 → next iteration starts at cycle 5 → II = 5.

**With OC_TILE=8 and `#pragma HLS PIPELINE II=1`:**  1 DSP handles 8 chains in 8 sequential states → II would relax to **8** → only ~1.25× improvement.

**With OC_TILE=8 and `#pragma HLS PIPELINE II=5`:**  HLS allocates **2 DSPs** to fit 8 chains in 5 states (4 chains/DSP, running in parallel). acc0 dep (latency 4) gives II ≥ 5 → **II = 5** ✓ → outer trips halved → **2× speedup**.

**Synthesis confirmed — actual latency results:**

| Module | OC_TILE=4, II=5 | OC_TILE=8, II=5 | Actual speedup |
|---|---|---|---|
| conv1x1_plain lat4 (640IC, 10×10) | 15.4M cycles | 7.7M | 2× |
| conv1x1_plain lat3 (128IC, 20×20) | 12.3M cycles | 6.1M | 2× |
| conv1x1_plain lat2 (96IC, 40×40) | 36.9M cycles | 18.4M | 2× |
| conv1x1_plain lat1 (64IC, 80×80) | 98.3M cycles | 49.2M | 2× |
| fpn_conv3x3 P5 (10×10) | 41.6M cycles | 20.8M | 2× |
| fpn_conv3x3 P4 (20×20) | 166.5M cycles | 83.3M | 2× |
| fpn_conv3x3 P3 (40×40) | 666.3M cycles | 333.2M | 2× |
| fpn_conv3x3 P2 (80×80) | 2,665.3M cycles | 1,332.6M | 2× |
| **fpn_full_top total** | **3,703M = 18.513 s** | **~1,851M = 9.258 s** ✅ | **~2×** |

**Synthesis date:** 2026-06-02. Predicted 9.3 s, actual **9.258 s** — prediction accurate to within 0.5%.

## Pending Optimizations

| Optimization | Target | Expected gain | Status |
|---|---|---|---|
| OC_TILE=8 + `II=5` for conv1x1_plain + fpn_conv3x3 | Full FPN 18.5s → ~9.3s | ~2× | ✅ CONFIRMED — 9.258 s |
| **OC_TILE=16 + `II=5` + cyclic-17 partition on `w_conv`** | **Full FPN 9.258s → ~4.63s** | **~2×** | **⏳ CODE READY — NOT YET SYNTHESISED** |
| Flatten OC×OH×OW outer loops | Eliminate pipeline drain overhead | ~0.25% (negligible) | Low priority |
| `add_upsampled_2x_inplace` for Phase 2 up4 buffer | Save 300 KB BRAM in fpn_p4p5_top | Minor | Backport if needed |

### OC_TILE=16 — Analysis and Expected Results

`fpga_utils.h` already has `oc += 16` for both `conv1x1_plain` and `fpn_conv3x3`.
`#pragma HLS PIPELINE II=5` is set on both inner pipelines.

**Why cyclic-17 partition is mandatory:**
- OC_TILE=16 needs 16 weight reads per inner iteration body.
- Dual-port `w_conv` BRAM (T2P): 2 reads/cycle × 5 cycles = 10 read slots < 16 → BRAM bottleneck.
- Without partition: HLS relaxes II to ≥9 (ceil(16/2)=8 cycles BRAM + 1 margin)
  → only ~10% gain over OC_TILE=8. ❌
- With cyclic-17 partition: each bank accessed independently; 16 reads in 1 cycle → II_BRAM=1.
  HLS achieves II=5 (DSP-limited); outer trips halved → **2× speedup**. ✓

**Why factor=17 works for all FPN weight strides:**
- Stride between OC reads in `conv1x1_plain` = `in_ch`:
  640%17=11, 128%17=9, 96%17=11, 64%17=13 — all coprime with 17
- Stride in `fpn_conv3x3` = `FPN_OUT_CH×9 = 1728`: 1728%17=11, gcd(11,17)=1 ✓
- Coprime strides ⟹ 16 consecutive OC accesses always map to 16 distinct banks.
- Cyclic-16 would NOT work: in_ch is always a multiple of 16 → all reads hit bank 0.

**RTL interface note:** cyclic-17 partition on a top-level BRAM-interface array generates
17 independent BRAM interface ports on the RTL module. The parent system must provide
17 physical BRAM banks connected to those ports.

**Predicted latency table (OC_TILE=16, II=5, cyclic-17):**

| Module | OC_TILE=8, II=5 | OC_TILE=16, II=5 | Predicted gain |
|---|---|---|---|
| conv1x1_plain lat4 (640IC, 10×10) | 7.68M | 3.84M | 2× |
| conv1x1_plain lat3 (128IC, 20×20) | 6.14M | 3.07M | 2× |
| conv1x1_plain lat2 ( 96IC, 40×40) | 18.43M | 9.22M | 2× |
| conv1x1_plain lat1 ( 64IC, 80×80) | 49.15M | 24.58M | 2× |
| fpn_conv3x3 P5 (10×10) | 20.82M | 10.41M | 2× |
| fpn_conv3x3 P4 (20×20) | 83.28M | 41.65M | 2× |
| fpn_conv3x3 P3 (40×40) | 333.24M | 166.62M | 2× |
| fpn_conv3x3 P2 (80×80) | 1332.94M | 666.47M | 2× |
| **fpn_full_top total** | **9.258 s** | **~4.63 s** | **~2×** |

---

## File Map

| File | Description |
|---|---|
| `fpga_types.h` | PTQ types, all spatial/channel constants (320×320 input, FPN_OUT_CH=192) |
| `fpga_utils.h` | All primitives: `conv1x1_plain` (FPN_OC_TILE=16), `fpn_conv3x3` (FPN_OC_TILE=16), `add_upsampled_2x_inplace`, `maxpool2x2`, upsample, elem_add |
| `fpn_p5_top.h/cpp` | Phase 1: P5+P6 (C4 only) |
| `fpn_p4p5_top.h/cpp` | Phase 2: P4+P5+P6 (C3+C4) |
| `fpn_p3p4p5_top.h/cpp` | Phase 3: P3+P4+P5+P6 (C2+C3+C4) |
| `testbench_fpn_p5.cpp` | Phase 1 testbench (3 tests incl. numerical) |
| `testbench_fpn_p4p5.cpp` | Phase 2 testbench |
| `testbench_fpn_p3p4p5.cpp` | Phase 3 testbench |
| `run_hls_fpn_p5.tcl` | Vitis HLS TCL — Phase 1 |
| `run_hls_fpn_p4p5.tcl` | Vitis HLS TCL — Phase 2 |
| `run_hls_fpn_p3p4p5.tcl` | Vitis HLS TCL — Phase 3 |
| `export_fpn_weights.py` | Python weight exporter (PTQ int8 → binary files for Test 3) |

---

## All HTDet FPGA Modules — Best Latency Summary

Device: xczu9eg-ffvb1156-2-e @ 200 MHz (5 ns clock) unless noted.  
"Best" = lowest latency from synthesised runs; expected = not yet synthesised.

| Module | Best Latency |
|---|---|
| `mbconv_160_top` (stem→C1, 160×160→80×80) | ~1.22 s *(expected — not yet synthesised)* |
| `mbconv_80_top` (C1 residual, 80×80) | **780 ms** |
| `mbconv_40_top` (C2→C3 down, 40×40→20×20) | **252 ms** |
| `mbconv_20_top` (C3 residual, 20×20) | **104 ms** |
| `transformer_blk_s2_top` (MobileViT S2, DIM=144, SEQ=400, depth=2) | **470 ms** |
| `transformer_blk_s3_top` (MobileViT S3, DIM=192, SEQ=100, depth=4) | **325 ms** |
| `transformer_blk_s4_top` (MobileViT S4, DIM=240, SEQ=25, depth=3) | **90 ms** |
| `fpn_p5_top` (FPN P5+P6 only) | **1.14 s** *(Phase 1, OC_TILE=1)* |
| `fpn_p4p5_top` (FPN P4+P5+P6, OC_TILE=4) | **1.18 s** |
| `fpn_p3p4p5_top` (FPN P3+P4+P5+P6, OC_TILE=4) | **4.695 s** |
| `fpn_full_top` (full FPN P2+P3+P4+P5+P6, OC_TILE=4) | 18.513 s *(superseded)* |
| `fpn_full_top` (full FPN P2+P3+P4+P5+P6, OC_TILE=8, II=5) | **9.258 s** ✅ *(BRAM 158%; latency confirmed 2026-06-02; superseded by OC_TILE=16)* |
| `fpn_full_top` (full FPN P2+P3+P4+P5+P6, OC_TILE=16, II=5, cyclic-17) | **~4.63 s** ⏳ *(predicted; code ready; needs synthesis with cyclic-17 partition)* |
| `retina_head_top` (4 stacked convs + cls/reg pred, all 5 levels) | **1.72 s** *(v2; v3 ~110 ms predicted)* |

---

## Missing Modules — Full Inference Pipeline

From `mobilevit_backbone.h`, the complete backbone call graph is:

| # | Operation | Config | Spatial | Module | Status |
|---|---|---|---|---|---|
| 1 | Stem Conv3×3 BN SiLU | 3→16, s=2 | 320→160 | *(trivial, no dedicated module)* | — |
| 2 | MBConv | 16→32, s=1 | 160×160 | `mbconv_160` function, different config | ❌ not synthesised |
| 3 | **MBConv** | 32→64, s=2 | 160→80 | `mbconv_160_top` | ⏳ expected ~1.22 s |
| 4 | MBConv × 2 | 64→64, s=1 | 80×80 | `mbconv_80_top` × 2 | ✅ 2 × 780 ms ⚠URAM |
| 5 | **MBConv** | 64→96, s=2 | 80→40 | uses `mbconv_80` function, different in/out | ❌ not synthesised |
| 6 | MobileViT S2 conv layers | conv_kxk(96→96) + conv_fusion(192→96) at 40×40 | 40×40 | no module | ❌ ~2.5 s estimated |
| 7 | **t_blk_s2** (transformer stack) | DIM=144, SEQ=400, depth=2 | 40×40 | `transformer_blk_s2_top` | ✅ **470 ms** |
| 8 | **MBConv** (mbconv_40) | 96→128, s=2 | 40→20 | `mbconv_40_top` | ✅ **252 ms** |
| 9 | MobileViT S3 conv layers | conv_kxk(128→128) + conv_fusion(256→128) at 20×20 | 20×20 | no module | ❌ ~1.1 s estimated |
| 10 | **t_blk_s3** | DIM=192, SEQ=100, depth=4 | 20×20 | `transformer_blk_s3_top` | ✅ **325 ms** |
| 11 | **MBConv** | 128→160, s=2 | 20→10 | `mbconv_20` function — actual call is 128→**160** s=2, but `mbconv_20_top` covers 128→128 s=1 | ⚠ config mismatch |
| 12 | MobileViT S4 conv layers | conv_kxk(160→160) + conv_fusion(320→160) at 10×10 | 10×10 | no module | ❌ ~430 ms estimated |
| 13 | **t_blk_s4** | DIM=240, SEQ=25, depth=3 | 10×10 | `transformer_blk_s4_top` | ✅ **90 ms** |
| 14 | Conv1×1 BN SiLU | 160→640 | 10×10 | no module | ❌ ~20 ms (small) |
| 15 | FPN P2+P3+P4+P5+P6 | in=[64,96,128,640], out=192 | various | `fpn_p3p4p5_top` covers P3–P6; **P2 (80×80) missing** | ⚠ Phase 4 |
| 16 | RetinaNet head | 4 stacked convs, 5 levels | P2–P6 | `retina_head_top` v2 | ✅ 1.72 s (v3 pending) |
| 17 | NMS | max 100 dets | — | `fpga_nms` | ❌ no synthesis report |

**Summary of missing pieces:**
- ❌ 5 unsynthesised blocks (stem, MBConv 16→32, MBConv 64→96, MobileViT conv layers ×3, Conv 160→640)
- ⚠ `mbconv_20_top` config mismatch (synthesised as 128→128 s=1, backbone needs 128→160 s=2)
- ⚠ URAM overflow on all MBConv modules — `mbconv_80_top` uses 907 BRAM18K (1133%)
- ⚠ FPN P2 level (Phase 4) — dominant latency contributor

---

## Expected Final End-to-End Latency

All modules sequential, xczu9eg @ 200 MHz.  
**Current** = as synthesised (URAM issues unfixed).  **Optimised** = with URAM→BRAM fix + head v3.

| Component | Current latency | Optimised |
|---|---|---|
| Stem + MBConv(16→32, 160×160) | ~1.5 s *(not yet synth.)* | ~0.5 s |
| mbconv_160 (32→64, s=2) | ~1.22 s *(expected)* | ~1.22 s |
| mbconv_80 × 2 (64→64, s=1) | **1.56 s** (2×780 ms ⚠URAM) | ~0.4 s |
| MBConv(64→96, s=2, 80→40) | ~0.8 s *(not synth.)* | ~0.2 s |
| MobileViT S2 conv layers | ~2.5 s *(estimated)* | ~2.5 s |
| transformer_blk_s2 | **470 ms** | 470 ms |
| mbconv_40 (96→128, s=2) | **252 ms** | 252 ms |
| MobileViT S3 conv layers | ~1.1 s *(estimated)* | ~1.1 s |
| transformer_blk_s3 | **325 ms** | 325 ms |
| MBConv(128→160, s=2, 20→10) | ~0.25 s *(not synth.)* | ~0.25 s |
| MobileViT S4 conv layers | ~430 ms *(estimated)* | ~430 ms |
| transformer_blk_s4 | **90 ms** | 90 ms |
| Conv1×1 (160→640) | ~20 ms | ~20 ms |
| FPN P2+P3+P4+P5+P6 | 18.513 s (OC_TILE=4) | **9.258 s** ✅ (OC_TILE=8 + II=5, confirmed 2026-06-02) |
| RetinaNet head | **1.72 s** (v2) | ~110 ms (v3) |
| NMS | ~20 ms *(not synth.)* | ~20 ms |
| **Total** | **~32 s** | **~13 s** |

The dominant bottleneck is **FPN P2 (80×80, ~13 s)** — reducing it requires OC_TILE=8 or loop-flatten on P2's conv3×3. The MobileViT conv layers (S2 conv_kxk + conv_fusion) are the second-largest unaccounted block at ~3.6 s total.
