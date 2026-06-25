# mbconv_160 Synthesis Run History

**Module**: `mbconv_160_top` — MobileNetV2-style inverted residual block (stride=2, downsampling)  
**Config**: in_ch=32 (STAGE0_CH), out_ch=64 (C1_CH), expand=4, stride=2, hid=128  
**Device**: xczu28dr-ffvg1517-2-e (ZynqUltraScale+), Clock: 5 ns (200 MHz)  
**Layers**: expand 1×1 (32→128) + DW 3×3 stride=2 (128) + proj 1×1 (128→64) [no residual — stride=2]  
**Spatial**: Input H=W=160 (STEM), Output H=W=80 (C1)

---

## Run History

### Run 1 — v4 port (HWC layout, OC_TILE=2, partial acc + tree, 2026-06-02)

| Setting | Value |
|---|---|
| Layout | HWC: `input[hw * ch + ic]` |
| 1×1 conv optimization | OC_TILE=2, partial acc, 4-level binary tree reduction |
| `in` partition | cyclic factor=16 |
| `w_conv` partition | cyclic factor=16 |
| `ex_buf` / `dw_buf` | URAM, bind_storage pragma |

#### Per-module latency

| Module | Function | II | Trip count | Cycles | @ 200 MHz | % total |
|---|---|---|---|---|---|---|
| expand 1×1 outer | conv1x1_bn_silu (32→128) | **—** | 819,200 (sequential) | 236,748,800 | **1,184 ms** | **97%** |
| ↳ IC tile (inner) | C1X1S_IC_T pipeline | **2 ⚠️** | 2 | 146 | — | (inner) |
| DW 3×3 | dw_conv3x3_bn_silu stride=2 (128 ch) | **5** | 819,200 | 4,096,185 | **20.5 ms** | 1.7% |
| proj 1×1 | conv1x1_bn (128→64, flattened) | **4 ⚠️** | 819,200 | 3,276,995 | **16.4 ms** | 1.3% |
| **Total** | | | | **244,121,984** | **1,221 ms** | 100% |

> No buf_copy or residual add (stride=2 block; in_ch ≠ out_ch).

#### Resources

| BRAM_18K | DSP | FF | LUT | URAM | Timing slack |
|---|---|---|---|---|---|
| 0 | 253 (5%) | 100,671 (11%) | 77,342 (18%) | 1,006 **(1257% ⚠️)** | **−0.29 ns ⚠️** |

Timing is violated at top level (−0.29 ns) and inside the DW pipeline (−0.15 ns).

#### Pipeline notes

| Pipeline module | Loop | II | Notes |
|---|---|---|---|
| C1X1S_IC_T | expand IC tile (2 tiles) | **2 ⚠️** | Only 2 tiles (IC=32, TILE=16); pipeline depth ~143 >> trip → II=2 not II=1 |
| C1X1_OC_C1X1_HW_C1X1_IC_T | proj (flattened, 128→64) | **4 ⚠️** | Weight bank conflicts with IC=128; HLS flattened OC×HW×IC_T into one pipeline |
| DW_C_DW_OH_DW_OW | depthwise 3×3 stride=2 | **5** | URAM 9-kernel bank conflict; 4× expensive `urem_64s` operators (8,651 FF + 6,607 LUT each) |

---

### Run 2 — v5 (HWC, OC_TILE=8, ex_buf cyclic partition, 4.5 ns clock, **pending**)

| Setting | Value |
|---|---|
| OC_TILE | **8** (was 4) — halves expand/proj outer iterations |
| ex_buf / dw_buf partition | `cyclic factor=128` — bank = addr%128 = c, eliminates urem_64s |
| Clock | **4.5 ns (222 MHz)** (was 5 ns) — clears timing violation |
| IC_T expected II | **4** (8 weight accesses / 2 BRAM ports per bank) |

#### Expected per-module latency (Run 2 estimate)

| Module | Cycles (est.) | @ 222 MHz (est.) | vs Run 1 |
|---|---|---|---|
| expand 1×1 outer | ~118M | ~531 ms | **−2.2×** |
| DW 3×3 | ~4.1M | ~18 ms | same |
| proj 1×1 | ~15M | ~68 ms | −4.6× (outer trips halved + II same) |
| **Total** | **~137M** | **~617 ms** | **~2× speedup** |

> DW latency unchanged — urem fix is resource/timing only, not cycle-count.
> Proj improvement larger than expand because proj outer trips also halve (64/8=8 groups vs 16).

---

## Bottleneck Analysis

### 1. Expand dominates (97% = 1,184 ms)

- Outer `C1X1S_OC × C1X1S_HW` loop is **sequential** (not pipelined)
- 819,200 iterations × 289 cycles/iter = 236.7M cycles
- Outer trip = (HID/OC_TILE) × HW = (128/2) × 25,600 with some loop normalisation = 819,200
- **Primary fix**: increase OC_TILE to 4 or 8 for expand  
  → OC_TILE=4: ~118M cycles (2× speedup); OC_TILE=8: ~59M cycles (4× speedup)

### 2. IC_T II=2 (pipeline under-utilised — only 2 tiles)

- IC=32, IC_TILE=16 → NUM_TILES=2; pipeline depth ≈ 143 cycles >> trip count=2
- HLS achieves II=2 (vs II=1 in mbconv_80 which has 4 IC tiles)
- Adds ~289−143 = 146 extra cycles per outer iteration vs ideal
- **Fix A**: reduce IC_TILE to 8 → NUM_TILES=4 → pattern matches mbconv_80, expect II=1
- **Fix B**: OC_TILE=4 increases compute per outer iteration, amortising tile overhead regardless

### 3. Proj II=4 (double mbconv_80's II=2)

- IC=128 for proj; with w_conv partitioned factor=16, deeper rows create more port conflicts
- HLS flattened OC×HW×IC_T into one 819,200-trip pipeline (same compensation as mbconv_80)
- Impact: only 1.3% of total — not a priority while expand is 97%
- **Fix when needed**: change to OC_TILE=4 for proj, or increase w_conv partition factor

### 4. URAM massively over-budget (1257%)

- ex_buf [160×160×128 × float32] = 12.5 MB; dw_buf [160×160×128 × float32] = 12.5 MB
- Device: 80 URAMs × 288 Kbits = 2.88 MB total
- Does not block C-synthesis; prevents P&R
- **Fix**: row-strip spatial tiling (e.g., 16 rows at a time) → ex_buf = 16×160×128 × 4B = 1.25 MB → fits in URAM

### 5. Timing violation (−0.29 ns at top, −0.15 ns in DW)

- Clock target 5 ns (200 MHz) is not met
- DW loop critical path is driven by the `urem_64s` 64-bit modulo divider chain
- **Fix A** (easy): relax clock to 4.5 ns (222 MHz) — likely clears both violations
- **Fix B**: replace urem with power-of-2 channel mask or explicit stride comparisons in DW address calculation

### 6. DW: urem_64s operators are resource-catastrophic

- 4× `urem_64s_5ns_4_68_1` instances in the DW pipeline: **8,651 FF and 6,607 LUT each**
- Total urem cost ≈ 34,604 FF + 26,428 LUT (≈ 40% of entire LUT budget just for DW address arithmetic!)
- These arise from non-power-of-2 modulo ops in the URAM bank-index computation
- HID=128 is a power of 2, so the source of `urem` is likely a stride or padding boundary check
- **Fix**: audit `dw_conv3x3_bn_silu` — replace `% constant` with bitwise `& (constant-1)` where constant is a power of 2, or use explicit if/else boundary logic

---

## Comparison with mbconv_80 Run 1

| Metric | mbconv_80 | mbconv_160 | Notes |
|---|---|---|---|
| Total cycles | 197.4M | **244.1M** | +24% despite smaller channel count |
| Total time @ 200 MHz | 987 ms | **1,221 ms** | +24% |
| Expand fraction | 92% | **97%** | Expand even more dominant |
| Expand IC_T II | **1 ✅** | **2 ⚠️** | 2 vs 4 IC tiles |
| Proj II | **2** | **4 ⚠️** | IC=128 vs 256 causes more conflicts |
| DW trips | 1,638,400 (256 ch, 80²) | **819,200** (128 ch, 80² out) | DW is actually faster in mbconv_160 |
| URAM | 1133% | **1257%** | 4× larger spatial (160² vs 80²) |
| Timing slack | ≈+1 ns ✅ | **−0.29 ns ⚠️** | Violation appears in 160 |

---

## Scaling Law (1×1 expand conv, HWC, OC_TILE=2, II=2)

```
IC_T latency (II=2)  ≈ 2 × (NUM_TILES - 1) + pipeline_depth  ≈ 146 cycles for NUM_TILES=2
Per outer iter       = IC_T latency + tree + BN/SiLU ≈ 289 cycles (observed)
Total expand         = (HID/OC_TILE) × H × W × 289
                     = 64 × 160 × 160 × 289  → normalised → 819,200 × 289 = 236.7M
```

With OC_TILE=4 and IC_T II=1 (4 tiles from IC_TILE=8):
```
IC_T latency (II=1)  ≈ 146 cycles (same depth, better throughput)
Per outer iter       ≈ 146 + ~96 = 242 cycles (rough estimate)
Outer trips          = (128/4) × 25,600 = 819,200 / 2 = 409,600
Total                ≈ 409,600 × 242 ≈ 99M cycles ≈ 495 ms  (2.5× speedup)
```

---

## Pending Optimizations

| Optimization | Target | Expected gain | Priority | Status |
|---|---|---|---|---|
| OC_TILE=8 for expand+proj | 1,221 ms → ~617 ms | ~2× overall | **High** | ✅ **Run 2** |
| Relax clock to 4.5 ns | timing −0.29 → cleared | Unblocks impl quality | **High** | ✅ **Run 2** |
| ex_buf/dw_buf cyclic factor=128 | eliminates urem_64s (~34K FF, ~26K LUT) | Resource + DW timing fix | **Medium** | ✅ **Run 2** |
| Row-strip URAM tiling | 1257% → ~20% | P&R feasibility | High (for impl) | Future |
| IC_TILE=8 (NUM_TILES=4) | IC_T II=4 → II=2 | ~10% on expand iter latency | Low (outer trip dominates) | Future |
| Fix proj II further | proj ~68ms → ~35ms | ~33 ms | Low | Future |

---

## Buffer Size Reference

| Buffer | Dimensions | Size | vs Device URAM |
|---|---|---|---|
| ex_buf | 160×160×128 × float32 | 12.5 MB | 434% over |
| dw_buf | 160×160×128 × float32 | 12.5 MB | 434% over |
| **Total** | | **25.0 MB** | **1257% ⚠️** |
| Device URAM capacity | 80 × 288 Kbits | 2.88 MB | — |

---

## File Map

| File | Description |
|---|---|
| `fpga_utils.h` | All conv primitives (v4: HWC, OC_TILE=2, partial acc, tree) |
| `mbconv_160_top.cpp` | Top-level, partition pragmas (in/w_conv factor=16) |
| `mbconv_160_top.h` | Constants: IN_CH=32, HID=128, OUT_CH=64 |
| `fpga_types.h` | Data types, device constants |
| `testbench_mbconv160.cpp` | C-sim testbench |
| `report2/` | Synthesis reports for Run 1 (2026-06-02) |
