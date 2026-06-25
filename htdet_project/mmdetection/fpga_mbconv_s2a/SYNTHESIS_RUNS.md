# mbconv_s2a Synthesis Run History

**Module**: `mbconv_s2a_top` — Stage-2a MBConv (C1→C2 downsampler)  
**Config**: in_ch=64 (C1_CH), out_ch=96 (C2_CH), expand=4, stride=2, hid=256  
**Device**: xczu28dr-ffvg1517-2-e (ZynqUltraScale+), Clock: 5 ns (200 MHz)  
**Layers**: expand 1×1 (64→256) + DW 3×3 stride=2 (256) + proj 1×1 (256→96) [no residual]  
**Spatial**: Input H=W=80 (C1) → Output H=W=40 (C2)

---

## Key Parameters

| Layer | in_ch | out_ch | IC tiles | Input HW | Output HW |
|---|---|---|---|---|---|
| expand 1×1 | 64 | 256 | 4 (64/16) | 6,400 | 6,400 |
| DW 3×3 s=2 | 256 | 256 | — | 6,400 | 1,600 |
| proj 1×1 | 256 | 96 | 16 (256/16) | 1,600 | 1,600 |

**Expand is identical to `fpga_mbconv_80` expand** (IC=64, hid=256, 80×80).  
Proj differs: 256→96 (24 OC groups) vs mbconv_80's 256→64 (16 OC groups).

---

## ⚠ URAM Budget

| Buffer | Size | vs Device URAM |
|---|---|---|
| ex_buf [80×80×256 × float32] | 6.25 MB | 217% over ⚠️ |
| dw_buf [40×40×256 × float32] | 1.5625 MB | **54% ✓ fits** |
| **Total** | **7.8 MB** | **271% ⚠️** |

Only `ex_buf` overflows — same situation as `fpga_mbconv_160`.

---

## Expected Latency (estimate, calibrated from fpga_mbconv_80)

**Expand** (IC=64, OC=256, OC_TILE=4, 80×80 spatial):
```
Outer trips = (256/4) × 6,400 = 64 × 6,400 = 409,600
Per iter ≈ 289 cycles (same as mbconv_80 expand — identical IC/OC/spatial)
Total ≈ 409,600 × 289 ≈ 118M cycles ≈ 592 ms
```

**DW** (256 ch, stride=2, 80×80 input → 40×40 output, II=5):
```
Trip = 256 × 40 × 40 = 409,600
Total ≈ 5 × 409,600 ≈ 2.05M cycles ≈ 10 ms
```
*(Half the mbconv_80 DW trip count due to stride=2 output)*

**Proj** (IC=256, OC=96, OC_TILE=4, 40×40 spatial):
```
Outer trips_flat = 24 × 1,600 × 16 = 614,400 at II≈2-4
Total ≈ 1.2–2.5M cycles ≈ 6–12 ms
```

**Total estimate: ~122M cycles ≈ 610 ms**

*(Expand dominates at ~97%, essentially same as fpga_mbconv_80)*

---

## Comparison with fpga_mbconv_80

| Metric | fpga_mbconv_80 (64→64, s=1) | fpga_mbconv_s2a (64→96, s=2) | Diff |
|---|---|---|---|
| Expand channels | 64→256 | **same** | identical |
| Expand spatial | 80×80 | **same** | identical |
| DW trips | 256×80×80=1.6M | 256×40×40=0.41M | 4× fewer (stride-2) |
| Proj OC groups | 16 (64/4) | **24 (96/4)** | +50% |
| Residual add | yes (409K elems) | **no** | removed |
| ex_buf URAM | 1133% (Run 1) | ~217% | 5× better |

---

## Synthesis Runs

| Date | OC_TILE | Total cycles | @ 200 MHz | DSP | URAM | Notes |
|---|---|---|---|---|---|---|
| pending | 4 | TBD | TBD | TBD | TBD | Run 1 |

---

## Pending Optimizations

| Optimization | Target | Status |
|---|---|---|
| Run 1 baseline | Confirm expand II matches mbconv_80 | Planned |
| OC_TILE=8 for expand | ~2× speedup (same logic as mbconv_80) | After Run 1 |
| ex_buf spatial tiling | ex_buf 217% → P&R feasible | For impl |

---

## File Map

| File | Description |
|---|---|
| `fpga_utils.h` | HWC conv primitives (v4, OC_TILE=4, from mbconv_40) |
| `mbconv_s2a_top.cpp` | Top-level, bind_storage, partition pragmas |
| `mbconv_s2a_top.h` | Constants: IN_CH=64, HID=256, OUT_CH=96 |
| `fpga_types.h` | Shared data types |
| `testbench_mbconv_s2a.cpp` | C-sim testbench |
