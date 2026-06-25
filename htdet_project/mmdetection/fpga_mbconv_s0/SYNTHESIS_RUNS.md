# mbconv_s0 Synthesis Run History

**Module**: `mbconv_s0_top` — Stage-0 MBConv (STEM→STAGE0)  
**Config**: in_ch=16 (STEM_CH), out_ch=32 (STAGE0_CH), expand=4, stride=1, hid=64  
**Device**: xczu28dr-ffvg1517-2-e (ZynqUltraScale+), Clock: 5 ns (200 MHz)  
**Layers**: expand 1×1 (16→64) + DW 3×3 stride=1 (64) + proj 1×1 (64→32) [no residual: 16≠32]  
**Spatial**: H=W=160 (STEM_H/STEM_W)

---

## Key Parameters

| Layer | in_ch | out_ch | IC tiles | HW pixels | Notes |
|---|---|---|---|---|---|
| expand 1×1 | 16 | 64 | **1** (16/16) | 25,600 | Single tile — II behaviour unknown |
| DW 3×3 | 64 | 64 | — | 25,600 | stride=1, same as s1b/s1c but smaller ch |
| proj 1×1 | 64 | 32 | 4 (64/16) | 25,600 | Standard, clean 4-tile |

---

## ⚠ URAM Warning

| Buffer | Size | vs Device URAM |
|---|---|---|
| ex_buf [160×160×64 × float32] | 6.25 MB | 217% over (2.88 MB capacity) |
| dw_buf [160×160×64 × float32] | 6.25 MB | 217% over |
| **Total** | **12.5 MB** | **434% ⚠️** |

Synthesis completes; P&R requires spatial tiling.

---

## Expected Latency (estimate)

**Expand** (OC_TILE=4, IC_T has only 1 tile):
```
OC groups = 64/4 = 16
Outer trip = 16 × 25,600 = 409,600
Per outer iter ≈ IC_T(1 tile, ~143 cy) + tree + BN/SiLU ≈ ~220 cycles
Total expand ≈ 409,600 × 220 ≈ 90M cycles ≈ 450 ms
```
*(Single IC tile may give II=1 or II=2 — observe from Run 1)*

**DW** (64 ch, 160×160, stride=1, II=5):
```
Trip = 64 × 160 × 160 = 1,638,400
Total ≈ 5 × 1,638,400 ≈ 8.2M cycles ≈ 41 ms
```

**Proj** (OC_TILE=4, 4 IC tiles, HLS auto-flattens):
```
Trip_flat = 8 × 25,600 × 4 = 819,200
At II=2: ≈ 1.64M cycles ≈ 8 ms
```

**Total estimate: ~100M cycles ≈ 500 ms**

---

## Synthesis Runs

| Date | OC_TILE | Total cycles | @ 200 MHz | DSP | URAM | Notes |
|---|---|---|---|---|---|---|
| pending | 4 | TBD | TBD | TBD | TBD | Run 1 — observe expand IC_T II |

---

## Pending Optimizations

| Optimization | Target | Status |
|---|---|---|
| Run 1 baseline | Characterise II for single-tile expand | Planned |
| OC_TILE=8 for expand | 2× outer trip reduction | After Run 1 |
| Spatial tiling (ex_buf/dw_buf) | URAM 434% → P&R feasible | For impl |

---

## File Map

| File | Description |
|---|---|
| `fpga_utils.h` | HWC conv primitives (v4, OC_TILE=4, from mbconv_40) |
| `mbconv_s0_top.cpp` | Top-level, bind_storage, partition pragmas |
| `mbconv_s0_top.h` | Constants: MBCONVS0_IN_CH=16, HID=64, OUT_CH=32 |
| `fpga_types.h` | Shared data types |
| `testbench_mbconv_s0.cpp` | C-sim testbench |
