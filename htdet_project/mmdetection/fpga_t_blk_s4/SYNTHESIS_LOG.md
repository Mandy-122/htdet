# transformer_block_s4 — HLS Synthesis Log

**Target**: Vitis HLS 2022.2, device `xczu9eg-ffvb1156-2-e` (ZCU102), clock 5 ns (200 MHz)  
**Module**: `transformer_blk_s4_top` — MobileViT S4 Transformer Block stack  
**Weight types**: all `float` (transformer layers stay float in W8A32 scheme)  
**Activation types**: `float` (W8A32)

---

## Stage Parameters

| Parameter | Value | Notes |
|---|---|---|
| in_ch | 160 | Input channels to MobileViTBlock |
| transformer_dim (DIM) | 240 | Attention/MLP embedding dimension |
| depth | 3 | Number of transformer blocks stacked |
| patch_size | 2×2 | Spatial patch for token extraction |
| seq (SEQ) | 25 | Patches per image: (10/2)×(10/2) for 320×320 input |
| heads | 4 | Attention heads |
| head_dim | 60 | DIM/heads = 240/4 |
| mlp_hidden | 480 | Feed-forward hidden = 2×DIM |

**Input feature map**: 10×10 (stride=32 from 320×320 input)

---

## Design Decisions — Inherited from S2

All HLS pragmas and structural fixes proven through S2 Steps 1–5 and confirmed in S3. Applied directly at full scale.

| Fix | Implementation | Notes specific to S4 |
|---|---|---|
| LN_MEAN/LN_VAR II=1 | Tile-based local `tile_sum` | N_TILES=DIM/4=60 |
| linear_f II=in_dim/2+1 | `UNROLL factor=2` matching dual-port BRAM | II=121 (in=240), II=241 (in=480) |
| MHSA_SM PIPELINE | `PIPELINE` + `DEPENDENCE variable=a_buf inter false` | SEQ=25 → II=5 (excellent) |
| MHSA_OK_T tile II=1 | `TB_SM_TILE=5` (25 not divisible by 4!) | Tile=5, Tiles=5 |
| SM_MAX PIPELINE | `PIPELINE II=1` instead of UNROLL | SEQ=25: 25 simultaneous reads from 10 ports → II=3 |
| SM_SUM tiled | `TB_SM_TILES=5` tile pipeline + tree-reduce | 5 tiles of 5 elements |
| a_buf partition | `cyclic factor=5` | SEQ²=625, even complete partition would be fine; factor=5 chosen for consistency |
| v_buf partition | `cyclic factor=7` | **DIM=240 divisible by 5! gcd(240,5)=5 → must use factor=7** |
| q_buf/k_buf | `cyclic factor=TB_HEAD_DIM=60` | MHSA_D unrolls 60 reads; 60 unique banks → II=1 |

---

## Key Difference from S2/S3: DIM=240 and SM_TILE=5

**Why v_buf factor=7 (not 5):**  
DIM=240 = 2⁴ × 3 × 5 → gcd(240, 5) = 5 ≠ 1.  
Using factor=5: all ki×240 values map to bank 0 → catastrophic II=∞.  
Factor=7: gcd(240, 7) = 1 (7 is prime, 240%7=2).

**Why SM_TILE=5 (not 4):**  
SEQ=25 = 5×5. 25 is not divisible by 4 → TB_SM_TILE=4 would leave remainder.  
25/5=5 tiles of 5 elements each → TB_SM_TILE=5, TB_SM_TILES=5.

---

## Partition Factor Verification for S4 (DIM=240)

**v_buf (factor=7):**  
MHSA_OK_T reads `v_buf[(t*5+kt)*240 + h*60 + d]` for kt=0..4 per tile:
```
{0, 240, 480, 720, 960} mod 7 = {0, 2, 4, 6, 1}
```
All 5 distinct → II=1 on v_buf access per tile ✓

**q_buf/k_buf (factor=60):**  
MHSA_D reads `q_buf[qi*240 + h*60 + d]` for d=0..59 (UNROLL):  
Bank = (h*60+d) % 60 = d → 60 unique banks → II=1 on MHSA_QI_MHSA_KI ✓

**a_buf (factor=5):**  
MHSA_OK_T reads `a_buf[qi*25 + t*5 + kt]` for kt=0..4:  
Banks = {base+0, base+1, base+2, base+3, base+4} % 5 → all 5 distinct ✓  
MHSA_SM reads/writes sequentially (PIPELINE, 1 access/stage) → fine ✓

---

## Overall Resource & Latency Summary

| Run | DIM | Latency (cycles) | Latency (ms) | BRAM_18K | DSP | FF | LUT | Timing (ns) |
|-----|-----|-----------------|-------------|----------|-----|----|-----|-------------|
| **S4 Full** | 240 | **18,061,423** | **90.3** | 274 (15%) | 357 (14%) | 276,387 (50%) | 270,700 (98%) | **4.451 ✅** |

Available resources on xczu9eg-ffvb1156-2-e (ZCU102):  
BRAM_18K=1,824 · DSP=2,520 · FF=548,160 · LUT=274,080

> ⚠️ LUT at 98% is very tight. HLS estimates are conservative (5–15% optimized away  
> by Vivado), but implementation may be challenging. Monitor routing congestion.  
> Fix if needed: `#pragma HLS PIPELINE II=2` on LF_S_LF_OD → halves mux LUTs,  
> doubles II to 242/482, adds ~90 ms latency.

---

## Per-Module Latency Breakdown (cycles)

| Pipeline Module | Trip | II | Latency | Notes |
|---|---|---|---|---|
| **COPY_IN** | 6,000 | 1 | 6,002 | SEQ×DIM=25×240 |
| **LN_MEAN** (each, ×2) | 60 | 1 | 94 | DIM/4=60 tiles ✅ |
| **LN_VAR** (each, ×2) | 60 | 1 | 107 | ✅ |
| **LN_NORM** (each, ×2) | 240 | 1 | 269 | ✅ |
| **LF_S_LF_OD** Q proj | 6,000 | **121** | 727,809 | II=240/2+1=121 ✅ |
| **LF_S_LF_OD1** K proj | 6,000 | **121** | 727,809 | Same as Q |
| **LF_S_LF_OD2** V proj | 6,000 | **121** | 727,809 | Same as Q |
| **LF_S_LF_OD3** O proj | 6,000 | **121** | 727,809 | Same as Q |
| **LF_S_LF_OD7** fc1 | 12,000 | **121** | 1,453,809 | in_dim=240, trip=25×480 |
| **LF_S_LF_OD8** fc2 | 6,000 | **241** | 1,449,609 | in_dim=480, II=480/2+1=241 ✅ |
| **MHSA_QI_MHSA_KI** | 625 | **1** | 1,118 | Trip=25×25=625, II=1 ✅ |
| **MHSA_CPY** | 6,000 | 1 | 6,002 | II=1 ✅ |
| **MHSA_SM** | 25 | **5** | 311 | 25 read/write×2=50 accesses / 10 ports → II=5 ✅ |
| **MHSA_OQ_MHSA_OD_MHSA_OK_T** | 7,500 | **3** | 22,653 | Trip=25×60×5, II=3 (FP latency) |
| **AR** residual (each, ×2) | 6,000 | 1 | 6,011 | ✅ |
| **MLP_ACT** SiLU | 12,000 | 1 | 12,021 | ✅ |
| **COPY_OUT** | 6,000 | 1 | 6,002 | ✅ |
| **VITIS_LOOP_84_1** LN outer (×2) | 25 | — | 37,700 | Tile FSM ×25 tokens |
| **DEPTH loop** (×3) | 3 | — | 18,049,416 | 3×6,016,472 per depth layer |
| **TOTAL** | | | **18,061,423** | |

---

## Per-Module Initiation Interval (II)

| Pipeline Module | Target | Actual | Root Cause |
|---|---|---|---|
| LN_MEAN | 1 | **1 ✅** | Tile approach (local tile_sum, no carry) |
| LN_VAR | 1 | **1 ✅** | Same |
| LN_NORM | 1 | **1 ✅** | No accumulation carry |
| LF_S_LF_OD Q/K/V/O | 1 | **121** | `w` BRAM dual-port: II = 240/2+1 |
| LF_S_LF_OD7 fc1 | 1 | **121** | in_dim=240 |
| LF_S_LF_OD8 fc2 | 1 | **241** | in_dim=480, II = 480/2+1 |
| MHSA_QI_MHSA_KI | 1 | **1 ✅** | q_buf cyclic factor=60=TB_HEAD_DIM |
| MHSA_SM | 1 | **5** | a_buf factor=5 (10 ports) ÷ 50 accesses (25 reads + 25 writes) |
| MHSA_OQ_MHSA_OD_MHSA_OK_T | 1 | **3** | FP multiply latency constraint with 5-element tile |
| AR (residual) | 1 | **1 ✅** | — |
| MLP_ACT (SiLU) | 1 | **1 ✅** | — |

**II formula for linear_f**: II = in_dim / 2 + 1 (BRAM dual-port: 2 reads/cycle, in_dim fully unrolled)

**MHSA_SM II=5 (best among all stages):** SEQ=25 is tiny — only 25 read+write operations across 10 BRAM ports = II=5.  
Contrast: S2(SEQ=400)→II=80, S3(SEQ=100)→II=20, S4(SEQ=25)→**II=5** ✓

---

## Latency Breakdown by Category

| Category | Cycles | % of Total |
|---|---|---|
| Linear projections (Q+K+V+O) ×3 depth | 4×727,809×3 = 8,733,708 | **48.3%** |
| fc1+fc2 ×3 depth | (1,453,809+1,449,609)×3 = 8,710,254 | **48.2%** |
| Weighted V-sum (MHSA_OQ_T) ×4 heads×3 depth | 22,653×4×3 = 271,836 | **1.5%** |
| MHSA attention scores ×4 heads×3 depth | 1,118×4×3 = 13,416 | **0.07%** |
| Layer norm FSM, residuals, I/O, SiLU | ~332,209 | **1.8%** |
| **TOTAL** | **18,061,423** | **100%** |

Linear projections (Q/K/V/O + fc1 + fc2) = **96.5% of total** — highest linear dominance across all stages due to smallest SEQ (least MHSA cost).

---

## Full Stage Comparison: S2 vs S3 vs S4

| Metric | S2 | S3 | S4 |
|---|---|---|---|
| DIM | 144 | 192 | **240** |
| SEQ | 400 | 100 | **25** |
| DEPTH | 2 | 4 | **3** |
| Total cycles | 94,039,743 | 65,031,383 | **18,061,423** |
| Latency | 470 ms | 325 ms | **90 ms** |
| LF_S_LF_OD II (Q/K/V/O) | 73 | 97 | **121** |
| fc2 II | 145 | 193 | **241** |
| MHSA_OQ trip | 1,440,000 | 120,000 | **7,500** |
| MHSA_SM II | 80 | 20 | **5** |
| BRAM utilization | 69% (est.) | 23% | **15%** |
| LUT utilization | 98% (est.) | 95% | **98%** |
| DSP utilization | 6% (est.) | 13% | **14%** |

S4 is the fastest (90 ms) due to tiny SEQ=25 — attention is near-zero cost.  
S4 has the highest DSP usage (357, 14%) because DIM=240 uses more FP units per linear projection.

---

## Known Issues and Notes

| Issue | Impact | Plan |
|---|---|---|
| LUT 98% on ZCU102 | Very tight; routing may fail | Add `#pragma HLS PIPELINE II=2` on LF_S_LF_OD if Vivado fails; halves mux LUTs, doubles II to 242/482 |
| MHSA_OQ_T II=3 (vs II=2 in S2/S3) | 271K cycles (1.5% of total) — negligible | Caused by 5-element tile FP latency; accepted |
| v_buf factor=7 (vs 5 in S2/S3) | No performance impact | Required because DIM=240 divisible by 5; factor=7 gives {0,2,4,6,1} distinct banks |
| SM_TILE=5 (vs 4 in S2/S3) | No performance impact | Required because SEQ=25 not divisible by 4; 25/5=5 tiles |
| linear_f dominates (96.5%) | Fundamental BRAM bandwidth limit | Needs wider memory bus (4-port BRAM or split w) to reduce II from 121/241 |

---

## Without-Pragma Expected Latency (Estimated)

**Basis**: Formulas calibrated from the actual S2 no-pragma run (measured: 1,228,024,873 cycles).  
**Method**: Scale each pipeline module using the observed per-element cost model.

### Derivation Formulas (from S2 no-pragma ground truth)

| Formula | Value | Derivation |
|---|---|---|
| `linear_f` iter_latency | `8 × in_dim + 14` cycles | FP-add accumulation II=8 (loop-carry), 1 weight read/cycle, +14 outer overhead. S2 check: 8×144+14=1,166 ✓ |
| `MHSA_QI_KI` per head | `SEQ² × (HEAD_DIM/2) + 300` | No q/k_buf partition → BRAM dual-port: II=HEAD_DIM/2. S2 check: 160,000×18+300≈2,880,300 ✓ |
| `MHSA_SM` iter_latency/token | `6.5 × SEQ` | 4 serial passes over SEQ softmax elements at ~1.6 cyc/elem. S2 check: 6.5×400=2,600 ≈ 2,596 ✓ |
| `MHSA_OQ` iter_latency/(qi,d) | `2.5 × SEQ + 50` | Serial V-buffer reads over SEQ keys, BRAM II≈2.5. S2 check: 2.5×400+50=1,050 ✓ |
| `LN` per token | `(DIM/144) × 1,066` | Scales with DIM: larger LN_MRED/VRED reduction tree + longer NORM loop. S2 check: 1×1,066 ✓ |

### S4 Module-by-Module Estimate (DIM=240, SEQ=25, HEAD_DIM=60, MLP_HID=480, DEPTH=3)

**iter_latency (no-pragma):**
- in_dim=240: 8×240+14 = **1,934** cycles
- in_dim=480: 8×480+14 = **3,854** cycles

**Per transformer_block:**

| Module | Trips | iter_latency | Cycles | % of block |
|---|---|---|---|---|
| Q projection (LF_S_LF_OD) | 25×240=6,000 | 1,934 | 11,604,000 | 12.4% |
| K projection | 6,000 | 1,934 | 11,604,000 | 12.4% |
| V projection | 6,000 | 1,934 | 11,604,000 | 12.4% |
| O projection | 6,000 | 1,934 | 11,604,000 | 12.4% |
| fc1 (LF_S_LF_OD7) | 25×480=12,000 | 1,934 | 23,208,000 | 24.8% |
| fc2 (LF_S_LF_OD8) | 25×240=6,000 | 3,854 | 23,124,000 | 24.7% |
| MHSA_QI_KI (×4 heads) | 625 trips/head | II=30 → 19,050/head | 76,200 | 0.1% |
| MHSA_SM (×4 heads) | 25 tokens | 163 cyc/token | 16,300 | ~0% |
| MHSA_OQ (×4 heads) | 1,500 trips/head | 113 cyc/trip | 678,000 | 0.7% |
| MHSA_CPY | 6,000 | — | 6,002 | ~0% |
| LN ×2 calls | 25 tokens each | (240/144)×1,066=1,777 | 88,850 | 0.1% |
| AR ×2 | 6,000 | II=1 | 12,022 | ~0% |
| MLP_ACT | 12,000 | II=1 | 12,021 | ~0% |
| **Per block** | | | **~93,637,395** | |

**Total estimate:**
- DEPTH×3: 3 × 93,637,395 = **280,912,185 cycles**
- COPY_IN + COPY_OUT: 12,004 cycles
- **Grand total: ~280,924,189 cycles ≈ 1.405 seconds at 200 MHz**

### S4 Without-Pragma Summary

| Metric | With Pragma | Without Pragma (Est.) | Ratio |
|---|---|---|---|
| **Total latency** | 18,061,423 cycles = 90 ms | **~280,924,000 cycles ≈ 1.405 sec** | **~15.6× slower** |
| Dominant module | linear_f (96.5%) | linear_f (99.1%) | Same bottleneck |
| linear_f cycle total | ~18.3M per block | ~92.7M per block | **~5.1× worse** |
| MHSA attention | ~270K per block | ~781K per block | ~2.9× worse |
| Timing (estimated) | 4.35 ns | ~3.9 ns (fewer parallel FUs) | Better timing |

**Key observation for S4**: With SEQ=25, the attention mechanism (MHSA) is already tiny (1.5% of latency) even *with* pragmas. Without pragmas MHSA still contributes only 0.8% — it was never the bottleneck. The full 15.6× slowdown comes from linear_f alone. The UNROLL pragma's impact on BRAM read bandwidth (II: 121→II=1 inner loop accumulation) is the single factor causing the regression.

### Cross-Stage Without-Pragma Comparison

| Stage | DIM | SEQ | DEPTH | With Pragma | **Without Pragma (Est.)** | **Ratio** |
|---|---|---|---|---|---|---|
| S2 | 144 | 400 | 2 | 94,039,743 (470 ms) | **1,228,024,873 (6.14 s)** | **13.1×** |
| S3 | 192 | 100 | 4 | 65,031,383 (325 ms) | **~980,649,000 (4.90 s)** | **~15.1×** |
| S4 | 240 | 25 | 3 | 18,061,423 (90 ms) | **~280,924,000 (1.41 s)** | **~15.6×** |

**Why ratio increases S2 → S3 → S4:**
- Larger DIM means the UNROLL on the inner weight-accumulation loop provides bigger benefit (more parallel MACs per BRAM read)
- Smaller SEQ means MHSA (which also benefits from pragmas) is a shrinking fraction, making the linear_f speedup the sole contributor to the ratio
- The 15× plateau suggests the FP-accumulation serialisation (II=8 without pragma, II=1 with) is the fundamental multiplier, giving an 8× ceiling on that path, but UNROLL×2 adds another 2× via BRAM dual-port, yielding ~16× theoretical max; actual 15× matches well
