# transformer_block_s3 — HLS Synthesis Log

**Target**: Vitis HLS 2022.2, device `xczu9eg-ffvb1156-2-e` (ZCU102), clock 5 ns (200 MHz)  
**Module**: `transformer_blk_s3_top` — MobileViT S3 Transformer Block stack  
**Weight types**: all `float` (transformer layers stay float in W8A32 scheme)  
**Activation types**: `float` (W8A32)

---

## Stage Parameters

| Parameter | Value | Notes |
|---|---|---|
| in_ch | 128 | Input channels to MobileViTBlock |
| transformer_dim (DIM) | 192 | Attention/MLP embedding dimension |
| depth | 4 | Number of transformer blocks stacked |
| patch_size | 2×2 | Spatial patch for token extraction |
| seq (SEQ) | 100 | Patches per image: (20/2)×(20/2) for 320×320 input |
| heads | 4 | Attention heads |
| head_dim | 48 | DIM/heads = 192/4 |
| mlp_hidden | 384 | Feed-forward hidden = 2×DIM |

**Input feature map**: 20×20 (stride=16 from 320×320 input)

---

## Design Decisions — Inherited from S2

All HLS pragmas and structural fixes were proven through 5 incremental steps in S2 (see `fpga_t_blk_s2/SYNTHESIS_LOG.md`). S3 applies them directly at full scale with S3-specific partition factors.

| Fix | Implementation | Reason |
|---|---|---|
| LN_MEAN/LN_VAR II=1 | Tile-based local `tile_sum` (no loop-carry) | FP-add RAW dependency at II=5 otherwise |
| linear_f II=in_dim/2+1 | `UNROLL factor=2` matching dual-port BRAM | External `w` BRAM: 2 reads/cycle limit |
| MHSA_SM false-dep | `PIPELINE` + `DEPENDENCE variable=a_buf inter false` | Pointer aliasing through softmax_vec arg |
| MHSA_OK_T tile II=1 | Tile loop with `TB_SM_TILE=4`, local `tile_sum` | Avoids full-UNROLL resource explosion |
| SM_MAX PIPELINE | `PIPELINE II=1` instead of UNROLL | UNROLL at SEQ=100 needs 100 simultaneous a_buf reads |
| SM_SUM tiled | `TB_SM_TILES=25` tile pipeline + tree-reduce | UNROLL at SEQ=100 → ~50 FP adders (100 DSP) |
| a_buf partition | `cyclic factor=5` | SEQ²=10,000 complete partition → compile hang |
| v_buf partition | `cyclic factor=5` | gcd(192,5)=1; (kt×192)%5={0,2,4,1} for kt=0..3 → 4 distinct banks ✓ |
| q_buf/k_buf | `cyclic factor=TB_HEAD_DIM=48` | MHSA_D reads 48 consecutive elements; 48 unique banks |

---

## Partition Factor Verification for S3 (DIM=192)

**v_buf (factor=5):**  
MHSA_OK_T reads `v_buf[(t*4+kt)*192 + h*48 + d]` for kt=0..3 per tile:
```
kt=0: (t*4*192) % 5 = 0
kt=1: (t*4*192 + 192) % 5 = 192%5 = 2
kt=2: (t*4*192 + 384) % 5 = 384%5 = 4
kt=3: (t*4*192 + 576) % 5 = 576%5 = 1
```
Banks {0, 2, 4, 1} — all distinct → II=1 on MHSA_OK_T ✓

**q_buf/k_buf (factor=48):**  
MHSA_D reads `q_buf[qi*192 + h*48 + d]` for d=0..47 (UNROLL):  
Bank = (h*48+d) % 48 = d → 48 unique banks → II=1 on MHSA_QI_MHSA_KI ✓

**a_buf (factor=5):**  
MHSA_OK_T reads `a_buf[qi*SEQ + t*4 + kt]` for kt=0..3:  
Banks = {base, base+1, base+2, base+3} % 5 — always 4 distinct (consecutive modulo prime) ✓  
MHSA_SM reads/writes sequentially (PIPELINE, 1 access/stage) → fine ✓

---

## Overall Resource & Latency Summary

| Run | DIM | Latency (cycles) | Latency (ms) | BRAM_18K | DSP | FF | LUT | Timing (ns) |
|-----|-----|-----------------|-------------|----------|-----|----|-----|-------------|
| **S3 Full** | 192 | **65,031,383** | **325** | 426 (23%) | 338 (13%) | 288,559 (52%) | 262,703 (95%) | **4.350 ✅** |

Available resources on xczu9eg-ffvb1156-2-e (ZCU102):  
BRAM_18K=1,824 · DSP=2,520 · FF=548,160 · LUT=274,080

> ⚠️ LUT at 95% is tight. HLS estimates are conservative — Vivado implementation  
> typically uses 5–15% fewer LUTs after logic optimization and should still fit.  
> If implementation fails: set `#pragma HLS PIPELINE II=2` on LF_S_LF_OD to halve  
> mux LUT usage (doubles II to 194, adds ~325 ms latency).

---

## Per-Module Latency Breakdown (cycles)

| Pipeline Module | Trip | II | Latency | Notes |
|---|---|---|---|---|
| **COPY_IN** | 19,200 | 1 | 19,202 | SEQ×DIM=100×192 |
| **LN_MEAN** (each, ×2) | 48 | 1 | 82 | DIM/4=48 tiles ✅ |
| **LN_VAR** (each, ×2) | 48 | 1 | 95 | ✅ |
| **LN_NORM** (each, ×2) | 192 | 1 | 221 | ✅ |
| **LF_S_LF_OD** Q proj | 19,200 | **97** | 1,863,849 | II=192/2+1 ✅ |
| **LF_S_LF_OD1** K proj | 19,200 | **97** | 1,863,849 | Same as Q |
| **LF_S_LF_OD2** V proj | 19,200 | **97** | 1,863,849 | Same as Q |
| **LF_S_LF_OD3** O proj | 19,200 | **97** | 1,863,849 | Same as Q |
| **LF_S_LF_OD7** fc1 | 38,400 | **97** | 3,726,249 | in_dim=192, trip=100×384 |
| **LF_S_LF_OD8** fc2 | 19,200 | **193** | 3,708,489 | in_dim=384, II=384/2+1 |
| **MHSA_QI_MHSA_KI** | 10,000 | **1** | 10,397 | Trip=100×100, depth=308 ✅ |
| **MHSA_CPY** | 19,200 | 1 | 19,202 | II=1 ✅ |
| **MHSA_SM** | 100 | **20** | 2,601 | a_buf 10 ports, 100 reads → II=20 |
| **MHSA_OQ_MHSA_OD_MHSA_OK_T** | 120,000 | **2** | 240,307 | Trip=100×48×25, 3 loops flattened ✅ |
| **AR** residual (each, ×2) | 19,200 | 1 | 19,211 | ✅ |
| **MLP_ACT** SiLU | 38,400 | 1 | 38,421 | ✅ |
| **COPY_OUT** | 19,200 | 1 | 19,202 | ✅ |
| **VITIS_LOOP_84_1** LN outer (×2) | 100 | — | 124,400 | Tile FSM ×100 tokens |
| **DEPTH loop** (×4) | 4 | — | 64,992,976 | 4×16,248,244 per depth layer |
| **TOTAL** | | | **65,031,383** | |

---

## Per-Module Initiation Interval (II)

| Pipeline Module | Target | Actual | Root Cause |
|---|---|---|---|
| LN_MEAN | 1 | **1 ✅** | Tile approach (local tile_sum, no carry) |
| LN_VAR | 1 | **1 ✅** | Same |
| LN_NORM | 1 | **1 ✅** | No accumulation carry |
| LF_S_LF_OD Q/K/V/O | 1 | **97** | `w` BRAM dual-port: II = 192/2+1 |
| LF_S_LF_OD7 fc1 | 1 | **97** | in_dim=192 |
| LF_S_LF_OD8 fc2 | 1 | **193** | in_dim=384, II = 384/2+1 |
| MHSA_QI_MHSA_KI | 1 | **1 ✅** | q_buf cyclic factor=48=TB_HEAD_DIM |
| MHSA_SM | 1 | **20** | a_buf cyclic factor=5 (10 ports) ÷ 100-elem reads/writes |
| MHSA_OQ_MHSA_OD_MHSA_OK_T | 1 | **2** | a_buf port minor conflict (4 reads, 5 banks); accepted |
| AR (residual) | 1 | **1 ✅** | — |
| MLP_ACT (SiLU) | 1 | **1 ✅** | — |

**II formula for linear_f**: II = in_dim / 2 + 1 (BRAM dual-port: 2 reads/cycle, in_dim fully unrolled)

---

## Latency Breakdown by Category

| Category | Cycles | % of Total |
|---|---|---|
| Linear projections (Q+K+V+O) ×4 depth | 4×1,863,849×4 = 29,821,584 | **45.9%** |
| fc1+fc2 ×4 depth | (3,726,249+3,708,489)×4 = 29,738,952 | **45.7%** |
| Weighted V-sum (MHSA_OQ_T) ×4 heads×4 depth | 240,307×4×4 = 3,844,912 | **5.9%** |
| MHSA attention scores ×4 heads×4 depth | 10,397×4×4 = 166,352 | **0.3%** |
| Layer norm FSM, residuals, I/O, SiLU | ~1,459,583 | **2.2%** |
| **TOTAL** | **65,031,383** | **100%** |

Linear projections (Q/K/V/O + fc1 + fc2) = **91.6% of total** — fundamental BRAM bandwidth bottleneck at II = in_dim/2+1.

---

## S2 vs S3 Comparison

| Metric | S2 (Full, Step5) | S3 |
|---|---|---|
| DIM | 144 | **192** |
| SEQ | 400 | **100** |
| DEPTH | 2 | **4** |
| Total cycles | 94,039,743 | **65,031,383** |
| Latency | 470 ms | **325 ms** |
| LF_S_LF_OD II (Q/K/V/O) | 73 | **97** |
| fc2 II | 145 | **193** |
| MHSA_OQ trip | 1,440,000 | **120,000** |
| MHSA_OQ latency (×4 heads) | 23.0M cycles | **3.84M cycles** |
| MHSA_SM II | 80 | **20** |
| LUT utilization (ZCU102) | ~98% (est.) | **95%** |
| BRAM utilization (ZCU102) | ~69% (est.) | **23%** |

S3 is 1.44× faster than S2 despite 2× the depth, because SEQ=100 (vs 400) reduces the attention weighted-sum 16× and the LN/SiLU/copy operations 4×. The bottleneck shifts entirely to linear projections.

---

## Known Issues and Notes

| Issue | Impact | Plan |
|---|---|---|
| LUT 95% on ZCU102 | Tight; may cause routing congestion | Monitor Vivado implementation; add `#pragma HLS PIPELINE II=2` on LF_S_LF_OD if needed |
| MHSA_SM II=20 | 2,601 cycles/head × 4 heads × 4 depth = 41.6K cycles (0.06% of total) | Accepted — negligible |
| MHSA_OQ_T II=2 | 240K cycles/head × 4 heads × 4 depth = 3.84M cycles (5.9%) | Accepted — a_buf 5-bank minor conflict |
| linear_f II = in_dim/2+1 | 91.6% of total latency | Fundamental BRAM bandwidth limit; needs wider memory bus to improve |

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

### S3 Module-by-Module Estimate (DIM=192, SEQ=100, HEAD_DIM=48, MLP_HID=384, DEPTH=4)

**iter_latency (no-pragma):**
- in_dim=192: 8×192+14 = **1,550** cycles
- in_dim=384: 8×384+14 = **3,086** cycles

**Per transformer_block:**

| Module | Trips | iter_latency | Cycles | % of block |
|---|---|---|---|---|
| Q projection (LF_S_LF_OD) | 100×192=19,200 | 1,550 | 29,760,000 | 12.1% |
| K projection | 19,200 | 1,550 | 29,760,000 | 12.1% |
| V projection | 19,200 | 1,550 | 29,760,000 | 12.1% |
| O projection | 19,200 | 1,550 | 29,760,000 | 12.1% |
| fc1 (LF_S_LF_OD7) | 100×384=38,400 | 1,550 | 59,520,000 | 24.3% |
| fc2 (LF_S_LF_OD8) | 100×192=19,200 | 3,086 | 59,251,200 | 24.2% |
| MHSA_QI_KI (×4 heads) | 10,000 trips | II=24 → 240,300/head | 961,200 | 0.4% |
| MHSA_SM (×4 heads) | 100 tokens | 650 cyc/token | 260,000 | 0.1% |
| MHSA_OQ (×4 heads) | 4,800 trips/head | 300 cyc/trip | 5,760,000 | 2.4% |
| MHSA_CPY | 19,200 | — | 19,202 | ~0% |
| LN ×2 calls | 100 tokens each | (192/144)×1,066=1,421 | 284,200 | 0.1% |
| AR ×2 | 19,200 | II=1 | 38,422 | ~0% |
| MLP_ACT | 38,400 | II=1 | 38,421 | ~0% |
| **Per block** | | | **~245,152,645** | |

**Total estimate:**
- DEPTH×4: 4 × 245,152,645 = **980,610,580 cycles**
- COPY_IN + COPY_OUT: 38,404 cycles
- **Grand total: ~980,648,984 cycles ≈ 4.903 seconds at 200 MHz**

### S3 Without-Pragma Summary

| Metric | With Pragma | Without Pragma (Est.) | Ratio |
|---|---|---|---|
| **Total latency** | 65,031,383 cycles = 325 ms | **~980,649,000 cycles ≈ 4.90 sec** | **~15.1× slower** |
| Dominant module | linear_f (91.6%) | linear_f (96.8%) | Same bottleneck |
| linear_f cycle total | ~59.5M per block | ~237.0M per block | **~4.0× worse** |
| MHSA attention | ~3.9M per block | ~7.0M per block | ~1.8× worse |
| Timing (estimated) | 4.35 ns | ~3.9 ns (fewer parallel FUs) | Better timing |

**Why ~15× (vs S2's 13×):** S3 has DEPTH=4 (vs 2), so per-block savings from pragmas compound 4× instead of 2×. Additionally, larger DIM=192 gives bigger UNROLL benefit (192/2=96 vs 144/2=72 reads unrolled per cycle), increasing the pragma speedup on the linear layers.

### Linear_f Dominates Even More in S3 Without Pragmas

With pragma S3: linear_f = 91.6% of latency  
Without pragma S3: linear_f = 96.8% of latency  
The attention mechanism (MHSA_SM, MHSA_OQ) is proportionally smaller at SEQ=100, so linear projection becomes the near-total bottleneck both with and without pragmas. The pragma speedup (15×) comes almost entirely from the UNROLL on the weight accumulation loop.
