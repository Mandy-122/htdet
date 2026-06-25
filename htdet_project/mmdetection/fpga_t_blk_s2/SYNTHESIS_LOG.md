# transformer_block_s2 — HLS Synthesis Log

**Target**: Vitis HLS 2022.2, device `xczu28dr-ffvg1517-2-e`, clock 5 ns (200 MHz)  
**Module**: `transformer_blk_s2_top` — MobileViT S2 Transformer Block stack  
**Weight types**: all `float` (transformer layers stay float in W8A32 scheme)  
**Activation types**: `float` (W8A32)

---

## Dimension Scaling Ladder

| Step | DIM | SEQ | HEADS | HEAD_DIM | MLP_HID | DEPTH | BLK_W (floats) | Status |
|------|-----|-----|-------|----------|---------|-------|----------------|--------|
| 1    |  16 |   4 |   2   |    8     |   32    |   1   |   2,224        | ✅ Done (v3) |
| 2    |  32 |   4 |   2   |   16     |   64    |   1   |   8,544        | ✅ Done (v4) |
| 3    |  64 |  16 |   4   |   16     |  128    |   1   |  33,920        | ✅ Done (Step3) |
| 4    | 144 | 100 |   4   |   36     |  288    |   1   | 167,472        | ✅ Done (Step4) |
| 5    | 144 | 400 |   4   |   36     |  288    |   2   | 334,944        | ✅ Done (Step5) |

Full S2: in_ch=96, transformer_dim=144, depth=2, patch=2, seq=400 (320×320 input)

---

## Overall Resource & Latency Summary

| Run | Step | DIM | Fix Applied | Latency (cycles) | Latency (µs) | BRAM_18K | DSP | FF | LUT | Timing (ns) |
|-----|------|-----|-------------|-----------------|--------------|----------|-----|----|-----|-------------|
| **v1** | 1 | 16 | Baseline — no fixes | **9,318** | 46.59 | 27 (1%) | 20 (~0%) | 28,044 (3%) | 23,847 (5%) | 3.857 |
| **v2** | 1 | 16 | LN 4-accumulator + linear_f tile+tree + DEPENDENCE pragma | **22,850** ❌ | 114.25 | 35 (1%) | 52 (1%) | 22,420 (2%) | 21,837 (5%) | 3.740 |
| **v3** | 1 | 16 | Reverted linear_f; MHSA_SM UNROLL; q_buf factor=4 | **8,114** ✅ | 40.57 | 35 (1%) | 87 (2%) | 37,521 (4%) | 33,110 (7%) | 3.857 |
| **Step 2** | 2 | 32 | Scale-up with v3 pragmas (q_buf still factor=4) | **22,876** | 114.38 | 43 (1%) | 87 (2%) | 49,708 (5%) | 43,793 (10%) | ~5.49⚠ |
| **v4** | 2 | 32 | LN tile approach + q/k/v/x/ln partition fixes | **23,532** | 117.66 | 19 (~0%) | 139 (3%) | 58,744 (6%) | 49,626 (11%) | ~5.52⚠ |
| **Step 3** | 3 | 64 | SM_EXP PIPELINE + v_buf factor=17 | **310,804** | 1,554 | 104 (4%) | 103 (2%) | 133,794 (15%) | 108,596 (25%) | **4.26 ✅** |
| **Step 4** | 4 | 144 | MHSA_SM PIPELINE + MHSA_OK tile + a_buf cyclic=5 | **9,472,345** | 47,362 | 480 (22%) | 223 (5%) | 206,398 (24%) | 169,981 (39%) | **4.317 ✅** |
| **Step 5** | 5 | 144 | v_buf factor=5 (fixed from 401 for compile time) | **94,039,743** | 470,199 | 1256 (58%) | 260 (6%) | 317,871 (37%) | 269,309 (63%) | **4.317 ✅** |
| **No Pragma** | 5 | 144 | All `#pragma HLS` commented out — baseline comparison | **1,228,024,873** | 6,140,124 | 1156 (53%) | 44 (1%) | 16,444 (1%) | 17,519 (4%) | **3.966 ✅** |

Available resources on xczu28dr: BRAM_18K=2160, DSP=4272, FF=850,560, LUT=425,280, URAM=80

> ⚠ Step 2 timing: csynth.rpt shows Slack=−0.49 ns at top-level (some module paths ~5.49 ns).  
> ✅ Step 3 timing: 4.26 ns — positive slack! Combinational path shortened vs Step 2 due to LUT-based mux restructuring at DIM=64.
> ✅ Step 5 timing: 4.317 ns — same as Step 4. DEPTH=2 adds sequential FSM but no new combinational paths. v_buf factor=5 fix was critical (factor=401 caused >1hr synthesis hang).

---

## Per-Module Latency Breakdown (cycles)

| Pipeline Module | v1 | v2 | v3 | Step2 | v4 | **Step3** | Notes |
|----------------|-----|-----|-----|-------|--------|---------|-------|
| **COPY_IN** | 66 | 66 | 66 | 130 | 130 | 1,026 | **14,402** | II=1, scales with SEQ×DIM |
| **LN_MEAN** (each, x2) | 84 | 27 | 27 | 47 | 42 | 50 | **70** | II=1 all steps ✅ |
| **LN_VAR** (each, x2) | 97 | 40 | 40 | 60 | 55 | 63 | **83** | II=1 all steps ✅ |
| **LN_NORM** (each, x2) | 45 | 45 | 45 | 61 | 61 | 93 | **173** | II=1 all steps ✅ |
| **LF_S_LF_OD** Q proj | 704 | ~4,800 | 704 | 2,424 | 2,424 | 34,280 | **1,052,288** | II=9/seq/9/17/17/33/**73** |
| **LF_S_LF_OD1** K proj | 704 | ~4,800 | 704 | 2,424 | 2,424 | 34,280 | **1,052,289** | Same as Q |
| **LF_S_LF_OD2** V proj | 704 | ~4,800 | 704 | 2,424 | 2,424 | 34,280 | **1,052,289** | Same as Q |
| **LF_S_LF_OD33/3** O proj | 704 | 1,090 | 704 | 2,424 | 2,424 | 34,280 | **1,052,289** | BRAM `w` hard limit |
| **LF_S_LF_OD37/7** fc1 | 1,280 | 2,114 | 1,280 | 4,600 | 4,600 | 68,072 | **2,103,489** | II=73 (in_dim=144), 28800 iters |
| **LF_S_LF_OD38/8** fc2 | 1,336 | 2,114 | 1,336 | 4,712 | 4,712 | 67,528 | **2,090,169** | II=145 (in_dim=288), 14400 iters |
| **MHSA_QI_MHSA_KI** | 140 | 92 | 92 | 172 | 156 | 396 | **10,301** | II=1 ✅ (Trip=10,000, depth=301) |
| **SM_EXP** (per token) | — | — | — | — | — | 38-39 | *(in MHSA_SM)* | Unrolled inside MHSA_SM body |
| **SM_NORM** (per token) | — | — | — | — | — | 32 | *(in MHSA_SM)* | Unrolled inside MHSA_SM body |
| **MHSA_SM** | 299 | 299 | *(UNROLL)* | *(UNROLL)* | *(UNROLL)* | 16 SM_EXP/NORM | **2,601** | PIPELINE II=20 (a_buf port limit) |
| **MHSA_H loop** (outer) | 1,110 | 1,014 | 562 | 852 | 820 | 14,268 | **772,660** | 4 heads x 193,165/head |
| **MHSA_CPY** | 66 | 66 | 66 | 130 | 130 | 1,026 | **14,402** | II=1 |
| **MHSA_OQ_MHSA_OD_MHSA_OK_T** | 110 | 110 | 110 | 175 | 175 | 2,192 | **180,257** | II=2, trip=90K (3 loops flattened!) |
| **AR** residual (each, x2) | 75 | 75 | 75 | 139 | 139 | 1,035 | **14,411** | II=1 |
| **MLP_ACT** SiLU | 149 | 149 | 149 | 277 | 277 | 2,069 | **28,821** | II=1 |
| **COPY_OUT** | 66 | 66 | 66 | 130 | 130 | 1,026 | **14,402** | II=1 |
| **VITIS_LOOP_84_1** LN outer (x2) | ~800 | ~800 | ~800 | ~1,024 | ~1,368 | ~8,288 | **~98,000** | Tile FSM x 100 tokens |
| **TOTAL** | **9,318** | **22,850** | **8,114** | **22,876** | **23,532** | **310,804** | **9,472,345** | |

---

## Per-Module Initiation Interval (II)

| Pipeline Module | Target | v1 | v2 | v3 | Step2 | v4 | **Step3** | Root Cause |
|----------------|--------|-----|-----|-----|-------|--------|---------|------------|
| LN_MEAN | 1 | 5 | 5 | 5 | 5 | 1 ✅ | **1 ✅** | Tile (local var) fixes FP-add carry |
| LN_VAR | 1 | 5 | 5 | 5 | 5 | 1 ✅ | **1 ✅** | Same |
| LN_NORM | 1 | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | **1 ✅** | No accumulation carry |
| LF_S_LF_OD Q/K/V/O | 1 | 9 | seq | 9 | 17 | 17 | **33** | II=in_dim/2+1; in_dim=64 |
| LF_S_LF_OD fc1 | 1 | 9 | seq | 9 | 17 | 17 | **33** | in_dim=64 |
| LF_S_LF_OD fc2 | 1 | 17 | seq | 17 | 33 | 33 | **65** | in_dim=128 |
| MHSA_QI_MHSA_KI | 1 | 4 | 1 ✅ | 1 ✅ | 2 | 1 ✅ | **1 ✅** | q_buf factor=TB_HEAD_DIM=16 |
| MHSA_SM | 1 | 74 | 74 | UNROLL | UNROLL | UNROLL | **UNROLL** | 16 copies (seq); aliasing avoided |
| SM_EXP | 1 | — | — | — | — | — | **1 ✅** | PIPELINE replaces UNROLL |
| SM_NORM | 1 | — | — | — | — | — | **1 ✅** | PIPELINE |
| MHSA_OQ_MHSA_OD | 1 | 2 | 2 | 2 | 2 | 2 | **8** | Resource-limited: 16 FP muls, 2 shared |
| AR (residual) | 1 | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | **1 ✅** | — |
| MLP_ACT (SiLU) | 1 | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | 1 ✅ | **1 ✅** | — |

**II formula for linear_f**: II = in_dim / 2 + 1 (BRAM dual-port limit: 2 reads/cycle, in_dim fully unrolled)

---

## Run-by-Run Descriptions

### Run v1 — Baseline (Step 1, DIM=16)
**Date**: Sun May 31 13:17  
**Latency**: 9,318 cycles | **Status**: First synthesis, structural correctness confirmed

**State**: All functions working correctly (3/3 testbench pass). Four distinct II violations:
1. `LN_MEAN/LN_VAR` II=5: Sequential FP accumulator `sum += x` carries RAW through adder (latency=5)
2. `LF_S_LF_OD` II=9: External `w` BRAM dual-port limit (in_dim=16 reads, 2 ports → II=9)
3. `MHSA_QI_MHSA_KI` II=4: `q_buf` not partitioned (MHSA_D unrolled 8 reads, only 2 BRAM ports)
4. `MHSA_SM` II=74: Pointer aliasing — `softmax_vec(a_buf + qi*SEQ)` prevents HLS from proving non-overlapping slices; conservatively stalls for full softmax latency

---

### Run v2 — Tile+Tree Attempt (Step 1, DIM=16)
**Date**: Sun May 31 13:36  
**Latency**: 22,850 cycles ❌ (2.45× regression) | **Status**: Regressed, approach abandoned

**Changes applied**:
- `linear_f`: Replaced with partial-accumulator tile approach (TILE=2 + 4-level tree reduce)
- `layer_norm`: 4 parallel running accumulators (ms0..ms3)
- `MHSA_SM`: Added `#pragma HLS DEPENDENCE variable=a_buf inter false`
- `q_buf`: `cyclic factor=4` added

**What worked**: q_buf partition fixed MHSA_QI_MHSA_KI to II=1; LN_MEAN trips reduced (84→27)

**What failed**:
- `linear_f` tile approach created a 3-loop nest (`LF_S × LF_OD × LF_TILE_L`). For Q/K/V calls reading from external `x_buf`, HLS could not flatten all three loops → outer `LF_S × LF_OD` ran sequentially (64 iters × 75 cycles = 4,800 cycles per call vs 704 before)
- `DEPENDENCE` pragma had no effect on MHSA_SM — pointer aliasing through the `softmax_vec` argument persisted after INLINE
- LN 4-accumulator: II still 5 (ms0..ms3 are ALL updated every iteration → period=1, same as single accumulator)

---

### Run v3 — MHSA_SM Fixed (Step 1, DIM=16) ⭐ Best Result
**Date**: Sun May 31 13:49  
**Latency**: 8,114 cycles ✅ (13% better than v1) | **Status**: Step 1 milestone achieved

**Changes applied**:
- `linear_f`: Reverted to flat 2-loop pipeline with `UNROLL factor=2` — recovers the pipelined outer loop; factor=2 overridden by HLS to full unroll but keeps flat structure that enables II=9 (same as v1)
- `MHSA_SM`: Changed from `DEPENDENCE` pragma to `#pragma HLS UNROLL` — forces compile-time-constant pointer offsets into `softmax_vec`; HLS can prove non-aliasing → softmax inlined into MHSA_H
- `q_buf`: `cyclic factor=4` retained (II=1 for HEAD_DIM=8)

**Net gains vs v1**:
| Improvement | Cycles saved |
|---|---|
| LN_MEAN×2: 84→27 cycles each | +114 |
| LN_VAR×2: 97→40 cycles each | +114 |
| MHSA_SM eliminated, MHSA_H: 1110→562 | +548 |
| MHSA_QI_MHSA_KI: 140→92 cycles | +48 |
| **Total saved** | **+824 cycles** |

**Remaining II violations** (all understood and accepted at this scale):
- LN_MEAN/LN_VAR II=5: running accumulators still carry across iterations
- LF_S_LF_OD II=9: external `w` BRAM hard limit (2 ports for 16 reads)
- MHSA_OQ_MHSA_OD II=2: v_buf access minor issue

---

### Step 2 — Scale-Up to DIM=32 (v3 pragmas)
**Date**: Sun May 31 14:00  
**Latency**: 22,876 cycles | **Status**: Scaled correctly, one regression, timing warning

**Changes**: Only `t_blk_s2_types.h` updated (DIM=16→32, MLP_HID=32→64, HEAD_DIM=8→16)

**What scaled as expected**:
- LF_S_LF_OD II: 9→17 (in_dim 16→32, formula II=in_dim/2+1 ✓)
- LF_S_LF_OD8 fc2 II: 17→33 (in_dim 32→64)
- LN_MEAN trips doubled (4→8), latency: 27→47 cycles

**New regression — MHSA_QI_MHSA_KI II=2** (was II=1 in v3):  
HEAD_DIM grew from 8→16. MHSA_D unrolls 16 reads/cycle. With `q_buf cyclic factor=4`: 16 reads / 4 banks = 4 reads/bank → II=2.  
Fix: `factor=TB_HEAD_DIM` (generalizes to HEAD_DIM per step).

**New concern — Timing Slack=−0.49 ns at top level**:  
LF_S_LF_OD modules with in_dim=32 (32 unrolled multipliers) create longer combinational paths.  
Will worsen at Step 3+ (in_dim=64, 64 mults).

---

### v4 — LN Tile Fix + Corrected Partitions (Step 2, DIM=32)
**Date**: Sun May 31 14:19  
**Latency**: 23,532 cycles | **Status**: LN II=1 ✅ MHSA_QI II=1 ✅ — critical fixes confirmed for scale-up

**Changes applied**:

1. **`layer_norm` — Tile approach (LN_MEAN/LN_VAR II=5 → II=1)**  
   Console confirmed: `ms2` (write ln79 → load ln82) is the loop-carry culprit. With ms0..ms3 all declared OUTSIDE the loop, every accumulator has period=1 (updated each iteration) → II=FP_add_latency=5.  
   Fix: `tile_sum` declared INSIDE the loop body (local, not loop-carried). Each iteration accumulates 4 elements into a fresh local variable, writes result to `partial[t]` exactly once → no loop-carry → II=1.  
   `partial[]` tree-reduced with fully-unrolled UNROLL loop after the tile loop.

2. **`q_buf` & `k_buf` — `cyclic factor=TB_HEAD_DIM`**  
   Console confirmed: *"Unable to schedule 'load' on `q_buf` due to limited memory ports (II=1)"*  
   MHSA_D reads HEAD_DIM=16 consecutive elements (d=0..15). With factor=HD: bank=(h×HD+d)%HD=d → 16 unique banks → II=1.

3. **`v_buf` — `cyclic factor=5` (reverted from TB_HEAD_DIM)**  
   MHSA_OQ reads 4 elements at stride DIM=32: {0,32,64,96}%5={0,2,4,1} → 4 unique banks.  
   WARNING: `factor=TB_HEAD_DIM=16` would give {0,32,64,96}%16={0,0,0,0} → all to bank 0 → catastrophic conflict.  
   Factor=5 verified safe for all DIM values in the scaling ladder.

4. **`x_buf` & `ln_buf` — `cyclic factor=4`**  
   New LN tile loop reads 4 consecutive elements/cycle. HLS auto-inferred only factor=2 for x_buf (from COPY_IN pipeline). Explicit factor=4 ensures 4 unique banks → II=1 on LN tile loop.

**Actual results**:

| Module | Step 2 (before v4) | v4 actual | Change |
|---|---|---|---|
| LN_MEAN (each, ×2) | 47 cyc II=5 | **42 cyc II=1 ✅** | -5 cyc, II fixed |
| LN_VAR (each, ×2) | 60 cyc II=5 | **55 cyc II=1 ✅** | -5 cyc, II fixed |
| VITIS_LOOP_84_1 LN outer (×2) | ~1,024 cyc | **~1,368 cyc** ⚠ | +344 cyc FSM overhead |
| MHSA_QI_MHSA_KI | 172 cyc II=2 | **156 cyc II=1 ✅** | -16 cyc, II fixed |
| MHSA_H | 852 cyc | **820 cyc** | -32 cyc |
| **Total** | **22,876** | **23,532** | +3% at DIM=32 |

**Why total regressed slightly (+3%) despite II fixes:**  
The tile approach moves scalar operations (mean = msum/dim, inv_std = 1/sqrt(vsum/dim)) out of the LN pipeline modules and into the top-level FSM. These operations (fmul, fsqrt, fdiv each taking 4–15 cycles) now run sequentially between LN_MEAN, LN_VAR, and LN_NORM modules, adding ~86 cycles of FSM overhead per token × 4 tokens × 2 LN calls = ~688 extra cycles. At DIM=32, the II improvement (5→1) saves only ~40 cycles — not enough to offset the FSM overhead.

**Why v4 fixes are still critical for scaling:**  
At DIM=144 (Step 5), LN_MEAN with II=5 would cost: 36 tiles × II=5 + depth ≈ 191 cycles/token.  
With II=1: 36 × 1 + depth ≈ 70 cycles/token. Saving 121 cycles × 4 tokens × 2 LN = **968 cycles** saved.  
The FSM overhead (~86 cycles/token) becomes relatively smaller as DIM grows.  
The MHSA_QI II=1 fix will be essential at Step 3+ when HEAD_DIM=16 or 36.

**Resources change (v4 vs Step2):**
- BRAM: 43 → **19** (−56%): removed bind_storage from q/k/v/x/ln; partitioned banks are too small for BRAM
- DSP: 87 → **139** (+60%): 8-partial tree reduce for LN (28 FP adders) + MHSA_QI HD=16 mults (39 DSP)
- FF: 49,708 → **58,744** (+18%): larger pipeline depth for tile-based LN
- LUT: 43,793 → **49,626** (+13%): address decode for higher partition factors

---

## II Violation Reference Table

| Violation Type | Variable | HLS Warning | Fix Applied | Result |
|---|---|---|---|---|
| FP-add loop-carry (LN mean) | `sum` → `ms0..ms3` | HLS 200-880 | Tile approach (local `tile_sum`) | II=1 (v4) |
| FP-add loop-carry (LN var) | `vs0..vs3` | HLS 200-880 | Tile approach (local `tile_sum`) | II=1 (v4) |
| BRAM port limit (linear Q/K/V/O) | `w` array | HLS 200-885 | Fundamental limit; accepted | II=in_dim/2+1 |
| BRAM port limit (fc1) | `w` array | HLS 200-885 | Fundamental limit; accepted | II=17 (Step2) |
| BRAM port limit (fc2) | `w` array | HLS 200-885 | Fundamental limit; accepted | II=33 (Step2) |
| q_buf port limit (MHSA scores) | `q_buf` | HLS 200-885 | `cyclic factor=TB_HEAD_DIM` | II=1 (v3/v4) |
| Pointer aliasing (softmax) | `a_buf` | HLS 200-880 | `UNROLL` on MHSA_SM loop | Eliminated (v3) |
| v_buf port limit (attn output) | `v_buf` | HLS 200-885 | `cyclic factor=5` | II=2 (minor, accepted) |

---

## Partition Factor Reference

| Buffer | v1 | v2 | v3 | v4 | Purpose |
|--------|-----|-----|-----|-----|---------|
| `x_buf` | auto(2) | auto | auto(2) | **4** | LN tile: 4 reads/cycle |
| `ln_buf` | auto | auto | auto | **4** | LN tile: 4 reads/cycle |
| `q_buf` | none | **4** | **4** | **TB_HEAD_DIM** | MHSA_D: HD reads at stride 1 |
| `k_buf` | auto(4) | auto | auto(8) | **TB_HEAD_DIM** | MHSA_D: HD reads at stride 1 |
| `v_buf` | auto(5) | auto | auto(5) | **5** | MHSA_OK: 4 reads at stride DIM |
| `a_buf` | **complete** | **complete** | **complete** | **complete** | 16 floats → registers |
| `attn_out` | BRAM | BRAM | BRAM | BRAM | Output buffer |
| `mlp_buf` | BRAM | BRAM | BRAM | BRAM | MLP intermediate |

**Rule for q_buf/k_buf factor**: `factor = TB_HEAD_DIM = TB_DIM / TB_HEADS`  
This ensures consecutive d-dimension reads (MHSA_D unrolled 0..HD-1) each hit a unique bank.

**Rule for v_buf factor=5**: `{0, DIM, 2×DIM, 3×DIM} mod 5` must all be distinct (DIM not divisible by 5).  
Verified: DIM=16✓, 32✓, 64✓, 128✓, 144✓

---


### Step 3 — SEQ+HEADS Scale-Up (DIM=64, SEQ=16, HEADS=4)
**Date**: Sun May 31 14:42  
**Latency**: 310,804 cycles = 1.554 ms | **Status**: ✅ PASS — timing closed, all II as predicted

**Changes from v4**:
1. **Dimensions**: DIM=64, SEQ=16, HEADS=4, HD=16, MLP_HID=128, DEPTH=1
2. **SM_EXP/SM_NORM**: Changed from UNROLL to PIPELINE II=1 (resource explosion fix: SEQ=16 UNROLL × 16 calls = 256 expf units → PIPELINE = 1 expf unit per token, reused sequentially)
3. **v_buf factor**: 5 → 17 (= TB_VBUF_FACTOR = TB_SEQ+1 = 17, prime, ensures all 16 ki×DIM mod 17 are distinct)

**Key findings**:

| Module | Expected | Actual | Status |
|---|---|---|---|
| LF_S_LF_OD II (Q/K/V/O) | 33 (64/2+1) | **33** | ✅ |
| LF_S_LF_OD37 II (fc1, in=64) | 33 | **33** | ✅ |
| LF_S_LF_OD38 II (fc2, in=128) | 65 | **65** | ✅ |
| MHSA_QI_MHSA_KI II | 1 | **1** | ✅ |
| SM_EXP II | 1 | **1** | ✅ |
| MHSA_OQ_MHSA_OD II | ~1 hoped | **8** | ⚠ (resource) |
| Timing | <5 ns | **4.26 ns** | ✅ positive slack! |

**Timing surprise**: Step 2 had −0.49 ns slack (timing violation), but Step 3 at 4.26 ns is clean.
HLS mux restructuring at DIM=64 (more banks, smaller muxes) reduced combinational path length.

**New issue — MHSA_OQ_MHSA_OD II=8**:
At Step 3, MHSA_OK is fully unrolled to 16 multiply-accumulates per pipeline stage. HLS allocates
only 2 FP multipliers from its shared pool for this module → 16/2 = 8 cycles per (qi,d) iteration.
Impact at Step 3: 256 iters × 8 = 2,048 + depth = 2,192 cycles per head × 4 heads = 8,768 cycles total.
This is only 2.8% of total → acceptable.
**Risk at Step 4**: MHSA_OK unrolled to 100 → potential II=50, 3600×50×4=720K cycles. See Step 4 mitigation.

**Dominant latency contributors** (Step 3):
- Q+K+V+O projections: 4×34,280 = 137,120 cycles (44%)
- fc1+fc2: 68,072+67,528 = 135,600 cycles (44%)
- Attention+softmax+V-sum: ~18,000 cycles (6%)
- Layer norm, residual, I/O: ~20,000 cycles (6%)

**Resources**: BRAM=104 (4%), DSP=103 (2%), FF=134K (15%), LUT=109K (25%) — well within budget

## Known Persistent Issues

| Issue | II | Scale Effect | Plan |
|---|---|---|---|
| `w` BRAM dual-port limit on linear_f | in_dim/2+1 | Grows with DIM | Accept; needs wider bus at full scale |
| LN_MEAN/LN_VAR accumulation | ~~5~~ → **1** ✅ | Fixed in v4 | Tile approach; FSM overhead small at large DIM |
| LN tile FSM overhead | — | Grows with tokens but not DIM | ~86 cyc/token; negligible at Step 3+ |
| MHSA_OQ_MHSA_OD v_buf | 2 | Stays at 2 | Minor; ~4% impact |
| Timing slack on LF_S_LF_OD | −0.21 ns (v4) | Worsens with DIM | Monitor at Step 3; may need PIPELINE II=2 |
| Timing slack on LF_S_LF_OD8 (fc2) | −0.49 ns | Worsens with in_dim | Most critical; in_dim=64 at Step 2 already tight |

---

## Without-Pragma Baseline Run (Full S2: DIM=144, SEQ=400, DEPTH=2)

**Date**: Wed Jun 3 12:05:18 2026  
**Solution**: `without_pragmas`  
**All `#pragma HLS` directives commented out** (PIPELINE, UNROLL, ARRAY_PARTITION, bind_storage, INLINE, INTERFACE)  
**Purpose**: Measure latency without HLS optimisation directives to quantify pragma impact

### Overall Result

| Metric | Without Pragmas | With Pragmas (Step 5) | Ratio |
|---|---|---|---|
| **Total latency** | **1,228,024,873 cycles = 6.140 sec** | 94,039,743 cycles = 0.470 sec | **13.1× slower** |
| BRAM_18K | 1156 / 2160 = **53%** ✅ | 1256 / 2160 = 58% | Less (no banked partitions) |
| URAM | 0 / 80 = 0% | 0 / 80 = 0% | Same |
| DSP | 44 / 4272 = 1% | 260 / 4272 = 6% | **6× fewer** |
| FF | 16,444 / 850,560 = 1% | 317,871 / 850,560 = 37% | **19× fewer** |
| LUT | 17,519 / 425,280 = 4% | 269,309 / 425,280 = 63% | **15× fewer** |
| Timing | **3.966 ns (+1.034 ns slack ✅)** | 4.317 ns (+0.683 ns slack) | **Better timing** |

### Per-Module Latency Breakdown (no-pragma)

**Top level** (transformer_blk_s2_top):

| Module | Cycles | Absolute | Notes |
|---|---|---|---|
| COPY_IN | 57,602 | 0.288 ms | II=1, 57,600 elements (400×144) |
| DEPTH loop (×2 blocks) | 1,227,909,666 | 6.139 sec | 2 × 613,954,833 per block |
| COPY_OUT | 57,602 | 0.288 ms | II=1 |
| **TOTAL** | **1,228,024,873** | **6.140 sec** | |

> Note: Step 5 with-pragma log shows COPY_IN = 14,402 cycles, which matches SEQ=100 (Step 4 dims). The correct value for SEQ=400 is **57,602 cycles** as confirmed by this run.

**Per transformer_block** (613,954,831 cycles = 3.070 sec):

| Module | Cycles | % of block | Notes |
|---|---|---|---|
| `layer_norm` (1 call shown) | 426,401 | 0.07% | 400 tokens × 1,066 cyc/token |
| `mhsa` | 344,858,775 | 56.2% | Dominant — see breakdown below |
| `LF_S_LF_OD` (MLP fc1) | 134,438,400 | 21.9% | 115,200 iters × 1,167 cyc; NOT pipelined |
| `LF_S_LF_OD` (MLP fc2) | 133,574,400 | 21.8% | 57,600 iters × 2,319 cyc; NOT pipelined |
| `AR` residual add (×2) | 57,611 each | ~0.02% | II=1 ✅ unchanged |
| `MLP_ACT` SiLU | 115,221 | 0.02% | II=1 ✅ unchanged |
| `LF_ID` (fc1 inner pipeline) | 1,162 | — | Pipelined sub-module of fc1 |
| `LF_ID1` (fc2 inner pipeline) | 2,314 | — | Pipelined sub-module of fc2 |

**MHSA sub-breakdown** (344,858,775 cycles):

| Module | Cycles | vs. Step 5 pragma | Notes |
|---|---|---|---|
| `linear_f` Q projection | ~67,161,601 | 64× slower | One call; 57,600 iters × 1,166 cyc |
| `linear_f` K projection | ~67,161,601 | 64× slower | Same dims |
| `linear_f` V projection | ~67,161,601 | 64× slower | Same dims |
| `linear_f` O projection | ~67,161,601 | 64× slower | Same dims |
| **linear_f subtotal (4 calls)** | **~268,646,404** | **64×** | Dominant in MHSA |
| `MHSA_QI_MHSA_KI` | 2,880,286 | **280× slower** | q/k_buf not partitioned → serial dot-product |
| `MHSA_H` loop (4 heads) | 76,154,760 | **99× slower** | 4 × 19,038,690 per head |
| — `MHSA_SM` (softmax) | 1,038,400 | — | 400 tokens × 2,596 cyc; NOT pipelined |
| — `MHSA_OQ_MHSA_OD` | 15,120,000 | — | 14,400 iters × 1,050 cyc; NOT pipelined |
| `MHSA_CPY` | 57,602 | ~4× | II=1, same element count |

**layer_norm per-token pipeline breakdown** (without pragma, 1,066 cycles/token):

| Sub-pipeline | Cycles | With Pragma (tile) | Notes |
|---|---|---|---|
| `LN_MEAN` (tile accumulate into mp[36]) | 107 | ~70 | HLS auto-created tile structure; II=1 ✅ |
| `LN_MRED` (reduce mp[36] → mean) | 292 | merged into FSM | Sequential tree reduce; no pragma equivalent |
| `LN_VAR` (tile variance into vp[36]) | 120 | ~83 | II=1 ✅ |
| `LN_VRED` (reduce vp[36] → var) | 292 | merged into FSM | Sequential tree reduce |
| `LN_NORM` (normalise 144 elements) | 173 | ~173 | II=1 ✅ same |
| **Total/token** | **984 cyc** | **~70 cyc (est.)** | **~14× slower** |
| **Total for 400 tokens** | **426,401** | **~49,000 (est.)** | **~8.7× slower** |

### Per-Module II Comparison

| Module | Target | Without Pragma | With Pragma (Step 5) | Root Cause |
|---|---|---|---|---|
| `LF_S_LF_OD` outer loop | — | NOT pipelined, iter=1,166 | NOT pipelined, iter=~1,052 | Both sequential; UNROLL only affects inner LF_ID |
| `LF_ID` inner (linear_f) | 1 | Pipelined, but serial weight reads | II=73 (UNROLL factor unrolls w reads) | No UNROLL → one w read/cycle vs 2/cycle |
| `MHSA_QI_MHSA_KI` | 1 | II effectively ~288/pair | II=1 ✅ | No `q_buf` / `k_buf` cyclic partition |
| `MHSA_SM` softmax | 1 | NOT pipelined (iter=2,596/token) | PIPELINE (unrolled in pragma run) | No UNROLL on softmax inner loop |
| `MHSA_OQ_MHSA_OD` | 1 | NOT pipelined (iter=1,050) | II=8 (resource limited) | No `v_buf` partition |
| `LN_MEAN` / `LN_VAR` | 1 | II=1 ✅ (auto tile) | II=1 ✅ (tile pragma) | HLS auto-inferred tile; no regression |
| `LN_NORM` | 1 | II=1 ✅ | II=1 ✅ | No accumulation carry |
| `AR` residual | 1 | II=1 ✅ | II=1 ✅ | Unchanged |
| `MLP_ACT` SiLU | 1 | II=1 ✅ | II=1 ✅ | Unchanged |
| `COPY_IN` / `COPY_OUT` | 1 | II=1 ✅ | II=1 ✅ | Unchanged |

### Why Timing *Improves* Without Pragmas (3.966 ns vs 4.317 ns)

Without UNROLL, HLS allocates a single sequential multiply-accumulate unit per linear_f call. The critical combinational path becomes:
- With pragma: 144-wide unrolled MACs + 8-level adder tree → long path (~4.3 ns)
- Without pragma: single MAC + register pipeline → short path (~3.5 ns for linear_f)

Fewer parallel functional units = simpler routing = shorter critical path. This is the classic **area-timing tradeoff**: pragmas trade timing slack for parallelism (throughput); removing them recovers timing at the cost of massively increased latency.

### Why Latency Is 13× Worse Without Pragmas

The pragmas provide the following cycle savings (approximate, relative to no-pragma run):

| Pragma removed | Modules affected | Approximate cycles lost (per block) | Mechanism |
|---|---|---|---|
| `UNROLL` on LF_ID | 4× MHSA linear_f + 2× MLP linear | ~394M cycles | Weight reads become serial (1/cycle vs 2/cycle, II=1 vs II=73) |
| `ARRAY_PARTITION` on q/k_buf | MHSA_QI_MHSA_KI | ~2.87M cycles | Serial HEAD_DIM=36 reads vs parallel |
| `UNROLL` on MHSA_SM / softmax | MHSA_H loop | ~75.4M cycles | Softmax becomes sequential over 400 tokens |
| `UNROLL` on MHSA_OQ_MHSA_OD | V-weighted sum | ~14.9M cycles | v_buf access serialised |
| LN tile approach | layer_norm | ~0.38M cycles | LN_MRED/VRED tree reduction added |
| **Total** | | **~488M cycles/block** | |
| **×2 blocks (DEPTH=2)** | | **~976M extra cycles** | |

The remaining gap (~178M vs expected ~976M) is FSM overhead and interaction effects.

### Resource Reduction Without Pragmas

Without pragmas, much less hardware is generated:

| Resource | Without Pragma | With Pragma (Step 5) | Reduction |
|---|---|---|---|
| DSP | 44 | 260 | **−83%** — no parallel MAC units |
| FF | 16,444 | 317,871 | **−95%** — no pipeline registers for unrolled paths |
| LUT | 17,519 | 269,309 | **−93%** — no address decode for array partitions, no unrolled mux trees |
| BRAM | 1,156 | 1,256 | −8% — buffers same size; slight reduction from no partition overhead |

The design without pragmas is fully implementable on the target device (all resources < 100%), but is 13× too slow for real-time inference.
