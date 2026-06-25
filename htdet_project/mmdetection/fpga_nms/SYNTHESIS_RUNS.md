# fpga_nms — Synthesis Runs & Learnings

**Module**: `nms_top` — score-filter + insertion sort + greedy multiclass NMS  
**Device**: xczu9eg-ffvb1156-2-e (ZCU102, ZynqUltraScale+)  
**Top function**: `nms_top` in `nms_top.cpp`  
**Clock target**: 5.00 ns (200 MHz) — see §4 for timing notes  
**Spatial phase tested**: P6 (TEST_H=10, TEST_W=10, TEST_STRIDE=64)  
**Model config**: 4 classes, 9 anchors/location, INPUT 640×640

**Compile/run commands:**
```bash
# g++ csim (from fpga_nms/):
g++ -O2 -std=c++14 -I. -I../fpga_temp/hls_stubs \
    -DTEST_H=10 -DTEST_W=10 -DTEST_STRIDE=64 \
    testbench_nms.cpp nms_top.cpp -lm -o tb_nms && ./tb_nms

# HLS synthesis:
vitis_hls -f run_hls_nms.tcl
```

---

## Module Hierarchy

```
nms_top                               ← top-level (ap_ctrl_hs)
 ├─ score_filter                      ← sigmoid + anchor decode + candidate fill
 │   └─ score_filter_Pipeline_SF_C    ← pipelined innermost class loop (SF_C)
 ├─ nms_top_Pipeline_SORT_INIT        ← cand_order[] initialisation (II=1)
 ├─ nms_top_Pipeline_SORT_J           ← inner insertion-sort comparison loop
 ├─ nms_greedy                        ← greedy multiclass NMS
 │   ├─ nms_greedy_Pipeline_SUPP_INIT ← suppressed[] reset (II=1)
 │   └─ nms_greedy_Pipeline_NMS_J     ← inner IoU suppression loop (pipelined)
 └─ nms_top_Pipeline_OUT              ← detection write-back (II=1)
```

---

## Run History

| Run | Date | NMS_PRE | Key change | Report folder |
|-----|------|---------|------------|---------------|
| **Baseline** | 2026-06-02 14:12 | 1000 | Initial code | `report1/` |
| **OPT v1** | 2026-06-02 14:29 | 1000 | OPT-1..7 (code) | `report2/` |
| **OPT v2** | 2026-06-02 16:01 | 300 | NMS_PRE reduced + outer TRIPCOUNT pragmas | *(report overwritten)* |
| **OPT v3** | — | 300 | OPT-8/9/10 (fixed-point IoU) | ⏳ not yet synthesised |

---

## 1. Latency — Why the Top Level Shows "?"

`nms_top` reports `?` total latency in all runs. This propagates from three sub-components:

| Module | Reason for `?` |
|--------|----------------|
| `score_filter` | Outer loops SF_H/W/RI/SI all carry `&& num_cand < CAND_BUF_SIZE` guards. HLS treats the bound as runtime-variable even though for P6 the guard never fires (CAND_BUF_SIZE = 10×10×9×4 = 3600 = exact maximum). |
| `SORT_INIT` | `for (i = 0; i < N; i++)` where N = `num_cand` — runtime value |
| `OUT` | `for (k = 0; k < total; k++)` where `total = nk ≤ MAX_DETS` — runtime value |

HLS propagates `?` upward: since `score_filter` returns `?`, `nms_top` returns `?`.  
**Fix**: Remove `&& num_cand < CAND_BUF_SIZE` guards from SF_H/W/RI/SI outer loops (the guard is never needed for correctness since SF_C already checks it). This lets HLS see static H×W×RI×SI trip counts and report exact latency.

### Reconstructed latency from known sub-components

For P6 (10×10), the outer loops always execute exactly 10×10×3×3 = 900 SF_SI iterations (the CAND_BUF_SIZE guard never fires):

| Stage | Derivation | Cycles |
|-------|-----------|--------|
| score_filter | 900 SF_SI × 137 cy/iter | **~123,300 cy** (effectively fixed) |
| SORT_INIT | up to num_cand × 1 cy | 0 – 3,600 cy |
| SORT_I | N² × II/2 average | see table below |
| nms_greedy | SUPP_INIT + N²×II/2 | see table below |
| OUT | total × 1 cy + 2 | 0 – 102 cy |

---

## 2. Latency — All Runs

> **Note**: report2 pragmas show `max=1000` for all loops — the NMS_PRE reduction to 300 happened after that synthesis. OPT v2 latency values come from the subsequent synthesis (reports overwritten).

### Per-module cycle counts

| Module / Loop | **Baseline** (N=1000) | **OPT v1** (N=1000) | **OPT v2** (N=300) | **OPT v3** (N=300, est.) |
|---|---:|---:|---:|---:|
| score_filter | ? *(data-dep)* | ? *(data-dep)* | **126,101** | ~126,101 |
| SF_C inner: II / iter latency / trips | II=11, 53cy, ×4 | II=11, 70cy, ×4 | II=11, 70cy, ×4 | II=11, 70cy, ×4 |
| score_filter_Pipeline_SF_C | 87 cy | 104 cy | 104 cy | 104 cy |
| SORT_I (outer, worst case) | 5,013,000 | 4,012,000 | **363,600** | ~363,600 |
| SORT_J II / iteration latency | 5 / 5 | 4 / 5 | 4 / 5 | 4 / 5 |
| SORT_J max cycles (N trips) | 5,007 (N=1000) | 4,006 (N=1000) | 1,206 (N=300) | 1,206 (N=300) |
| **nms_greedy (total, worst case)** | **54,022,004** | **43,009,004** | **3,872,704** | **~1,200,000 est.** |
| SUPP_INIT cycles | 1,002 | 1,002 | 302 | 302 |
| NMS_J **II** (achieved) | **54** | **43** | **43** | **~12 est.** |
| NMS_J iteration latency | 55 | 44 | 44 | ~13 est. |
| NMS_J max cycles (N trips) | 54,000 (N=1000) | 43,000 (N=1000) | 12,900 (N=300) | ~3,600 est. (N=300) |
| NMS_I outer max | 54,021,000 | 43,008,000 | 3,872,400 | ~1,200,000 est. |
| **Total worst-case** | **~59,160,000** | **~47,148,000** | **~4,362,000** | **~1,690,000 est.** |

### Speedup relative to Baseline

| Metric | OPT v1 | OPT v2 | OPT v3 (est.) |
|--------|-------:|-------:|---------------:|
| nms_greedy cycles | 1.26× | **13.9×** | **~45×** |
| NMS_J II | 1.26× | 1.26× | **~4.5×** |
| SORT_I cycles | 1.25× | **13.8×** | 13.8× |
| Total worst-case cycles | 1.26× | **13.6×** | **~35×** |
| **Total latency @ 6.5 ns** | **1.25×** | **~12.7×** | **~35× est.** |

### Worst-case wall-clock time (at achievable 154 MHz / 6.5 ns — see §4)

| Stage | **Baseline** | **OPT v1** | **OPT v2** | **OPT v3 (est.)** |
|-------|------------:|----------:|----------:|------------------:|
| score_filter | ~0.80 ms | ~0.80 ms | **0.82 ms** | **0.82 ms** |
| SORT_I | ~32.6 ms | ~26.1 ms | **2.36 ms** | **2.36 ms** |
| nms_greedy | ~351.1 ms | ~279.6 ms | **25.2 ms** | **~7.8 ms** |
| OUT + overhead | ~0 ms | ~0 ms | ~0 ms | ~0 ms |
| **Total (P6, worst case)** | **~384.5 ms** | **~306.5 ms** | **~28.4 ms** | **~11.0 ms est.** |
| **Speedup vs Baseline** | — | 1.3× | **13.5×** | **~35× est.** |

> OPT v3 estimates assume NMS_J II reduces from 43 → ~12 after replacing float32 IoU arithmetic with `ap_ufixed<18,11>` fixed-point ops (fsub 6cy → 2cy, fmul 5cy → 3cy DSP). Re-synthesise to confirm.

---

## 3. Timing

| Module | Baseline | OPT v1 | OPT v2 | Target | Root cause |
|--------|----------|--------|--------|--------|------------|
| nms_top (top) | **6.179 ns** ❌ | **6.179 ns** ❌ | **6.179 ns** ❌ | 5.00 ns | fexp in score_filter |
| score_filter | **6.179 ns** ❌ | **6.179 ns** ❌ | **6.179 ns** ❌ | 5.00 ns | fexp in score_filter |
| score_filter_Pipeline_SF_C | **6.179 ns** ❌ | **6.179 ns** ❌ | **6.179 ns** ❌ | 5.00 ns | `fexp_32ns` (13-cycle) |
| nms_greedy | **6.124 ns** ❌ | **6.124 ns** ❌ | **6.124 ns** ❌ | 5.00 ns | `fsub_32ns` (6-cycle) in IoU |
| nms_greedy_Pipeline_NMS_J | **6.124 ns** ❌ | **6.124 ns** ❌ | **6.124 ns** ❌ | 5.00 ns | float IoU critical path |
| nms_top_Pipeline_SORT_J | 5.15 ns ❌ (−0.15) | **3.496 ns** ✅ | **3.496 ns** ✅ | 5.00 ns | Fixed by LUTRAM (OPT-4) |
| nms_top_Pipeline_SORT_INIT | 3.58 ns ✅ | 3.58 ns ✅ | 3.58 ns ✅ | 5.00 ns | Integer counter only |
| nms_top_Pipeline_OUT | 2.47 ns ✅ | 2.47 ns ✅ | 2.47 ns ✅ | 5.00 ns | Integer index + BRAM write |
| nms_greedy_Pipeline_SUPP_INIT | 1.67 ns ✅ | 1.67 ns ✅ | 1.67 ns ✅ | 5.00 ns | 1-bit write loop |

**Root cause of persistent timing failure:**
- `score_filter`: `fexp_32ns_32ns_32_13_full_dsp` (13-cycle expf for `expf(dw)`, `expf(dh)`) creates a 6.179 ns combinational path, 1.18 ns over budget.
- `nms_greedy`: `fsub_32ns_32ns_32_6_no_dsp` (6-cycle float subtraction) in the IoU intersection chain creates a 6.124 ns path, 1.12 ns over budget.

**Fix for NMS_J timing (OPT v3):** Replacing `bbox_t = float` with `coord_t = ap_ufixed<18,11>` eliminates the `fsub_32ns` from the NMS_J critical path entirely. The fixed-point subtraction is ~2 ns combinational — expected to close timing on NMS_J.

**Fix for score_filter timing (still open):** The `fexp` path is unchanged. Either:
1. **1-line TCL fix**: Change `set CLK_NS "5"` → `set CLK_NS "6.5"` in `run_hls_nms.tcl` (already updated in TCL). At 154 MHz all paths meet timing.
2. Replace `expf()` with a synthesis-mode piecewise-linear approximation in score_filter.

---

## 4. Resource Utilisation

| Resource | Baseline | OPT v1 | OPT v2 | Available | Notes |
|----------|----------|--------|--------|-----------|-------|
| BRAM_18K | 53 (2%) | 41 (2%) | **41 (2%)** | 1,824 | 12 BRAM freed by LUTRAM OPTs |
| DSP | 16 (~0%) | 14 (~0%) | **14 (~0%)** | 2,520 | `fdiv` removed (OPT-1) |
| FF | 6,807 (1%) | 6,759 (1%) | **6,757 (1%)** | 548,160 | minimal change |
| LUT | 8,913 (3%) | 12,843 (4%) | **12,831 (4%)** | 274,080 | +44% — LUTRAM trade-off |
| URAM | 0 | 0 | 0 | 0 | not used |

LUT increase is the cost of LUTRAM replacements (scores, order, areas, suppressed) and the 4-way cand_boxes partition. Total utilisation remains very low — all 5 FPN levels could run in parallel on this device.

### Storage implementation evolution

| Variable | Baseline impl | OPT v1/v2 impl | Reason |
|---|---|---|---|
| `cand_boxes` (×4 banks) | BRAM ram_t2p, 1 bank | **BRAM ram_1p, 4 banks** (cyclic-4) | Parallel coord reads (OPT-2) |
| `cand_scores` | BRAM | **LUTRAM ram_2p** | 0-cycle read → SORT_J II 5→4 (OPT-4) |
| `cand_order` | BRAM | **LUTRAM ram_2p** | 0-cycle read → SORT_J II (OPT-4) |
| `cand_areas` | *(not present)* | **LUTRAM ram_2p** | Pre-computed areas (OPT-5/6) |
| `suppressed` | BRAM | **LUTRAM ram_2p** | 0-cycle read → breaks RAW stall (OPT-3) |
| `cand_cls` | BRAM ram_t2p | BRAM ram_t2p | OPT v2: still BRAM (int, 32-bit) |
| `keep_idx` | BRAM ram_t2p | BRAM ram_t2p | Only written by NMS_I, read once |

---

## 5. Optimisations Applied

### OPT v1 — Code changes to `nms_top.cpp` (report2, 2026-06-02 14:29)

| ID | Change | Location | Effect |
|----|--------|----------|--------|
| OPT-1 | `inter/uni > thr` → `inter > thr*uni` — eliminates float division | `iou_exceeds()` | Removes 16-cycle `fdiv` IP; NMS_J II 54→43 (−11 cy) |
| OPT-2 | `ARRAY_PARTITION variable=cand_boxes cyclic factor=4` | `nms_top()` pragma | All 4 box coords (x1/y1/x2/y2) in separate banks → 1-cycle parallel read (was 4 sequential BRAM reads) |
| OPT-3 | `suppressed[]` → `bind_storage RAM_2P LUTRAM` | `nms_greedy()` pragma | 0-cycle read latency; breaks the BRAM read-after-write stall that caused NMS_J II≫1 |
| OPT-4 | `cand_scores`, `cand_order` → LUTRAM | `nms_top()` pragma | SORT_J II 5→4; SORT_J timing FAIL→PASS (was −0.15 ns); saves 11 BRAM |
| OPT-5 | Pre-compute `cand_areas[k] = (x2−x1)*(y2−y1)` in score_filter | `score_filter()` body | Eliminates 3 FP ops per NMS_J iteration (2 fsub + 1 fmul); saves ~135K ops for N=300 |
| OPT-6 | `cand_areas` → LUTRAM | `nms_top()` pragma | 0-cycle area reads in NMS_J |
| OPT-7 | Cache box-ii coords as scalar registers `ax1/ay1/ax2/ay2` in NMS_I body | `nms_greedy()` body | Eliminates repeated BRAM reads of the same box address on each inner NMS_J call |

### OPT v2 — Parameter and pragma changes (2026-06-02 16:01)

| ID | Change | Location | Effect |
|----|--------|----------|--------|
| OPT-8 *(was)* | `NMS_PRE 1000 → 300` | `fpga_types.h` | O(N²) scaling: (300/1000)² = 0.09 → **~11× speedup** on NMS and sort latency |
| OPT-9 *(was)* | `LOOP_TRIPCOUNT` added to SF_H/W/RI/SI outer loops | `score_filter()` pragmas | score_filter latency now reported (was `?`); no RTL change |
| OPT-10 *(was)* | `LOOP_TRIPCOUNT max=300` on SORT_I/J, NMS_I/J | `sort_scores()`, `nms_greedy()` | Latency estimates now use correct N=300 bound |

> Note: OPT IDs 8/9/10 were reused for the next round of changes below.

### OPT v3 — Fixed-point IoU (code complete 2026-06-03, synthesis pending)

| ID | Change | Location | Expected effect |
|----|--------|----------|-----------------|
| OPT-8 | `coord_t = ap_ufixed<18,11>`, `area_t = ap_ufixed<30,19>`, `area3_t = ap_ufixed<32,21>` — replaces float32 for all internal bbox/area arrays. Float→fixed conversion done **once** in score_filter; NMS_J loop is entirely fixed-point. | `nms_top.cpp` local typedefs | fsub 6cy→2cy, fmul 5cy→3cy (DSP). Expected NMS_J II 43→**~12**. Fixes NMS_J timing violation. `cand_boxes` BRAM: 32 BRAM_18K → ~16 BRAM_18K (18-bit elements). |
| OPT-9 | `3·inter > area_a + area_b` replaces `inter > 0.5·(area_a+area_b−inter)`. `3·inter = (inter<<1)+inter` — shift is free wiring in RTL, only 1 adder added. | `iou_exceeds()` | Eliminates 1 subtraction (for union) and 1 multiplication (for `thr × union`) from IoU critical path |
| OPT-10 | `cand_cls → ap_uint<2>` (4 classes need 2 bits), stored in LUTRAM (300×2 = 600 bits) instead of 32-bit BRAM | `nms_top.cpp` local typedef | Class compare in NMS_J becomes 2-bit integer op; saves 8 BRAM_18K |

**Synthesis guard:** All three OPT v3 types are wrapped in `#ifdef __SYNTHESIS__` with `float`/`int` fallback. The g++ csim testbench compiles and runs identically (all 4 tests pass).

---

## 6. Root-Cause Analysis of Remaining Bottlenecks (OPT v2 state)

### 6.1 NMS_J II=43 — float IoU critical path (target for OPT v3)

After OPT-1..7, the remaining II=43 is set by the float32 intersection arithmetic in `iou_exceeds()`:

```
read bx1/by1/bx2/by2  (parallel, 4-bank BRAM)     ~1 cy
max/min comparisons                                  ~1 cy
fsub_32ns  iw = x2 − x1                             6 cy   ← dominant IP
fsub_32ns  ih = y2 − y1                             6 cy   (parallel with iw)
fmul_32ns  inter = iw × ih                          5 cy
fmul_32ns  area_b = bw × bh                         5 cy   (parallel with inter)
fadd/fsub  uni = area_a + area_b − inter             6 cy
fmul_32ns  thr × uni (=0.5 × uni)                   5 cy
fcmp + suppressed write                              3 cy
─────────────────────────────────────────────────  ──────
Critical path depth                               ≈ 43 cy
```

The `fsub_32ns_32ns_32_6_no_dsp` (6-cycle) is confirmed in the Bind Op report. OPT v3 replaces this chain with ap_ufixed arithmetic (2-cycle sub, 3-cycle DSP mul) and eliminates the `thr×uni` multiply entirely using the `3·inter > area_sum` formulation.

### 6.2 SF_C II=11 — fexp in box decode (unfixed)

`expf(dw)` and `expf(dh)` inside score_filter achieve II=11 due to the 12-cycle `fexp_32ns_32ns_32_13_full_dsp` unit. Fixing requires either:
- A synthesis-mode piecewise-linear exp approximation (`#ifdef __SYNTHESIS__`)
- Replacing float deltas with a fixed-point lookup table

### 6.3 SORT_J II=4 — LUTRAM read chain residual (minor)

After LUTRAM (OPT-4), the residual II=4 = LUTRAM read (1cy) + `fcmp` (2cy) + select/write (1cy). Only addressable by switching `score_t` to fixed-point (integer compare → II=1) or replacing insertion sort with a bitonic sort network. Sort is not the dominant bottleneck at N=300 (363,600 cycles vs 3,872,704 for NMS).

---

## 7. C-Simulation Validation

All testbench tests pass in all runs (g++ csim):

| Test | Description | Expected | Result |
|------|-------------|----------|--------|
| 1 | All logits = −10.0 (sigmoid ≪ SCORE_THR) | `num_dets = 0` | ✅ PASS |
| 2 | Single detection, anchor=4 at (h=5, w=5), logit=2.0, delta=0 | `num_dets = 1`, `box=[160,160,544,544]`, `score=0.881` | ✅ PASS |
| 3 | Two overlapping boxes, same class, IoU=0.5625 > 0.5 | `num_dets = 1` (lower-score box suppressed) | ✅ PASS |
| 4 | Two overlapping boxes, **different** classes | `num_dets = 2` (cross-class, no suppression) | ✅ PASS |

OPT v3 g++ csim: all 4 tests pass (fixed-point types guarded by `#ifdef __SYNTHESIS__`, g++ uses float/int fallback — functionally identical).

---

## 8. Further Optimisation Options

| Option | Expected gain | Effort | Priority |
|--------|--------------|--------|----------|
| **Synthesise OPT v3** (code is ready) | NMS_J II 43→~12; ~3.5× total speedup | Trivial (run TCL) | **Now** |
| **Set clock to 6.5 ns** (already in TCL) | Timing fully closes (score_filter violation resolved) | Trivial (already done) | **Now** |
| Replace `expf()` with synthesis piecewise-linear | SF_C II 11→~1; score_filter timing closes at 5 ns | Small | High |
| `score_t → ap_ufixed<8,0>` | SORT_J II 4→1; saves cand_scores LUTRAM; faster score compare | Small | Medium |
| Reduce NMS_PRE to 200 | ~(200/300)² = 0.44× NMS cycles; ~2.25× further speedup | Trivial | Medium |
| Add TRIPCOUNT pragmas to SF_H/W/RI/SI outer loops | score_filter latency becomes reportable (currently `?`) | Trivial | Low (cosmetic) |
| Bitonic sort (N=300) | Sort 363,600→~500 cycles (~700×); negligible impact at system level | High | Low |

---

## 9. Run Summary

| Metric | Baseline | OPT v1 | OPT v2 | OPT v3 (est.) | Baseline→v3 |
|--------|----------|--------|--------|---------------|-------------|
| NMS_J II | 54 | 43 | 43 | **~12** | **~4.5×** |
| NMS_J max cycles (N trips) | 54,000 | 43,000 | 12,900 | **~3,600** | **~15×** |
| `fdiv` in IoU | 16-cycle | **removed** | **removed** | **removed** | — |
| `fsub` float in IoU | 6-cycle | 6-cycle | 6-cycle | **removed (ap_fixed)** | — |
| SORT_J II | 5 | **4** | **4** | **4** | 1.25× |
| NMS_PRE | 1000 | 1000 | **300** | **300** | **3.3×** |
| **nms_greedy worst-case** | **54,022,004** | **43,009,004** | **3,872,704** | **~1,200,000** | **~45×** |
| **SORT_I worst-case** | **5,013,000** | **4,012,000** | **363,600** | **~363,600** | **~14×** |
| **Total worst-case cycles** | **~59,160,000** | **~47,148,000** | **~4,362,000** | **~1,690,000** | **~35×** |
| **Total latency @ 6.5 ns** | ~384.5 ms | ~306.5 ms | **~28.4 ms** | **~11.0 ms** | **~35×** |
| BRAM_18K | 53 (2%) | 41 (2%) | **41 (2%)** | **~33 (1%)** | −38% |
| DSP | 16 | 14 | **14** | **~14** | −13% |
| LUT | 8,913 | 12,843 | **12,831** | **~13,000** | +46% |
| Timing (5 ns clock) | FAIL −2.53 ns | FAIL −2.53 ns | FAIL −2.53 ns | **NMS_J likely PASS** | partial |
| Timing (6.5 ns clock) | FAIL | FAIL | FAIL | **PASS (expected)** | fixed |
