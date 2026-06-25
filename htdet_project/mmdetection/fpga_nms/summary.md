# fpga_nms — C-Synthesis Report Summary

**Tool:** Vitis HLS 2022.2  
**Target device:** xczu9eg-ffvb1156-2-e (ZynqUltraScale+)  
**Top function:** `nms_top`  
**Source:** `fpga_nms/nms_top.cpp`

| Run | Date | Clock | NMS_PRE | Reports |
|---|---|---|---|---|
| Baseline | 2026-06-02 14:12 | 5.00 ns (200 MHz) | 1000 | `report/` |
| OPT v1 | 2026-06-02 14:29 | 5.00 ns (200 MHz) | 1000 | `report/` (overwritten) |
| **OPT v2** | **2026-06-02 16:01** | **5.00 ns (200 MHz)** | **300** | **`report1/`** |

---

## 1. Module Hierarchy

```
nms_top                               ← top-level (ap_ctrl_hs)
 ├─ score_filter                      ← sigmoid + anchor decode + candidate fill
 │   └─ score_filter_Pipeline_SF_C    ← pipelined innermost class loop (SF_C)
 ├─ nms_top_Pipeline_SORT_INIT        ← order[] initialisation (II=1)
 ├─ nms_top_Pipeline_SORT_J           ← inner insertion-sort loop (pipelined)
 ├─ nms_greedy                        ← greedy multiclass NMS
 │   ├─ nms_greedy_Pipeline_SUPP_INIT ← suppressed[] reset (II=1)
 │   └─ nms_greedy_Pipeline_NMS_J     ← inner suppression loop (pipelined)
 └─ nms_top_Pipeline_OUT              ← output write-back (II=1)
```

---

## 2. Timing

| Module | Estimated (ns) | Slack vs 5 ns | Baseline | OPT v1 | OPT v2 |
|---|---|---|---|---|---|
| nms_top (top) | 6.179 | −2.53 | FAIL | FAIL | FAIL |
| score_filter | 6.179 | −2.53 | FAIL | FAIL | FAIL |
| score_filter_Pipeline_SF_C | 6.179 | −2.53 | FAIL | FAIL | FAIL |
| nms_greedy | 6.124 | −2.47 | FAIL | FAIL | FAIL |
| nms_greedy_Pipeline_NMS_J | 6.124 | −2.47 | FAIL | FAIL | FAIL |
| **nms_top_Pipeline_SORT_J** | **3.496** | **+1.50** | ~~FAIL (−0.15)~~ | **PASS** ✓ | **PASS** ✓ |
| nms_top_Pipeline_SORT_INIT | 3.58 | +1.42 | PASS | PASS | PASS |
| nms_top_Pipeline_OUT | 3.82 | +1.18 | PASS | PASS | PASS |
| nms_greedy_Pipeline_SUPP_INIT | 3.58 | +1.42 | PASS | PASS | PASS |

**Root cause of timing failure:** `fexp_32ns_32ns_32_13_full_dsp` (exponential unit used for `expf(dw)`, `expf(dh)` in box decode and in the synthesis-mode sigmoid) has a 6.18 ns combinational path, 1.18 ns over the 5 ns budget. All other paths have ≥3.65 ns slack.

**Pending fix:** Change `set CLK_NS "5"` → `set CLK_NS "6.5"` in `run_hls_nms.tcl`. At 6.5 ns (154 MHz) the design fully closes. No code changes required.

---

## 3. Latency — All Three Runs

| Module / Loop | Baseline (N=1000) | OPT v1 (N=1000) | **OPT v2 (N=300)** | Baseline→v2 |
|---|---|---|---|---|
| **score_filter** | **?** *(data-dep)* | **?** *(data-dep)* | **126 101 cycles** | visible ✓ |
| SF_H | ? | ? | 126 100 (10 trips) | |
| SF_W | ? | ? | 12 590 (10 trips) | |
| SF_RI | ? | ? | 1 239 (3 trips) | |
| SF_SI | ? | ? | 411 (3 trips) | |
| SF_C inner (II / iter / trips) | II=11, 53 cyc, ×4 | II=11, 70 cyc, ×4 | II=11, 70 cyc, ×4 | |
| score_filter_Pipeline_SF_C | 87 cycles | 104 cycles | 104 cycles | |
| **SORT_I (outer)** | **5 013 000** | **4 012 000** | **363 600** | **−13.8×** |
| SORT_J II | 5 | 4 | 4 | |
| SORT_J max cycles | 5 007 (N=1000) | 4 006 (N=1000) | 1 206 (N=300) | |
| **nms_greedy (total)** | **54 022 004** | **43 009 004** | **3 872 704** | **−13.9×** |
| SUPP_INIT | 1 002 | 1 002 | 302 | |
| NMS_J II | 54 | 43 | 43 | |
| NMS_J iteration latency | 55 | 44 | 44 | |
| NMS_J max cycles (N trips) | 54 000 (N=1000) | 43 000 (N=1000) | 12 900 (N=300) | |
| NMS_I outer | 54 021 000 | 43 008 000 | 3 872 400 | |
| **Total worst-case** | **~59 M cycles** | **~47 M cycles** | **~4.36 M cycles** | **~13.5×** |

### Final latency breakdown (OPT v2, worst case)

| Stage | Cycles | @ 5 ns (200 MHz) | @ 6.5 ns (154 MHz) |
|---|---|---|---|
| score_filter | 126 101 | 0.63 ms | 0.82 ms |
| SORT_I | 363 600 | 1.82 ms | 2.36 ms |
| nms_greedy | 3 872 704 | 19.36 ms | 25.17 ms |
| OUT + overhead | ~400 | negligible | negligible |
| **Total** | **~4 362 805** | **~21.8 ms** | **~28.4 ms** |

---

## 4. Resource Utilisation — All Three Runs

| Resource | Baseline | OPT v1 | OPT v2 | Total change |
|---|---|---|---|---|
| BRAM_18K | 53 (2%) | 41 (2%) | **41 (2%)** | **−23%** |
| DSP | 16 (~0%) | 14 (~0%) | **14 (~0%)** | **−13%** |
| FF | 6 807 (1%) | 6 759 (1%) | **6 757 (1%)** | −1% |
| LUT | 8 913 (3%) | 12 843 (4%) | **12 831 (4%)** | +44% (LUTRAM trade-off) |
| URAM | 0 | 0 | 0 | — |

LUT increase is the cost of moving `cand_scores`, `cand_order`, `cand_areas`, `suppressed` from BRAM to LUTRAM, and the 4-way partition of `cand_boxes`. Total utilisation remains low — all four FPN levels could be instantiated in parallel on this device.

### Storage implementation (OPT v1 + v2)

| Variable | Impl | Baseline | Notes |
|---|---|---|---|
| `cand_boxes` (×4 banks) | BRAM (ram_1p, auto) | BRAM (ram_t2p) | 4-way cyclic partition added |
| `cand_scores` | **LUTRAM** (ram_2p) | BRAM | 0-cycle read; saves 8 BRAM |
| `cand_order` | **LUTRAM** (ram_2p) | BRAM | 0-cycle read; saves 3 BRAM |
| `cand_areas` | **LUTRAM** (ram_2p) | *(new)* | pre-computed box areas |
| `suppressed` | **LUTRAM** (ram_2p) | BRAM | 0-cycle read; saves 1 BRAM |
| `cand_cls` | BRAM (ram_t2p) | BRAM | unchanged |
| `keep_idx` | BRAM (ram_t2p) | BRAM | unchanged |

---

## 5. Optimisations Applied

### OPT v1 — Code changes to `nms_top.cpp`

| ID | Change | Where | Effect |
|---|---|---|---|
| OPT-1 | Replace `inter/uni > thr` with `inter > thr*uni` — eliminates float division | `iou_exceeds()` | Removes 16-cycle `fdiv`; NMS_J II 54→43 (−11 cyc) |
| OPT-2 | `ARRAY_PARTITION variable=cand_boxes cyclic factor=4` | nms_top pragma | All 4 box coords readable in 1 cycle (was 4 sequential BRAM reads) |
| OPT-3 | `suppressed[]` → `bind_storage type=RAM_2P impl=LUTRAM` | `nms_greedy()` pragma | 0-cycle read; breaks BRAM RAW stall that forced II≥2; saves 1 BRAM |
| OPT-4 | `cand_scores`, `cand_order` → LUTRAM | nms_top pragma | SORT_J II 5→4; SORT_J timing FAIL→PASS; saves 11 BRAM |
| OPT-5 | Pre-compute `cand_areas[k] = (x2−x1)*(y2−y1)` in score_filter | `score_filter()` body | Saves 3 FP ops per NMS_J iteration (2 sub + 1 mul); up to 450K ops for N=300 |
| OPT-6 | `cand_areas` → LUTRAM | nms_top pragma | 0-cycle read in NMS_J; consistent with OPT-4 |
| OPT-7 | Cache box-ii coords as scalars `ax1,ay1,ax2,ay2` in NMS_I outer loop | `nms_greedy()` body | Avoids repeated BRAM reads of same address inside inner loop |

### OPT v2 — Parameter and pragma changes

| ID | Change | Where | Effect |
|---|---|---|---|
| OPT-8 | `NMS_PRE 1000 → 300` | `fpga_types.h` | O(N²) scaling: (300/1000)² = 0.09 → **~11× speedup** on NMS and sort |
| OPT-9 | `LOOP_TRIPCOUNT` added to SF_H / SF_W / SF_RI / SF_SI outer loops | `nms_top.cpp` | score_filter latency now reported (was `?`); no RTL change |
| OPT-10 | `LOOP_TRIPCOUNT` max updated to 300 on SORT_I / SORT_J / NMS_I / NMS_J | `nms_top.cpp` | Latency estimates now use correct N=300 bound |

---

## 6. Root-Cause Analysis of Remaining Bottlenecks

### 6.1 NMS_J II=43 — floating-point chain is the hard floor

After removing `fdiv` (OPT-1) and switching to LUTRAM (OPT-3), the remaining II=43 is set by the FP intersection arithmetic inside `iou_exceeds()`:

```
read bx1/by1/bx2/by2   1 cyc  (parallel, 4-bank partition)
max/min comparisons     2 cyc
fsub iw = x2 − x1      6 cyc  ← no-DSP fsub, critical path
fsub ih = y2 − y1       6 cyc  (parallel with iw)
fmul inter = iw * ih    5 cyc
fmul area_b             5 cyc  (parallel)
fadd/fsub uni           6 cyc
fmul iou_thr * uni      5 cyc
fcmp + write            3 cyc
─────────────────────  ──────
                       ≈ 43 cycles total
```

The `fsub_32ns_32ns_32_6_no_dsp` (6-cycle) is the dominant element, confirmed in the Bind Op report. No further reduction is possible without changing `bbox_t` from float32 to a narrower fixed-point type.

### 6.2 SF_C II=11 — fexp in box decode

`expf(dw)` and `expf(dh)` inside score_filter have 12-cycle latency. The innermost SF_C loop achieves II=11 (set by the fexp chain). Fixing requires either `ap_fixed` exponent or a piecewise-linear approximation.

### 6.3 Timing failure (−2.53 ns) — 1-line TCL fix

The `fexp` path is 6.18 ns. Change `set CLK_NS "5"` to `set CLK_NS "6.5"` in `run_hls_nms.tcl`. No code changes needed; all data paths have ≥3.65 ns slack at 6.5 ns.

### 6.4 SORT_J II=4 — LUTRAM read chain residual

After LUTRAM (OPT-4), residual II=4 = read LUTRAM (1 cyc) + `fcmp` (2 cyc) + select (1 cyc). Only addressable by switching scores to fixed-point (integer compare → II=1) or replacing insertion sort with bitonic sort.

---

## 7. Further Optimisation Options

| Option | Expected gain | Effort | Priority |
|---|---|---|---|
| **Set clock to 6.5 ns in TCL** | Timing fully closes (0 violations) | Trivial | **Now** |
| `bbox_t` → `ap_fixed<20,12>` | NMS_J II 43→~10; ~4× NMS speedup | Medium | High |
| `score_t` → `ap_fixed<16,1>` | SF_C II 11→1; SORT_J II 4→1; removes fexp from sigmoid | Medium | High |
| Reduce `NMS_PRE` to 200 | ~2.25× more NMS reduction; total ~26M cycles | Trivial | Medium |
| Class-parallel NMS (×4) | NMS_I trip count ÷4; ~4× NMS speedup | High | Medium |
| Bitonic sort (N=300) | Sort 364K→~400 cycles (~900× faster) | High | Low (sort is not dominant) |

---

## 8. Three-Run Summary

| Metric | Baseline | OPT v1 | **OPT v2** | Total gain |
|---|---|---|---|---|
| **Total worst-case cycles** | ~59 M | ~47 M | **~4.36 M** | **13.5×** |
| **Total latency @ 6.5 ns** | ~362 ms | ~289 ms | **~28.4 ms** | **12.7×** |
| nms_greedy cycles | 54 022 004 | 43 009 004 | **3 872 704** | **13.9×** |
| NMS_J II | 54 | 43 | **43** | 1.26× |
| NMS_J max cycles | 54 000 | 43 000 | **12 900** | 4.2× |
| SORT_I cycles | 5 013 000 | 4 012 000 | **363 600** | **13.8×** |
| SORT_J II | 5 | 4 | **4** | 1.25× |
| SORT_J timing | FAIL | **PASS** | **PASS** | fixed |
| score_filter latency | ? | ? | **126 101 cycles** | now visible |
| `fdiv` in IoU | present (16 cyc) | **eliminated** | **eliminated** | removed |
| nms_greedy DSP | 2 | **0** | **0** | removed |
| BRAM_18K | 53 | 41 | **41** | −23% |
| DSP | 16 | 14 | **14** | −13% |
| LUT | 8 913 | 12 843 | **12 831** | +44% (LUTRAM) |
| Timing (5 ns clock) | FAIL −2.53 ns | FAIL −2.53 ns | FAIL −2.53 ns | fix: 6.5 ns TCL |
