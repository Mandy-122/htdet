# retina_head_top — HLS Synthesis Log

**Module**: `retina_head_top` — RetinaNet detection head (stacked convs + prediction convs)  
**Device**: xczu9eg-ffvb1156-2-e (ZCU102, ZynqUltraScale+), Clock: 5 ns (200 MHz)  
**Config**: HEAD_FEAT_CH=192, HEAD_STACKED_CONVS=4, CLS_OUT_CH=REG_OUT_CH=36  
**Spatial**: TEST_H=10, TEST_W=10 (P6 level, 640×640 input)  
**Weight types**: int8 (W8A32 PTQ conv kernels), float32 (dequant scale + bias)  
**Activation type**: float32

**Pipeline** (per FPN level):
```
feat_in [192, H, W]
  ├─ cls branch: 4 × Conv3x3(192→192, int8+scale+bias, ReLU)  →  Conv3x3(192→36, int8+scale+bias)
  │              → cls_logits [36, H, W]
  └─ reg branch: same 4-layer stack                            →  Conv3x3(192→36, …)
                 → reg_deltas [36, H, W]
```

---

## Run History

| Run | Date | II (CO_ICT) | Latency | BRAM_18K | DSP | FF | LUT | Timing | Status |
|-----|------|-------------|---------|----------|-----|----|-----|--------|--------|
| **v1** | 2026-06-01 | — | ∞ (hung) | — | — | — | — | — | ❌ Scheduler hang |
| **v2** | 2026-06-01 | **72** | 1.721 sec | 192 (10%) | 35 (1%) | 78,762 (14%) | 148,671 (54%) | 4.202 ns ✅ | ✅ First working synthesis |
| **v3** | 2026-06-01 | — | killed | — | — | — | — | — | ❌ Elaboration OOM (w_row[1728] complete × 8 calls) |
| **v4** | 2026-06-02 | **72** | 1.721 sec | 192 (10%) | 35 (1%) | 78,762 (14%) | 148,671 (54%) | 4.202 ns ✅ | ✅ Same as v2 — confirmed bandwidth bottleneck |
| **v5** | pending | **≤4** (predicted) | **~55 ms** | ~192 | ~35 | ~80K | ~20% | <5 ns | ⏳ w_row cyclic-144 |

Available on xczu9eg: BRAM_18K=1824, DSP=2520, FF=548,160, LUT=274,080, URAM=0

---

## Run v1 — Original (HUNG, never completed)

**Problem 1 — `#pragma HLS PIPELINE II=1` on `CO_OW` with 144 fully-unrolled FP32 MACs**

The body of the spatial output-pixel loop contained `IC_TILE=16 × KH=3 × KW=3 = 144` MACs.
FP32 multiply-accumulate has ~7-cycle latency → achievable II is ≥ 35, not 1.
HLS's ILP scheduler spends unbounded time trying to prove an infeasible constraint.
Synthesis never produced output after "Finished Source Code Analysis and Preprocessing".

**Problem 2 — `ARRAY_PARTITION cyclic factor=16` on 1.3 M-element BRAM-interface ports**

`cls_conv_w` / `reg_conv_w` each have 1,327,104 int8 elements. With factor=16, HLS creates
82,944-element banks and must verify 16 simultaneously independent accesses across the full
1.3 M-element address space for every pipeline cycle. Analysis blows up the scheduling graph.

**Fix applied for v2**: Pipeline moved to `CO_ICT` (IC-tile loop); `ARRAY_PARTITION` removed
from all weight port arrays; H and W changed to compile-time `TEST_H` / `TEST_W` macros so
boundary conditions are statically pruned.

---

## Run v2 — First Working Synthesis (2026-06-01 16:01)

### Overall latency

| Module | Cycles | @ 200 MHz | % total |
|--------|--------|-----------|---------|
| `retina_conv_once` — cls branch (4 layers) | 42,624,001 | 213 ms | 49% |
| `retina_conv_once_1` — reg branch (4 layers) | 42,643,201 | 213 ms | 49% |
| `retina_pred_conv` — cls + reg | 3,111,760 | 15.6 ms | 1% |
| **Total** | **344,218,985** | **1.721 sec** | 100% |

### Resources

| BRAM_18K | DSP | FF | LUT | URAM | Timing (est.) |
|----------|-----|----|-----|------|---------------|
| 192 (10%) | 35 (1%) | 78,762 (14%) | 148,671 (**54%** ⚠) | 0 | **4.202 ns ✅** |

### Per-pipeline loop

| Loop | Module | II achieved | II target | Trip count | Latency | Notes |
|------|--------|-------------|-----------|------------|---------|-------|
| `CO_ICT` | `retina_conv_once_Pipeline_CO_ICT` | **72** | 1 | 12 | 2,105 cy | Weight BRAM bandwidth |
| `CO_OC_CO_OH_CO_OW` | `retina_conv_once` | sequential | — | 19,200 | 42,624,000 cy | Outer spatial loop |
| `RP_OC_RP_OW_RP_ICT` | `retina_pred_conv` | **72** | 1 | 43,200 | 3,111,758 cy | Same bottleneck |

### Instance resource breakdown

| Instance | DSP | FF | LUT |
|----------|-----|----|-----|
| `retina_conv_once` (cls) | 10 | 25,668 | 46,918 |
| `retina_conv_once_1` (reg) | 10 | 26,410 | 48,887 |
| `retina_pred_conv` | 15 | 26,663 | 47,219 |

### BRAM breakdown

192 BRAM_18K total — entirely from the ping-pong `tmp` buffer inside `retina_stacked_convs`.
- Buffer: `act_t tmp[192 × 10 × 10] = 19,200 floats`, partitioned cyclic-16 → 16 banks
- HLS instantiated 3 copies of the function → 3 × 16 banks × 4 BRAM_18K = 192 BRAM_18K

---

## Bottleneck Analysis (v2)

### 1. CO_ICT II=72 — Weight BRAM Bandwidth (primary bottleneck)

Each ICT pipeline body has `IC_TILE=16 × KH=3 × KW=3 = 144` weight reads from `lw`.
`cls_conv_w` is a 2-port BRAM-interface port with **no ARRAY_PARTITION** (removed in v2 fix).

```
II = weight reads per tile / BRAM read ports = 144 / 2 = 72  (exact match)
```

HLS allocated only 5 DSPs (1 fadd + 1 fmul + address adders) because only one FP result is
needed every 72 cycles — the design is **memory-bandwidth-bound, not compute-bound**.

### 2. LUT=54% — Boundary Condition Mux Explosion

The synthesis report shows **164 instances of `mux_164_32_1_1`** (164-input, 32-bit output mux).
Each of the 144 unrolled kernel positions (IC_TILE=16, KH=3, KW=3) generates one address mux
for the boundary-condition check `if (ih >= 0 && ih < TEST_H && iw >= 0 && iw < TEST_W)`.
With 164 possible address states per mux and 164 mux instances, this dominates LUT usage.

Even with compile-time TEST_H=10 and TEST_W=10, HLS generates separate mux trees for every
unrolled (k, kh, kw) combination because the output address depends on the (oc, oh, ow) loop
variables which are not fully unrolled.

---

## Run v3 — Weight Row Buffer (FAILED — elaboration killed)

**Change**: `weight_t w_row[1728]` with `#pragma HLS ARRAY_PARTITION complete` declared inside
`retina_conv_once`. The function is called 8 times (4 layers × 2 branches). HLS elaboration
creates 8 × 1728 individually-wired registers, each needing routing analysis for 144 concurrent
read paths → 8 × 1728 × 144 ≈ 2 M connection checks. HLS ran out of memory/time.

**Symptom**: "Finished Source Code Analysis" then "Finished C synthesis." with immediate
"Task has been cancelled!" — process was OOM-killed during routing elaboration, not during
scheduling. Different from the v1 scheduler hang.

**Lesson**: `complete` partition on arrays larger than ~64 elements inside non-trivially-called
functions causes elaboration explosion. Use `cyclic factor=N` instead, or restrict buffer size.

---


## Run v4 — Remove KH/KW UNROLL (2026-06-02 01:00) — II=72, SAME as v2

**Synthesis result**: Identical to v2. CO_ICT II=72, latency=1.721 sec, LUT=54%.

**Why removing KH/KW UNROLL had no effect — the fundamental insight:**

The II=72 is NOT caused by concurrent reads within a single pipeline stage.
It is caused by **total BRAM bandwidth consumed per tile across overlapping pipeline stages**.

```
Every tile t reads:  IC_TILE × KH × KW = 16 × 3 × 3 = 144 weights from lw
Available BRAM bandwidth: 2 reads/cycle
Minimum II = 144 / 2 = 72   ← same whether reads are concurrent (v2) or sequential (v4)
```

When CO_ICT pipelines with II=N, up to N overlapping tile iterations share the lw BRAM
simultaneously. Whether each tile's 144 reads are done "all at once" (unrolled) or "9 steps
of 16 at a time" (sequential), the aggregate bandwidth demand is identical.
HLS correctly determines II=72 in both cases.

**Lesson**: Pragma changes to the unroll structure cannot break a fundamental bandwidth bottleneck.
The only fix is to eliminate BRAM reads from the hot loop entirely → weight row buffer.

---

## Run v5 — Weight Row Buffer cyclic-144 (pending synthesis)

**Root cause of v3 OOM**: `complete` partition on 1728 elements = 1728 individual connection
checks per access × 8 function calls = 2M elaboration paths → OOM killed.

**v5 fix**: Use `cyclic factor=144` instead of `complete`.

```
complete:      1728 banks × 1 element  — 1728 connection checks per access
cyclic-144:      144 banks × 12 elements — 144 connection checks per access  (12× less)
```

HLS does not inline `retina_conv_once` (creates hw module instances), so w_row exists in
ONE copy only — elaboration is 144 banks × 12 elements = completely manageable.

**Bank collision analysis (why cyclic-144 achieves II=1 from the weight side):**

For tile t, the 144 simultaneous CO_ICT accesses are:
```
w_row[ic × 9 + kh × 3 + kw]   where ic = t × 16 + k
index = (t×16+k)×9 + kh×3+kw  = t×144 + (k×9 + kh×3+kw)
bank  = index % 144            = (k×9 + kh×3+kw) % 144
```
For k∈[0,15], kh∈[0,2], kw∈[0,2]: `k×9+kh×3+kw` takes every value in [0..143] exactly once.
→ All 144 accesses land on **distinct banks** → zero bank conflict → II=1 from weights. ✓

**LOAD loop bank pattern** (`for i in 0..1727: w_row[i] = lw[oc×1728+i]`):
- bank = i % 144; consecutive i → consecutive banks → always distinct → II=1. ✓

**Remaining bottleneck after v5**: src activation array (cyclic-16).
With KH/KW sequential, only 16 concurrent src reads per pipeline stage.
src bank = (ic×100 + ih×10+iw) % 16. For k=0..15: k×100%16 = 4k%16 → 4 distinct banks.
16 reads / 4 banks = 4 reads/bank → II ≤ 4 from src (better than II=72 from lw).

**Expected latency (v5):**

| Phase | Cycles | @ 200 MHz |
|---|---|---|
| LOAD loop (per OC row) | 192 × 1728 = 331,776 | 1.7 ms/call |
| CO_ICT compute (II≤4) | 19,200 × 12 × 4 = 921,600 | 4.6 ms/call |
| **Total per retina_conv_once call** | **~1.3 M** | **~6.5 ms** |
| **Total head (P6, 8 calls + pred)** | **~11 M** | **~55 ms** |

vs v2: **344 M cycles → 1.721 sec**  →  **~32× speedup predicted**

---

## II Violation Reference Table (complete)

| Run | Violation | Root Cause | Fix | Result |
|-----|-----------|------------|-----|--------|
| v1 | hung | II=1 on OW + 144 unrolled MACs; large ARRAY_PARTITION on BRAM ports | Restructure to CO_ICT pipeline | v2 synthesised |
| v2 | CO_ICT II=72 | Total 144 weight reads / 2 BRAM ports = 72 | Weight row buffer (v5) | **≤4** predicted |
| v3 | OOM killed | w_row[1728] complete × 8 calls → 2M connection checks | Use cyclic factor=144 (v5) | Manageable elaboration |
| v4 | CO_ICT II=72 | Same as v2 — total bandwidth bottleneck | Confirmed: only row buffer fixes it | v5 |
| v5 | CO_ICT II≤4 | src cyclic-16, 16 reads, 4 banks → 4 reads/bank | Accept or: add src line-buffer | Next if needed |

---

## Pending Optimizations

| Optimization | Target II | Expected Latency | Status |
|---|---|---|---|
| Weight row buffer cyclic-144 (v5) | **≤4** | **~55 ms** | ⏳ Pending synthesis |
| src line-buffer (per-row cache) | **1** | **~15 ms** | Planned after v5 confirmed |
| Scale to P5 (20×20) | — | ~4× P6 | After v5 on P6 |
| Scale to P3 (80×80) | — | ~64× P6 | Future |

---

## File Map

| File | Description |
|---|---|
| `fpga_types.h` | Types + constants. Set `TEST_H` / `TEST_W` to select FPN level. |
| `fpga_utils.h` | `sigmoid`, `relu` (piecewise-linear under synthesis). |
| `retina_head_top.h` | Top function declaration + element-count macros. |
| `retina_head_top.cpp` | v5: w_row[1728] cyclic-144, KH/KW sequential, CO_IC UNROLL. |
| `testbench_retina_head.cpp` | Test 1: zero weights → [PASS]. Test 2: real PTQ weights + synthetic feature → stats. |
| `run_hls_retina_head.tcl` | Vitis HLS script. Edit `TEST_H` / `TEST_W` to step through P6→P2. |
| `gen_csim_ref.py` | Python reference. Run `--syn --h 10 --w 10` to cross-check CSIM stats. |
| `hls_stubs/` | `ap_int.h`, `ap_fixed.h`, `hls_stream.h` for standalone `g++` compile. |
