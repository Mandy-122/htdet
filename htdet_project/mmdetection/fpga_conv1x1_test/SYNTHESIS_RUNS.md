# conv1x1_bn_silu Synthesis Run History

**Module**: `conv1x1_top` — tiled IC conv1x1 + BN + SiLU  
**Device**: xczu28dr-ffvg1517-2-e (ZynqUltraScale+)  
**Clock**: 5 ns target (200 MHz)  
**Layout**: HWC — `in[hw * IC + ic]`, `w_conv[oc * IC + ic]`  
**Partitions**: `in` cyclic factor=16, `w_conv` cyclic factor=16  

---

## Performance Results

| Run | IC | OC | H×W | IC tiles | II (C1X1S_IC_T) | IC_T latency | Outer iter | Total cycles | @ 200 MHz | Synth time |
|-----|----|----|-----|----------|-----------------|--------------|------------|--------------|-----------|------------|
| Tiny      | 32  | 64  | 8×8   | 2  | 5 | 151 cycles | 207 cycles | 847,874    | 4.24 ms   | < 1 min |
| 1/4 scale | 64  | 64  | 20×20 | 4  | 5 | 161 cycles | 217 cycles | 5,555,201  | 27.78 ms  | < 1 min |
| 1/2 scale | 128 | 128 | 40×40 | 8  | 5 | 181 cycles | 237 cycles | 48,537,600 | 242.7 ms  | < 1 min |
| Full scale | 256 | 256 | 80×80 | 16 | 5 | 221 cycles | 277 cycles | 453,836,800 | **2.269 s** | **17 s** |
| v3 (II=1, tree reduce) | 256 | 256 | 80×80 | 16 | **1** | **155 cycles** | **243 cycles** | **398,131,200** | **1.99 s** | < 1 min |
| v4 (OC_TILE=2)         | 256 | 256 | 80×80 | 16 | **1** | **155 cycles** | **243 cycles** | **199,065,600** | **0.995 s** | < 1 min |

### Latency scaling law (II=5 runs)
```
IC_T latency  = 5 × (NUM_TILES - 1) + 147      (pipeline depth = 147 fixed)
Per outer iter = IC_T latency + ~56              (BN + SiLU + call overhead)
Total cycles   = OC × H × W × per_outer_iter
```

---

## Hardware Resources

| Run | DSP | LUT | FF | BRAM | URAM | Timing (est.) | Fmax |
|-----|-----|-----|----|------|------|---------------|------|
| Tiny       | 27 | 4,787 | 6,079 | 0 | 0 | 3.798 ns | ~263 MHz |
| 1/4 scale  | 27 | 4,787 | 6,076 | 0 | 0 | 3.798 ns | ~263 MHz |
| 1/2 scale  | 27 | 4,773 | 6,076 | 0 | 0 | 3.798 ns | ~263 MHz |
| Full scale | 28 | 4,805 | 6,130 | 0 | 0 | 3.798 ns | 263.27 MHz |
| v3 (II=1, tree reduce) | **88** | **8,411** | **11,485** | 0 | 0 | 3.828 ns | — |
| v4 (OC_TILE=2)         | **175** | **16,109** | **21,685** | 0 | 0 | 3.828 ns | — |

**Key observation**: Resources are invariant to IC/OC/H/W — determined entirely by the pipeline structure (TILE=16). Scaling the dimensions costs no extra hardware.

---

## Failed / Corrected Runs

| Issue | Symptom | Root cause | Fix |
|-------|---------|------------|-----|
| factor=17 with HWC layout | II=8, `urem_12ns` (16-cycle division), `mux_175` | `(hw×IC + ic + k) % 17` requires hardware divider since 17 is not power-of-2 | Change to `factor=16`; bank = `(ic+k) % 16 = k` (constant per k → trivial mask) |
| `#ifndef CONV1X1_TOP_H` guard (old `conv1x1_` naming) | HLS parse error: "extra tokens at end of #ifndef" at col 14 | Vitis HLS 2022.2 Clang stops parsing `CONV1X1_TOP_H` at the `X` character | Rename files to `conv1_1_*` with guard `CONV1_1_TOP_H` |
| PIPELINE on OC×HW loop (old approach) | 24+ hour scheduling stall | Full unroll of IC=256/384 → 200K+ instruction basic block; ILP infeasible at II=1 | Tile IC in blocks of 16, PIPELINE on IC tile loop |

---

## Optimization History

### v1 — Old (broken): PIPELINE on OC×HW outer loop
```cpp
C1X1S_HW: for (int hw ...) {
    #pragma HLS PIPELINE II=1          // forces full unroll of all inner loops
    C1X1S_IC: for (int ic ...) {       // ← gets fully unrolled: 256/384 iterations
        #pragma HLS UNROLL factor=16   // ← IGNORED (overridden by outer PIPELINE)
        acc += ...
```
- Result: 200K+ instruction basic block → scheduler ran for 24+ hours

### v2 — Tiled IC, PIPELINE on IC_T (current): II=5
```cpp
C1X1S_IC_T: for (int ic = 0; ic < in_ch; ic += 16) {
    #pragma HLS PIPELINE II=1
    tile = sum of 16 MACs;
    acc += tile;            // ← loop-carried float-add (latency=5) → II=5
}
```
- Result: scheduling in seconds, II=5, 2.27s @ full scale

### v3 — Partial accumulators, no recurrence (pending): II=1
```cpp
acc_t partial[NUM_TILES];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1

C1X1S_IC_T: for (int t = 0; t < NUM_TILES; t++) {
    #pragma HLS PIPELINE II=1
    partial[t] = sum of 16 MACs;   // ← independent write, no recurrence → II=1
}
// unrolled log2 reduction tree after loop
acc = reduce(partial[]);
```
- Expected: II=1, ~2× speedup on total latency

---

## Pending / Future Optimizations

| Idea | Expected gain | Cost | Status |
|------|--------------|------|--------|
| Partial accumulators + tree (v3) | 12% latency reduction (2.27s→1.99s), II=1 | 3× DSP (28→88) — poor trade-off for production | **Done** |
| OC_TILE=2 (process 2 OCs per IC pass) | ~2× additional | 2× weight bandwidth, 2× DSPs | Planned |
| OC_TILE=4 | ~4× additional | 4× weight bandwidth, 4× DSPs | Future |
| Push clock to 250 MHz (4 ns) | 25% throughput | Timing margin = 1.2 ns (tight) | Investigate |
| Apply to fpga_mbconv_80 / mbconv_40 | Production modules | Port conv1_1 pragma pattern to fpga_utils.h | Next after v3 validated |

---

## Notes

- **II=5 root cause**: `acc += tile` float add has latency ~5 cycles. This is the minimum physical limit for FP accumulation with loop-carry — cannot be reduced without restructuring (see v3).
- **Pipeline depth = 147 cycles**: Fixed by the MAC computation graph (sitofp + fmul + fadd tree). Independent of IC/OC/H/W.
- **Fmax 263 MHz** at full scale: 32% headroom over 200 MHz target. Could run at up to 250 MHz (4 ns clock) with some margin.
- **Synthesis time 17s** at full scale: Confirms tiled approach solves the 24-hour scheduling problem across all dimensions.
- **BRAM = 0**: All arrays are passed as external BRAM ports (16 banks for `in`, 16 for `w_conv`). No internal BRAMs consumed.
