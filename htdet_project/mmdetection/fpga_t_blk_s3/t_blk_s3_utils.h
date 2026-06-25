/*
 * t_blk_s3_utils.h  (v2 — II-violation fixes)
 * Primitive operations for transformer_block_s2:
 *   silu_f, layer_norm, linear_f, softmax_vec, mhsa, add_residual
 *
 * Fixes applied vs v1 (based on synthesis report analysis):
 *
 *  Fix 1 — layer_norm LN_MEAN/LN_VAR: II=5 → II=1
 *    Root cause: sequential FP accumulation `sum += x` carries RAW through `sum`
 *    (float adder latency = 5 cycles → II = 5).
 *    Fix: 4 independent parallel accumulators per pass → no loop-carried dep → II=1.
 *    Requires dim % 4 == 0 (true for all TB_DIM values: 16, 32, 64, 128, 144).
 *
 *  Fix 2 — linear_f LF_ID: II=9 (in_dim=16) / II=17 (in_dim=32) → II=1 (tile loop)
 *    Root cause: LF_ID was completely unrolled (16 or 32 parallel reads needed) but
 *    `w` is an external dual-port BRAM (only 2 reads/cycle) → II = in_dim/2 + 1.
 *    Fix: partial accumulator tile approach (LF_TILE=2 matches BRAM dual-port).
 *    Each tile reads exactly 2 `w` elements (one per BRAM port) → II=1 on tile loop.
 *    Tree reduce at the end (same pattern as conv1x1_bn in fpga_mbconv_80).
 *    Requires in_dim % 2 == 0 (true for all dims in this module).
 *
 *  Fix 3 — MHSA_SM: II=74 → II=1 (with DEPENDENCE false)
 *    Root cause: `softmax_vec(a_buf + qi*TB_SEQ)` — HLS can't prove non-aliasing
 *    between successive qi iterations through the pointer (stores to a_buf[qi*N:...]
 *    in iter qi, loads from a_buf[(qi+1)*N:...] in iter qi+1 — different slices,
 *    but HLS conservatively stalls for the full softmax latency = 74 cycles).
 *    Fix: `#pragma HLS DEPENDENCE variable=a_buf inter false` tells HLS that
 *    inter-iteration dependences on a_buf are false (slices are non-overlapping).
 *
 *  Fix 4 — MHSA_QI_MHSA_KI q_buf: II=4 → II=1
 *    Root cause: q_buf not partitioned; MHSA_D inner loop (HEAD_DIM=8) is unrolled,
 *    needing 8 q_buf reads/cycle but only 2 BRAM ports available.
 *    Fix: `#pragma HLS ARRAY_PARTITION variable=q_buf cyclic factor=4` in top.cpp.
 *    With 4 banks each having 2 ports → 8 reads/cycle → II=1.
 *    (k_buf and v_buf already had this inferred automatically by HLS.)
 */

#ifndef T_BLK_S3_UTILS_H
#define T_BLK_S3_UTILS_H

#include "t_blk_s3_types.h"
#include <cmath>

// (tile macros removed — v3 reverts linear_f to flat 2-loop pipeline with UNROLL factor=2)

// ============================================================
// SiLU activation
// ============================================================
inline act_t silu_f(act_t x) {
    #pragma HLS INLINE
    #ifdef __SYNTHESIS__
    act_t sig = (x < (act_t)(-4.0f)) ? (act_t)0.0f :
                (x > (act_t)( 4.0f)) ? (act_t)1.0f :
                (act_t)0.5f + x * (act_t)0.125f;
    return x * sig;
    #else
    return (act_t)((float)x / (1.0f + expf(-(float)x)));
    #endif
}

// ============================================================
// LAYER NORM  (v4: tile approach — local tile_sum, no loop-carry → II=1)
//
// Previous v2/v3 used 4 running accumulators (ms0..ms3) declared OUTSIDE the
// loop.  They are updated every iteration → period=1, FP-add latency=5 → II=5.
// Console confirmed: "Unable to enforce carried dependence on ms2 (II=1..4)".
//
// Fix: declare `tile_sum` INSIDE the loop body (local, not loop-carried).
//   Each iteration accumulates 4 elements into a fresh local variable, then
//   writes the result to partial[t] exactly once → zero loop-carry → II=1.
//   The partial[] array is tree-reduced after the loop (fully unrolled).
//
// Requires dim % 4 == 0 (true for DIM=16,32,64,128,144).
// partial[] size is TB_DIM/4 (compile-time constant → complete partition OK).
// ============================================================
inline void layer_norm(
    const act_t*  in,  act_t*  out,
    const meta_t* w,   const meta_t* b,
    int seq, int dim
) {
    #pragma HLS INLINE
    const int N_TILES = TB_DIM / 4;   // compile-time: 4 / 8 / 16 / 36 per step

    for (int s = 0; s < seq; s++) {
        const act_t* row = in  + s * dim;
        act_t*       dst = out + s * dim;

        // ── Mean: tile approach ──────────────────────────────────────────────
        acc_t mp[N_TILES];
        #pragma HLS ARRAY_PARTITION variable=mp complete dim=1

        LN_MEAN: for (int t = 0; t < dim / 4; t++) {
            #pragma HLS PIPELINE II=1
            acc_t tile = 0.0f;          // LOCAL → no loop-carry
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                tile += (acc_t)row[t * 4 + k];
            }
            mp[t] = tile;               // written once per iter → no carry
        }
        // Tree reduce (fully unrolled, combinational)
        acc_t msum = 0.0f;
        LN_MRED: for (int t = 0; t < N_TILES; t++) {
            #pragma HLS UNROLL
            msum += mp[t];
        }
        act_t mean = (act_t)(msum / (acc_t)dim);

        // ── Variance: same tile approach ──────────────────────────────────────
        acc_t vp[N_TILES];
        #pragma HLS ARRAY_PARTITION variable=vp complete dim=1

        LN_VAR: for (int t = 0; t < dim / 4; t++) {
            #pragma HLS PIPELINE II=1
            acc_t tile = 0.0f;          // LOCAL → no loop-carry
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                acc_t d = (acc_t)row[t * 4 + k] - (acc_t)mean;
                tile += d * d;
            }
            vp[t] = tile;
        }
        acc_t vsum = 0.0f;
        LN_VRED: for (int t = 0; t < N_TILES; t++) {
            #pragma HLS UNROLL
            vsum += vp[t];
        }
        act_t inv_std = (act_t)(1.0f / sqrtf((float)(vsum / (acc_t)dim) + 1e-5f));

        // ── Normalize + affine (no accumulation → II=1) ───────────────────────
        LN_NORM: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            acc_t n = ((acc_t)row[i] - (acc_t)mean) * (acc_t)inv_std;
            dst[i]  = (act_t)(n * (acc_t)w[i] + (acc_t)b[i]);
        }
    }
}

// ============================================================
// LINEAR LAYER  (v3: flat 2-loop pipeline, UNROLL factor=2)
//
// [seq, in_dim] × [out_dim, in_dim]^T + bias → [seq, out_dim]
// weights: w[od * in_dim + id]
//
// v1 used factor=8 → HLS fully unrolled LF_ID → needed 16+ reads from w per cycle
//    → II=9 (dual-port BRAM can only serve 2 reads/cycle → II = in_dim/2 + 1).
// v2 tile+tree approach → broke outer-loop flatness for Q/K/V → 4800 cycles regression.
//
// v3 fix: factor=2 exactly matches dual-port BRAM (2 reads/cycle).
//   - Flat (LF_S × LF_OD) pipelined outer loop is preserved (no inner tile nesting).
//   - HLS sees 2 w reads per outer pipeline stage → II=2 achievable.
//   - Expected: 64 iterations × II=2 + depth ≈ 200+ cycles per call (vs 704 v1, 4800 v2).
//   - Accumulation FP-add latency goes into pipeline depth (not II), because `acc`
//     is local to each (s,od) iteration — no cross-iteration dependency.
// ============================================================
inline void linear_f(
    const act_t*  in, act_t* out,
    const meta_t* w,  const meta_t* bias,
    int seq, int in_dim, int out_dim
) {
    #pragma HLS INLINE
    LF_S: for (int s = 0; s < seq; s++) {
        LF_OD: for (int od = 0; od < out_dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)bias[od];
            LF_ID: for (int id = 0; id < in_dim; id++) {
                // factor=2: exactly 2 w reads per outer pipeline stage
                // → matches external dual-port BRAM → II=2 on outer loop
                #pragma HLS UNROLL factor=2
                acc += (acc_t)in[s * in_dim + id] * (acc_t)w[od * in_dim + id];
            }
            out[s * out_dim + od] = (act_t)acc;
        }
    }
}

// ============================================================
// SOFTMAX  (v5: scalable to SEQ=400 without resource explosion)
//
// SM_MAX  — UNROLL: LUT comparator tree, no DSP, combinational.
// SM_EXP  — PIPELINE II=1: 1 expf() unit reused.
// SM_SUM  — TILE approach (TB_SM_TILE=4): avoids large FP-adder tree.
//             At SEQ=100 UNROLL would instantiate ~50 FP adders (100 DSP).
//             Tile: local tile_sum (no loop-carry) → II=1 on SM_SUM_T,
//             tree-reduce TB_SM_TILES partials (25 FP adders max) → ~50 DSP.
// SM_NORM — PIPELINE II=1: 1 fdiv unit reused.
//
// MHSA_SM caller must use PIPELINE (not UNROLL) at SEQ>16 to prevent
// replication: UNROLL at SEQ=100 would create 100 copies of softmax_vec,
// each with its own SM_SUM tree (100×50=5000 FP adders — infeasible).
// With MHSA_SM PIPELINE: all sub-loops are 1 instance, reused per token.
// ============================================================
inline void softmax_vec(act_t* v) {
    #pragma HLS INLINE

    // Max: sequential pipeline (1 read/cycle from a_buf — avoids 100-reg simultaneous read).
    // UNROLL at SEQ>16 requires complete partition of a_buf (10K registers at SEQ=100).
    // PIPELINE: float comparison+select is 1-2 cycles, II=1 on xczu28dr.
    act_t mx = v[0];
    SM_MAX: for (int i = 1; i < TB_SEQ; i++) {
        #pragma HLS PIPELINE II=1
        if (v[i] > mx) mx = v[i];
    }

    // Exp: PIPELINE II=1 — 1 expf() unit reused
    act_t ex[TB_SEQ];
    #pragma HLS ARRAY_PARTITION variable=ex complete dim=1
    SM_EXP: for (int i = 0; i < TB_SEQ; i++) {
        #pragma HLS PIPELINE II=1
        ex[i] = (act_t)expf((float)(v[i] - mx));
    }

    // Sum: tile approach — local tile_sum (no loop-carry) → II=1
    //   At SEQ=100: 25 tiles; tree-reduce 25 partials (~13 FP adders vs 50 with UNROLL)
    acc_t sp[TB_SM_TILES];
    #pragma HLS ARRAY_PARTITION variable=sp complete dim=1
    SM_SUM_T: for (int t = 0; t < TB_SM_TILES; t++) {
        #pragma HLS PIPELINE II=1
        acc_t tile = 0.0f;      // LOCAL: no loop-carry → II=1
        for (int k = 0; k < TB_SM_TILE; k++) {
            #pragma HLS UNROLL
            tile += (acc_t)ex[t * TB_SM_TILE + k];
        }
        sp[t] = tile;
    }
    acc_t s = 0.0f;
    SM_SUM_R: for (int t = 0; t < TB_SM_TILES; t++) {
        #pragma HLS UNROLL
        s += sp[t];
    }

    // Normalize: PIPELINE II=1 — 1 fdiv unit reused
    SM_NORM: for (int i = 0; i < TB_SEQ; i++) {
        #pragma HLS PIPELINE II=1
        v[i] = (act_t)((acc_t)ex[i] / s);
    }
}

// ============================================================
// MULTI-HEAD SELF-ATTENTION  (Fix 3: DEPENDENCE false on MHSA_SM)
// ============================================================
inline void mhsa(
    const act_t*  x,
          act_t*  out,
    const meta_t* q_w, const meta_t* q_b,
    const meta_t* k_w, const meta_t* k_b,
    const meta_t* v_w, const meta_t* v_b,
    const meta_t* o_w, const meta_t* o_b,
    act_t* q_buf, act_t* k_buf, act_t* v_buf, act_t* a_buf
) {
    #pragma HLS INLINE

    const float scale_f = 1.0f / sqrtf((float)TB_HEAD_DIM);
    const act_t scale   = (act_t)scale_f;

    // Step 1: Project Q, K, V
    linear_f(x, q_buf, q_w, q_b, TB_SEQ, TB_DIM, TB_DIM);
    linear_f(x, k_buf, k_w, k_b, TB_SEQ, TB_DIM, TB_DIM);
    linear_f(x, v_buf, v_w, v_b, TB_SEQ, TB_DIM, TB_DIM);

    // Step 2: Per-head attention
    MHSA_H: for (int h = 0; h < TB_HEADS; h++) {

        // Attention scores
        MHSA_QI: for (int qi = 0; qi < TB_SEQ; qi++) {
            MHSA_KI: for (int ki = 0; ki < TB_SEQ; ki++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0.0f;
                MHSA_D: for (int d = 0; d < TB_HEAD_DIM; d++) {
                    #pragma HLS UNROLL
                    dot += (acc_t)q_buf[qi * TB_DIM + h * TB_HEAD_DIM + d]
                         * (acc_t)k_buf[ki * TB_DIM + h * TB_HEAD_DIM + d];
                }
                a_buf[qi * TB_SEQ + ki] = (act_t)((acc_t)dot * (acc_t)scale);
            }
        }

        // Softmax (v5): PIPELINE + DEPENDENCE for Step 4+ (SEQ=100/400).
        // At SEQ<=16: UNROLL was used (16 sequential copies, compile-time offsets).
        // At SEQ=100: UNROLL creates 100 copies of SM_SUM/SM_MAX → resource explosion.
        //
        // PIPELINE strategy: HLS schedules one new qi per II cycles.
        // a_buf is complete-partitioned (registers) → no BRAM port conflicts.
        // DEPENDENCE pragma: slices [qi*SEQ..(qi+1)*SEQ-1] are non-overlapping;
        // HLS can't prove this through the pointer, so we declare it explicitly.
        // If DEPENDENCE is honored: II=1 (new token every cycle, depth ~300+).
        // If not: HLS assigns II=softmax_latency (sequential, still correct).
        MHSA_SM: for (int qi = 0; qi < TB_SEQ; qi++) {
            #pragma HLS PIPELINE
            #pragma HLS DEPENDENCE variable=a_buf inter false direction=RAW
            softmax_vec(a_buf + qi * TB_SEQ);
        }

        // Weighted V sum — tile approach for MHSA_OK (avoids 100-element full unroll).
        // TB_SM_TILE=4 reads per tile: a_buf at consecutive ki indices (ki=t*4+kt, kt=0..3).
        //   a_buf cyclic factor=5: (base+0..3) % 5 = 4 distinct banks → II=1 on inner tile.
        //   v_buf factor=TB_VBUF_FACTOR=101: (t*4+kt)*DIM % 101 distinct for kt=0..3 ✓.
        // op[] partial sums: complete partition (TB_SM_TILES registers, max 100 at Step5).
        // Tree-reduce after tile loop (MHSA_OK_R unrolled, combinational).
        MHSA_OQ: for (int qi = 0; qi < TB_SEQ; qi++) {
            MHSA_OD: for (int d = 0; d < TB_HEAD_DIM; d++) {
                acc_t op[TB_SM_TILES];
                #pragma HLS ARRAY_PARTITION variable=op complete dim=1
                MHSA_OK_T: for (int t = 0; t < TB_SM_TILES; t++) {
                    #pragma HLS PIPELINE II=1
                    acc_t tile = 0.0f;      // LOCAL: no loop-carry → II=1
                    for (int kt = 0; kt < TB_SM_TILE; kt++) {
                        #pragma HLS UNROLL
                        int ki = t * TB_SM_TILE + kt;
                        tile += (acc_t)a_buf[qi * TB_SEQ + ki]
                              * (acc_t)v_buf[ki * TB_DIM + h * TB_HEAD_DIM + d];
                    }
                    op[t] = tile;
                }
                acc_t acc = 0.0f;
                MHSA_OK_R: for (int t = 0; t < TB_SM_TILES; t++) {
                    #pragma HLS UNROLL
                    acc += op[t];
                }
                out[qi * TB_DIM + h * TB_HEAD_DIM + d] = (act_t)acc;
            }
        }
    }

    // Step 3: Output projection (copy out→q_buf as temp, then project)
    MHSA_CPY: for (int i = 0; i < TB_SEQ * TB_DIM; i++) {
        #pragma HLS PIPELINE II=1
        q_buf[i] = out[i];
    }
    linear_f(q_buf, out, o_w, o_b, TB_SEQ, TB_DIM, TB_DIM);
}

// ============================================================
// ELEMENT-WISE ADD IN-PLACE  a[i] += b[i]
// ============================================================
inline void add_residual(act_t* a, const act_t* b, int n) {
    #pragma HLS INLINE
    AR: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        a[i] = a[i] + b[i];
    }
}

#endif // T_BLK_S3_UTILS_H
