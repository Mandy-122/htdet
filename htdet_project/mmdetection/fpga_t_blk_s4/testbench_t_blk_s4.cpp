/*
 * testbench_t_blk_s4.cpp
 * C-simulation testbench for transformer_blk_s4_top.
 *
 * Tests:
 *   1. Zero input, zero weights (unit LN scales) → output == input == 0
 *   2. Structured non-zero input, identity-like weights → HLS matches CPU reference
 *   3. Non-trivial weight values, random-ish input → HLS matches CPU reference
 *
 * The CPU reference (ref_*) implements the same algorithm in plain C
 * so we can catch any tiling / offset bug introduced by HLS pragmas.
 *
 * Compile standalone (no Vitis):
 *   g++ -std=c++14 -I. -O2 testbench_t_blk_s4.cpp t_blk_s4_top.cpp -lm -o tb_t_blk_s4
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include "t_blk_s4_top.h"

// ── CPU reference helpers ─────────────────────────────────────────────────────

static float ref_silu_c(float x) { return x / (1.0f + expf(-x)); }

static void ref_layer_norm(
    const float* in, float* out,
    const float* w,  const float* b,
    int seq, int dim
) {
    for (int s = 0; s < seq; s++) {
        const float* row = in  + s * dim;
        float*       dst = out + s * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += row[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = row[i]-mean; var += d*d; }
        float inv_std = 1.0f / sqrtf(var / (float)dim + 1e-5f);
        for (int i = 0; i < dim; i++)
            dst[i] = (row[i]-mean) * inv_std * w[i] + b[i];
    }
}

static void ref_linear(
    const float* in, float* out,
    const float* w,  const float* bias,
    int seq, int in_dim, int out_dim
) {
    for (int s = 0; s < seq; s++)
        for (int od = 0; od < out_dim; od++) {
            float acc = bias[od];
            for (int id = 0; id < in_dim; id++)
                acc += in[s*in_dim+id] * w[od*in_dim+id];
            out[s*out_dim+od] = acc;
        }
}

static void ref_softmax(float* v, int N) {
    float mx = v[0];
    for (int i = 1; i < N; i++) if (v[i] > mx) mx = v[i];
    float s = 0.0f;
    for (int i = 0; i < N; i++) { v[i] = expf(v[i]-mx); s += v[i]; }
    for (int i = 0; i < N; i++) v[i] /= s;
}

// Full CPU reference for one transformer block
static void ref_transformer_block(
    float* x,           // [SEQ*DIM] — updated in-place
    const float* bw     // block weights at BLK_*_OFF offsets
) {
    static float ln1[TB_IN_ELEMS], q[TB_IN_ELEMS], k[TB_IN_ELEMS], v[TB_IN_ELEMS];
    static float scores[TB_SEQ * TB_SEQ];
    static float attn_head[TB_IN_ELEMS];
    static float attn_out[TB_IN_ELEMS];
    static float mlp_h[TB_SEQ * TB_MLP_HID];
    static float mlp_out[TB_IN_ELEMS];

    // LN1
    ref_layer_norm(x, ln1, bw+BLK_LN1_W_OFF, bw+BLK_LN1_B_OFF, TB_SEQ, TB_DIM);

    // QKV projections
    ref_linear(ln1, q, bw+BLK_Q_W_OFF, bw+BLK_Q_B_OFF, TB_SEQ, TB_DIM, TB_DIM);
    ref_linear(ln1, k, bw+BLK_K_W_OFF, bw+BLK_K_B_OFF, TB_SEQ, TB_DIM, TB_DIM);
    ref_linear(ln1, v, bw+BLK_V_W_OFF, bw+BLK_V_B_OFF, TB_SEQ, TB_DIM, TB_DIM);

    // Multi-head attention
    float hscale = 1.0f / sqrtf((float)TB_HEAD_DIM);
    memset(attn_head, 0, sizeof(attn_head));

    for (int h = 0; h < TB_HEADS; h++) {
        // Scores
        for (int qi = 0; qi < TB_SEQ; qi++) {
            for (int ki = 0; ki < TB_SEQ; ki++) {
                float dot = 0.0f;
                for (int d = 0; d < TB_HEAD_DIM; d++)
                    dot += q[qi*TB_DIM+h*TB_HEAD_DIM+d] * k[ki*TB_DIM+h*TB_HEAD_DIM+d];
                scores[qi*TB_SEQ+ki] = dot * hscale;
            }
            ref_softmax(scores+qi*TB_SEQ, TB_SEQ);
        }
        // Weighted V sum
        for (int qi = 0; qi < TB_SEQ; qi++)
            for (int d = 0; d < TB_HEAD_DIM; d++) {
                float acc = 0.0f;
                for (int ki = 0; ki < TB_SEQ; ki++)
                    acc += scores[qi*TB_SEQ+ki] * v[ki*TB_DIM+h*TB_HEAD_DIM+d];
                attn_head[qi*TB_DIM+h*TB_HEAD_DIM+d] = acc;
            }
    }
    // Output projection
    ref_linear(attn_head, attn_out, bw+BLK_O_W_OFF, bw+BLK_O_B_OFF, TB_SEQ, TB_DIM, TB_DIM);

    // Residual 1
    for (int i = 0; i < TB_IN_ELEMS; i++) x[i] += attn_out[i];

    // LN2
    ref_layer_norm(x, ln1, bw+BLK_LN2_W_OFF, bw+BLK_LN2_B_OFF, TB_SEQ, TB_DIM);

    // fc1 + SiLU
    ref_linear(ln1, mlp_h, bw+BLK_FC1_W_OFF, bw+BLK_FC1_B_OFF, TB_SEQ, TB_DIM, TB_MLP_HID);
    for (int i = 0; i < TB_SEQ*TB_MLP_HID; i++) mlp_h[i] = ref_silu_c(mlp_h[i]);

    // fc2
    ref_linear(mlp_h, mlp_out, bw+BLK_FC2_W_OFF, bw+BLK_FC2_B_OFF, TB_SEQ, TB_MLP_HID, TB_DIM);

    // Residual 2
    for (int i = 0; i < TB_IN_ELEMS; i++) x[i] += mlp_out[i];
}

// Full CPU reference wrapping depth loop
static void ref_transformer_blk_s4(
    const float* in, float* out, const float* w
) {
    static float work[TB_IN_ELEMS];
    memcpy(work, in, TB_IN_ELEMS * sizeof(float));
    for (int d = 0; d < TB_DEPTH; d++)
        ref_transformer_block(work, w + d * TB_BLK_W_ELEMS);
    memcpy(out, work, TB_OUT_ELEMS * sizeof(float));
}

// ── Weight fill helpers ───────────────────────────────────────────────────────

// All-zero weights, unit LN scales (bias=0)
static void fill_zero_weights(meta_t* w) {
    memset(w, 0, TB_TOTAL_W_ELEMS * sizeof(meta_t));
    for (int d = 0; d < TB_DIM; d++) {
        w[BLK_LN1_W_OFF + d] = 1.0f;
        w[BLK_LN2_W_OFF + d] = 1.0f;
    }
}

// Identity-like weights:
//   LN scale=1 bias=0,  Q/K/V/O = identity,  fc1/fc2 = identity (first DIM rows/cols)
static void fill_identity_weights(meta_t* w) {
    memset(w, 0, TB_TOTAL_W_ELEMS * sizeof(meta_t));
    for (int d = 0; d < TB_DIM; d++) {
        // LN scales
        w[BLK_LN1_W_OFF + d] = 1.0f;
        w[BLK_LN2_W_OFF + d] = 1.0f;
        // Q, K, V, O: identity [DIM x DIM]
        w[BLK_Q_W_OFF + d * TB_DIM + d] = 1.0f;
        w[BLK_K_W_OFF + d * TB_DIM + d] = 1.0f;
        w[BLK_V_W_OFF + d * TB_DIM + d] = 1.0f;
        w[BLK_O_W_OFF + d * TB_DIM + d] = 1.0f;
        // fc1: first TB_DIM rows = identity (remaining rows stay 0)
        w[BLK_FC1_W_OFF + d * TB_DIM + d] = 1.0f;
        // fc2: first TB_DIM cols = identity (remaining cols stay 0)
        w[BLK_FC2_W_OFF + d * TB_MLP_HID + d] = 1.0f;
    }
}

// Small randomish weights: values derived from index mod pattern
static void fill_rand_weights(meta_t* w) {
    for (int i = 0; i < TB_TOTAL_W_ELEMS; i++)
        w[i] = (meta_t)(((i * 13 + 7) % 17 - 8) * 0.05f);
    // Overwrite LN scales with positive values to avoid collapsed norm
    for (int d = 0; d < TB_DIM; d++) {
        w[BLK_LN1_W_OFF + d] = (meta_t)(0.5f + 0.1f * (d % 3));
        w[BLK_LN2_W_OFF + d] = (meta_t)(0.5f + 0.1f * (d % 5));
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

static float max_abs_err(const act_t* hls, const float* ref, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++)
        mx = fmaxf(mx, fabsf((float)hls[i] - ref[i]));
    return mx;
}

static float abs_sum(const act_t* v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf((float)v[i]);
    return s;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=========================================================\n";
    std::cout << "transformer_blk_s4_top  C-simulation testbench\n";
    std::cout << "Config: DIM=" << TB_DIM
              << "  SEQ="   << TB_SEQ
              << "  HEADS=" << TB_HEADS
              << "  HD="    << TB_HEAD_DIM
              << "  MLP="   << TB_MLP_HID
              << "  DEPTH=" << TB_DEPTH << "\n";
    std::cout << "Per-block weights: " << TB_BLK_W_ELEMS
              << "  Total: " << TB_TOTAL_W_ELEMS << "\n";
    std::cout << "=========================================================\n";

    static act_t  in  [TB_IN_ELEMS];
    static act_t  out [TB_OUT_ELEMS];
    static meta_t w   [TB_TOTAL_W_ELEMS];
    static float  ref [TB_OUT_ELEMS];
    int pass_count = 0;
    // Scale tolerance with problem size: larger DIM/SEQ accumulate more FP rounding.
    // 1e-4 suffices for Step 1-3 (DIM<=64), 1e-3 for Step 4+ (DIM=144).
    const float ATOL = (TB_DIM >= 128) ? 1e-3f : 1e-4f;

    // ── Test 1: zero input, zero weights (unit LN scales) → output = 0 ────────
    std::cout << "\n[Test 1] zero input, zero proj weights, unit LN scales\n";
    memset(in, 0, sizeof(in));
    fill_zero_weights(w);

    transformer_blk_s4_top(in, out, w);

    float s1 = abs_sum(out, TB_OUT_ELEMS);
    std::cout << "  HLS abs-sum = " << s1 << "  (expected 0.0)\n";
    if (s1 < 1e-4f) {
        std::cout << "  PASS\n";
        pass_count++;
    } else {
        std::cout << "  FAIL — nonzero output for zero input\n";
        for (int i = 0; i < TB_OUT_ELEMS; i++)
            if (fabsf(out[i]) > 1e-6f)
                std::cout << "    out[" << i << "] = " << out[i] << "\n";
    }

    // ── Test 2: identity weights, structured input vs CPU reference ───────────
    std::cout << "\n[Test 2] identity-like weights, structured input vs CPU ref\n";
    // Input: token s, dim d = (s+1) * (d+1) * 0.05
    for (int s = 0; s < TB_SEQ; s++)
        for (int d = 0; d < TB_DIM; d++)
            in[s * TB_DIM + d] = (act_t)((s + 1) * (d + 1) * 0.05f);
    fill_identity_weights(w);
    memset(out, 0, sizeof(out));

    transformer_blk_s4_top(in, out, w);
    ref_transformer_blk_s4(in, ref, w);

    float max2 = max_abs_err(out, ref, TB_OUT_ELEMS);
    float asum2 = abs_sum(out, TB_OUT_ELEMS);
    std::cout << "  HLS abs-sum = " << asum2 << "  (nonzero expected)\n";
    std::cout << "  max |HLS - ref| = " << max2 << "  (threshold " << ATOL << ")\n";
    if (max2 < ATOL && asum2 > 1e-3f) {
        std::cout << "  PASS\n";
        pass_count++;
    } else {
        std::cout << "  FAIL\n";
        std::cout << "  First 16 HLS vs ref:\n";
        for (int i = 0; i < 16; i++)
            std::cout << "    [" << i << "]  hls=" << out[i] << "  ref=" << ref[i] << "\n";
    }

    // ── Test 3: randomish weights and input vs CPU reference ──────────────────
    std::cout << "\n[Test 3] small random weights, random-ish input vs CPU ref\n";
    for (int i = 0; i < TB_IN_ELEMS; i++)
        in[i] = (act_t)(((i * 7 + 3) % 11 - 5) * 0.1f);
    fill_rand_weights(w);
    memset(out, 0, sizeof(out));

    transformer_blk_s4_top(in, out, w);
    ref_transformer_blk_s4(in, ref, w);

    float max3 = max_abs_err(out, ref, TB_OUT_ELEMS);
    float asum3 = abs_sum(out, TB_OUT_ELEMS);
    std::cout << "  HLS abs-sum = " << asum3 << "  (nonzero expected)\n";
    std::cout << "  max |HLS - ref| = " << max3 << "  (threshold " << ATOL << ")\n";
    if (max3 < ATOL && asum3 > 1e-3f) {
        std::cout << "  PASS\n";
        pass_count++;
    } else {
        std::cout << "  FAIL\n";
        std::cout << "  First 16 HLS vs ref:\n";
        for (int i = 0; i < 16; i++)
            std::cout << "    [" << i << "]  hls=" << out[i] << "  ref=" << ref[i] << "\n";
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n=========================================================\n";
    std::cout << "Results: " << pass_count << " / 3 tests passed\n";
    if (pass_count == 3)
        std::cout << "ALL PASS — ready for csynth\n";
    else
        std::cout << "SOME FAILED — fix before csynth\n";
    std::cout << "=========================================================\n";

    return (pass_count == 3) ? 0 : 1;
}
