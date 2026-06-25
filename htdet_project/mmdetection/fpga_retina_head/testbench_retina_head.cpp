/*
 * testbench_retina_head.cpp
 * C-simulation testbench for retina_head_top — isolated retina head module.
 *
 * ── Two tests ──────────────────────────────────────────────────────────────
 *
 * Test 1 — Synthetic all-zero weights:
 *   Zero int8 weights, unit dequant-scale, zero bias, checkerboard input.
 *   Expected: cls_logits = 0, reg_deltas = 0.
 *
 * Test 2 — Real PTQ weights + synthetic deterministic feature:
 *   Loads actual int8 weights from ptq_results_192_ep47/.
 *   Feature map is a reproducible sin/cos pattern (same formula in gen_csim_ref.py).
 *   Run gen_csim_ref.py --syn to get the matching Python-side statistics.
 *   This avoids the 256-ch vs 192-ch mismatch in the old csim_validation bins.
 *
 * ── Binary files needed for Test 2 ────────────────────────────────────────
 *   (relative to fpga_retina_head/ where Vitis HLS runs)
 *   ../ptq_results_192_ep47/ptq_int8_weights/cls_conv_int8.bin   (1 327 104 int8)
 *   ../ptq_results_192_ep47/cls_conv_meta_float.bin              (    1 536 float)
 *   ../ptq_results_192_ep47/ptq_int8_weights/reg_conv_int8.bin
 *   ../ptq_results_192_ep47/reg_conv_meta_float.bin
 *   ../ptq_results_192_ep47/ptq_int8_weights/cls_pred_int8.bin   (   62 208 int8)
 *   ../ptq_results_192_ep47/cls_pred_scale_float.bin             (       36 float)
 *   ../ptq_results_192_ep47/cls_pred_bias_float.bin
 *   ../ptq_results_192_ep47/ptq_int8_weights/reg_pred_int8.bin
 *   ../ptq_results_192_ep47/reg_pred_scale_float.bin
 *   ../ptq_results_192_ep47/reg_pred_bias_float.bin
 *
 * ── Standalone compile ─────────────────────────────────────────────────────
 *   g++ -std=c++14 -I. -Ihls_stubs -DTEST_H=10 -DTEST_W=10 \
 *       testbench_retina_head.cpp retina_head_top.cpp -lm -o tb_retina
 *   ./tb_retina
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "retina_head_top.h"

#define PTQ_DIR  "../ptq_results_192_ep47"

// ── helpers ────────────────────────────────────────────────────────────────
static bool load_int8_bin(const char* path, weight_t* buf, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "  [WARN] Cannot open: " << path << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(buf), n * sizeof(int8_t));
    int got = (int)f.gcount();
    std::cout << "  int8  " << got << " bytes  (" << n << " expected)  " << path << "\n";
    return (got == n);
}

static bool load_float_bin(const char* path, float* buf, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "  [WARN] Cannot open: " << path << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(buf), n * sizeof(float));
    int got = (int)f.gcount() / (int)sizeof(float);
    std::cout << "  float " << got << " elems  (" << n << " expected)  " << path << "\n";
    return (got == n);
}

static void print_stats(const char* label, const act_t* arr, int n) {
    float mn = arr[0], mx = arr[0], sm = 0.0f;
    int   nonzero = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        sm += arr[i];
        if (arr[i] != 0.0f) nonzero++;
    }
    std::cout << "  " << label
              << "\n    n=" << n
              << "  min=" << mn << "  max=" << mx
              << "  mean=" << sm / n << "  nonzero=" << nonzero
              << "\n    first5=[" << arr[0] << ", " << arr[1] << ", "
              << arr[2] << ", " << arr[3] << ", " << arr[4] << "]\n";
}

// Fills stacked-conv meta with scale=1.0, bias=0.0 per layer
static void fill_stacked_meta_unit(meta_t* w_meta) {
    memset(w_meta, 0, STACKED_CONV_M_ELEMS * sizeof(meta_t));
    for (int l = 0; l < HEAD_STACKED_CONVS; l++) {
        meta_t* sc = w_meta + l * STACKED_CONV_M_LAYER;
        for (int c = 0; c < HEAD_FEAT_CH; c++) sc[c] = 1.0f;
    }
}

// Deterministic sin/cos feature — MUST match gen_csim_ref.py gen_syn_feat()
static void fill_syn_feat(act_t* feat, int ch, int H, int W) {
    for (int c = 0; c < ch; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                feat[c*H*W + h*W + w] =
                    (act_t)(sinf(c * 0.05f) * cosf(h * 0.2f + w * 0.3f) * 2.0f);
}

// ── main ───────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=================================================\n";
    std::cout << "retina_head_top  C-simulation testbench\n";
    std::cout << "  TEST_H=" << TEST_H << "  TEST_W=" << TEST_W << "\n";
    std::cout << "  HEAD_FEAT_CH=" << HEAD_FEAT_CH
              << "  CLS_OUT_CH=" << CLS_OUT_CH
              << "  REG_OUT_CH=" << REG_OUT_CH << "\n";
    std::cout << "  FEAT_ELEMS=" << FEAT_ELEMS
              << "  CLS_LOGITS_ELEMS=" << CLS_LOGITS_ELEMS << "\n";
    std::cout << "=================================================\n\n";

    static act_t    feat_in       [FEAT_ELEMS];
    static act_t    cls_logits    [CLS_LOGITS_ELEMS];
    static act_t    reg_deltas    [REG_DELTAS_ELEMS];
    static weight_t cls_conv_w    [STACKED_CONV_W_ELEMS];
    static meta_t   cls_conv_m    [STACKED_CONV_M_ELEMS];
    static weight_t reg_conv_w    [STACKED_CONV_W_ELEMS];
    static meta_t   reg_conv_m    [STACKED_CONV_M_ELEMS];
    static weight_t cls_pred_w    [PRED_W_ELEMS];
    static meta_t   cls_pred_scale[PRED_META_ELEMS];
    static meta_t   cls_pred_bias [PRED_META_ELEMS];
    static weight_t reg_pred_w    [PRED_W_ELEMS];
    static meta_t   reg_pred_scale[PRED_META_ELEMS];
    static meta_t   reg_pred_bias [PRED_META_ELEMS];

    // ══════════════════════════════════════════════════════════════════════
    // TEST 1: Zero weights — expected zero output
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 1] Zero int8 weights — expect cls_logits=0, reg_deltas=0\n";

    memset(cls_conv_w, 0, sizeof(cls_conv_w));
    memset(reg_conv_w, 0, sizeof(reg_conv_w));
    memset(cls_pred_w, 0, sizeof(cls_pred_w));
    memset(reg_pred_w, 0, sizeof(reg_pred_w));
    fill_stacked_meta_unit(cls_conv_m);
    fill_stacked_meta_unit(reg_conv_m);
    for (int i = 0; i < PRED_META_ELEMS; i++) cls_pred_scale[i] = 1.0f;
    for (int i = 0; i < PRED_META_ELEMS; i++) reg_pred_scale[i] = 1.0f;
    memset(cls_pred_bias, 0, sizeof(cls_pred_bias));
    memset(reg_pred_bias, 0, sizeof(reg_pred_bias));

    // Checkerboard input
    for (int c = 0; c < HEAD_FEAT_CH; c++)
        for (int h = 0; h < TEST_H; h++)
            for (int w = 0; w < TEST_W; w++)
                feat_in[c*TEST_H*TEST_W + h*TEST_W + w] = ((h+w) % 2 == 0) ? 0.5f : -0.5f;

    memset(cls_logits, 0, sizeof(cls_logits));
    memset(reg_deltas, 0, sizeof(reg_deltas));

    retina_head_top(feat_in, cls_logits, reg_deltas,
                    cls_conv_w, cls_conv_m, reg_conv_w, reg_conv_m,
                    cls_pred_w, cls_pred_scale, cls_pred_bias,
                    reg_pred_w, reg_pred_scale, reg_pred_bias);

    float cls_abs = 0.0f, reg_abs = 0.0f;
    for (int i = 0; i < CLS_LOGITS_ELEMS; i++) cls_abs += fabsf(cls_logits[i]);
    for (int i = 0; i < REG_DELTAS_ELEMS; i++) reg_abs += fabsf(reg_deltas[i]);
    std::cout << "  cls_logits abs-sum = " << cls_abs << "  (expected 0.0)\n";
    std::cout << "  reg_deltas abs-sum = " << reg_abs << "  (expected 0.0)\n";
    bool t1_pass = (fabsf(cls_abs) < 1e-3f && fabsf(reg_abs) < 1e-3f);
    std::cout << "  " << (t1_pass ? "[PASS]" : "[FAIL]") << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // TEST 2: Real PTQ weights + synthetic deterministic feature
    // Feature formula (also used in gen_csim_ref.py --syn):
    //   feat[c,h,w] = sin(c*0.05) * cos(h*0.2 + w*0.3) * 2.0
    // Compare output stats against gen_csim_ref.py --syn output.
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 2] Real PTQ weights + deterministic synthetic feature\n";
    std::cout << "  Feature: feat[c,h,w] = sin(c*0.05)*cos(h*0.2+w*0.3)*2.0\n";

    bool ok = true;
    ok &= load_int8_bin (PTQ_DIR "/ptq_int8_weights/cls_conv_int8.bin", cls_conv_w,     STACKED_CONV_W_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/cls_conv_meta_float.bin",             cls_conv_m,     STACKED_CONV_M_ELEMS);
    ok &= load_int8_bin (PTQ_DIR "/ptq_int8_weights/reg_conv_int8.bin", reg_conv_w,     STACKED_CONV_W_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/reg_conv_meta_float.bin",             reg_conv_m,     STACKED_CONV_M_ELEMS);
    ok &= load_int8_bin (PTQ_DIR "/ptq_int8_weights/cls_pred_int8.bin", cls_pred_w,     PRED_W_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/cls_pred_scale_float.bin",            cls_pred_scale, PRED_META_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/cls_pred_bias_float.bin",             cls_pred_bias,  PRED_META_ELEMS);
    ok &= load_int8_bin (PTQ_DIR "/ptq_int8_weights/reg_pred_int8.bin", reg_pred_w,     PRED_W_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/reg_pred_scale_float.bin",            reg_pred_scale, PRED_META_ELEMS);
    ok &= load_float_bin(PTQ_DIR "/reg_pred_bias_float.bin",             reg_pred_bias,  PRED_META_ELEMS);

    if (!ok) {
        std::cout << "  [SKIP] Weight files not found in " PTQ_DIR "\n\n";
    } else {
        fill_syn_feat(feat_in, HEAD_FEAT_CH, TEST_H, TEST_W);
        memset(cls_logits, 0, sizeof(cls_logits));
        memset(reg_deltas, 0, sizeof(reg_deltas));

        retina_head_top(feat_in, cls_logits, reg_deltas,
                        cls_conv_w, cls_conv_m, reg_conv_w, reg_conv_m,
                        cls_pred_w, cls_pred_scale, cls_pred_bias,
                        reg_pred_w, reg_pred_scale, reg_pred_bias);

        std::cout << "\n  === Output statistics (compare with gen_csim_ref.py --syn) ===\n";
        print_stats("cls_logits", cls_logits, CLS_LOGITS_ELEMS);
        print_stats("reg_deltas", reg_deltas, REG_DELTAS_ELEMS);

        // Count anchors above score threshold
        int above_thr = 0;
        const float thr_logit = -1.386f;   // sigmoid^{-1}(0.20)
        for (int i = 0; i < CLS_LOGITS_ELEMS; i++)
            if (cls_logits[i] >= thr_logit) above_thr++;
        std::cout << "  anchors above score_thr=0.20: "
                  << above_thr << " / " << CLS_LOGITS_ELEMS << "\n";
        std::cout << "  [INFO] Run: python3 gen_csim_ref.py --syn --h " << TEST_H
                  << " --w " << TEST_W << "\n\n";
    }

    std::cout << "=================================================\n";
    std::cout << "retina_head_top  CSIM complete\n";
    std::cout << "=================================================\n";
    return t1_pass ? 0 : 1;
}
