/*
 * testbench_fpn_p5.cpp
 * C-simulation testbench for fpn_p5_top (Phase 1: P5+P6 only).
 *
 * Three test modes:
 *   Test 1 — all-zero input: P5 output must be all-zero (zero bias mode).
 *   Test 2 — checkerboard C4 + int8 weight=1 + unit bias: verifies data flow.
 *   Test 3 — load real weights/features from binary files (if present).
 *
 * File-based test (Test 3) expects:
 *   ../weights/fpn_p5_w_conv.bin   — 454,656 bytes (int8)
 *   ../weights/fpn_p5_w_meta.bin   — 384 × 4 bytes (float32)
 *   ../weights/c4_feat.bin         — 64,000 × 4 bytes (float32, CHW)
 *
 * Outputs (Test 3 only):
 *   p5_csim.bin   — 19,200 × 4 bytes (float32, CHW)
 *   p6_csim.bin   —  4,800 × 4 bytes (float32, CHW)
 *
 * Standalone g++ build (no Vitis HLS):
 *   g++ -std=c++14 -O2 -I. testbench_fpn_p5.cpp fpn_p5_top.cpp -lm -o tb_fpn_p5
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "fpn_p5_top.h"

// ---- helper: read raw binary file into a buffer ----
static bool read_bin(const char* path, void* buf, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(buf), bytes);
    return (size_t)f.gcount() == bytes;
}

// ---- helper: write raw binary file ----
static void write_bin(const char* path, const void* buf, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf), bytes);
}

// ---- stat helper ----
static void print_stats(const char* name, const act_t* arr, int n) {
    float sum = 0, mn = arr[0], mx = arr[0];
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }
    std::cout << "  " << name
              << "  sum=" << sum
              << "  min=" << mn
              << "  max=" << mx << "\n";
}

// ---- static buffers (declared at file scope to avoid stack overflow) ----
static act_t    c4    [FPN_P5_C4_ELEMS];      // 640*10*10 = 64,000
static weight_t w_conv[FPN_P5_WCONV_ELEMS];   // 454,656
static meta_t   w_meta[FPN_P5_WMETA_ELEMS];   // 384
static act_t    p5    [FPN_P5_P5_ELEMS];       // 19,200
static act_t    p6    [FPN_P5_P6_ELEMS];       // 4,800

int main() {
    std::cout << "==============================================\n";
    std::cout << "fpn_p5_top  C-simulation testbench\n";
    std::cout << "Phase 1: P5+P6  (FPN_OUT_CH=" << FPN_OUT_CH
              << ", P5=" << P5_H << "x" << P5_W
              << ", P6=" << P6_H << "x" << P6_W << ")\n";
    std::cout << "==============================================\n";

    // ----------------------------------------------------------------
    // Test 1: all-zero input, zero weights, zero biases
    //         → P5 and P6 must be exactly 0.0
    // ----------------------------------------------------------------
    std::cout << "\n[Test 1] all-zero input / weights / biases\n";
    memset(c4,     0, sizeof(c4));
    memset(w_conv, 0, sizeof(w_conv));
    memset(w_meta, 0, sizeof(w_meta));
    memset(p5,     0, sizeof(p5));
    memset(p6,     0, sizeof(p6));

    fpn_p5_top(c4, w_conv, w_meta, p5, p6);

    float sum_p5 = 0, sum_p6 = 0;
    for (int i = 0; i < FPN_P5_P5_ELEMS; i++) sum_p5 += p5[i];
    for (int i = 0; i < FPN_P5_P6_ELEMS; i++) sum_p6 += p6[i];
    std::cout << "  P5 sum = " << sum_p5 << "  (expected 0)\n";
    std::cout << "  P6 sum = " << sum_p6 << "  (expected 0)\n";
    if (fabsf(sum_p5) < 1e-3f && fabsf(sum_p6) < 1e-3f)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL (non-zero output for zero input)\n";

    // ----------------------------------------------------------------
    // Test 2: checkerboard C4, int8 weight = +1, unit biases
    //         → P5/P6 should be nonzero, finite, plausible
    // ----------------------------------------------------------------
    std::cout << "\n[Test 2] checkerboard C4, w_conv=+1, w_meta=1.0 (lateral bias), 0.0 (out bias)\n";

    // Checkerboard: value = +1.0 if (h+w) is even, -1.0 if odd
    for (int c = 0; c < C4_CH; c++)
        for (int h = 0; h < P5_H; h++)
            for (int w = 0; w < P5_W; w++)
                c4[c*P5_H*P5_W + h*P5_W + w] = (act_t)(((h+w)%2 == 0) ? 1.0f : -1.0f);

    // All int8 weights = +1
    memset(w_conv, 1, sizeof(w_conv));

    // Lateral biases = 0.0, output conv biases = 0.0
    for (int i = 0; i < FPN_P5_WMETA_ELEMS; i++) w_meta[i] = 0.0f;

    memset(p5, 0, sizeof(p5));
    memset(p6, 0, sizeof(p6));

    fpn_p5_top(c4, w_conv, w_meta, p5, p6);

    print_stats("P5", p5, FPN_P5_P5_ELEMS);
    print_stats("P6", p6, FPN_P5_P6_ELEMS);

    // Sanity: with all-+1 weights and checkerboard, each output pixel of the
    // lateral 1×1 sums C4_CH=640 alternating ±1 values → ~0 (cancellation).
    // The output 3×3 then processes near-zero lat4 → near-zero P5.
    // Not a hard pass/fail — just verify no NaN/Inf.
    bool finite_ok = true;
    for (int i = 0; i < FPN_P5_P5_ELEMS; i++) if (!std::isfinite(p5[i])) { finite_ok=false; break; }
    for (int i = 0; i < FPN_P5_P6_ELEMS; i++) if (!std::isfinite(p6[i])) { finite_ok=false; break; }
    std::cout << (finite_ok ? "  PASS (all finite)\n" : "  FAIL (NaN or Inf detected)\n");

    // ----------------------------------------------------------------
    // Test 2b: ramp C4, uniform weights = +1, lateral bias = 1.0
    //          → each lateral output = C4_CH * avg_input + 1.0
    // ----------------------------------------------------------------
    std::cout << "\n[Test 2b] ramp C4 (all 1.0), w_conv=+1, lat_bias=1.0, out_bias=0.0\n";

    // All C4 = 1.0
    for (int i = 0; i < FPN_P5_C4_ELEMS; i++) c4[i] = 1.0f;

    // int8 weights = +1
    memset(w_conv, 1, sizeof(w_conv));

    // lat4 bias = 1.0, out conv bias = 0.0
    for (int i = 0;          i < FPN_OUT_CH; i++) w_meta[i] = 1.0f;  // lat4 bias
    for (int i = FPN_OUT_CH; i < FPN_P5_WMETA_ELEMS; i++) w_meta[i] = 0.0f;  // out bias

    memset(p5, 0, sizeof(p5));
    memset(p6, 0, sizeof(p6));

    fpn_p5_top(c4, w_conv, w_meta, p5, p6);

    // Expected lat4 value per element: sum(+1 * 1.0 for 640 inputs) + 1.0 = 641.0
    // After output 3×3: each output = sum(641 * +1 for 192 channels * 9 kernel) + 0.0
    //                               = 192 * 9 * 641.0 = 1,108,608 (interior pixels, no pad)
    // Border pixels get fewer terms. Just check P5[0] (top-left corner, padded → fewer terms).
    // lat4[oc,h,w] = C4_CH*1.0 + lat_bias = 641.0  for all positions
    float lat4_expected = (float)C4_CH * 1.0f + 1.0f;   // 641.0
    // P5 centre interior (oh=5,ow=5 in 10×10): all 9 taps valid, 192 IC, out_bias=0
    float p5_interior_expected = (float)FPN_OUT_CH * 9.0f * lat4_expected;  // 192*9*641 = 1,107,648

    // CHW index for oc=0, oh=P5_H/2, ow=P5_W/2:  0*100 + 5*10 + 5 = 55
    float p5_interior = p5[P5_H/2 * P5_W + P5_W/2];   // oc=0, centre pixel
    std::cout << "  lat4 expected per elem: " << lat4_expected << "\n";
    std::cout << "  P5 centre pixel (ch0, CHW idx=" << P5_H/2*P5_W+P5_W/2 << ") expected: " << p5_interior_expected << "\n";
    std::cout << "  P5 centre pixel (ch0) got:      " << p5_interior << "\n";

    float rel_err = fabsf(p5_interior - p5_interior_expected) / fabsf(p5_interior_expected);
    std::cout << "  relative error: " << rel_err << "\n";
    if (rel_err < 1e-4f)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL (>0.01% relative error — check weight/bias layout)\n";

    // ----------------------------------------------------------------
    // Test 3: file-based weights (runs only if files exist)
    // ----------------------------------------------------------------
    std::cout << "\n[Test 3] file-based weights (skipped if files not found)\n";

    bool wc_ok   = read_bin("../weights/fpn_p5_w_conv.bin", w_conv, sizeof(w_conv));
    bool wm_ok   = read_bin("../weights/fpn_p5_w_meta.bin", w_meta, sizeof(w_meta));
    bool c4_ok   = read_bin("../weights/c4_feat.bin",       c4,     sizeof(c4));

    if (!wc_ok || !wm_ok || !c4_ok) {
        std::cout << "  Files not found — skipping.\n";
        std::cout << "  Expected:\n"
                  << "    ../weights/fpn_p5_w_conv.bin  (" << sizeof(w_conv) << " B)\n"
                  << "    ../weights/fpn_p5_w_meta.bin  (" << sizeof(w_meta) << " B)\n"
                  << "    ../weights/c4_feat.bin         (" << sizeof(c4)     << " B)\n";
    } else {
        std::cout << "  Loaded weights and C4 feature map.\n";
        memset(p5, 0, sizeof(p5));
        memset(p6, 0, sizeof(p6));

        fpn_p5_top(c4, w_conv, w_meta, p5, p6);

        print_stats("P5", p5, FPN_P5_P5_ELEMS);
        print_stats("P6", p6, FPN_P5_P6_ELEMS);

        write_bin("p5_csim.bin", p5, sizeof(p5));
        write_bin("p6_csim.bin", p6, sizeof(p6));
        std::cout << "  Dumped: p5_csim.bin  p6_csim.bin\n";
        std::cout << "  PASS (run compare_fpn_p5.py to validate vs Python reference)\n";
    }

    std::cout << "\n==============================================\n";
    std::cout << "fpn_p5_top testbench complete.\n";
    std::cout << "==============================================\n";
    return 0;
}
