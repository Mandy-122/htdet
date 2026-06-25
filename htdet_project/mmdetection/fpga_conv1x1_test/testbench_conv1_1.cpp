#include <iostream>
#include <cstring>
#include <cmath>
#include "conv1_1_top.h"

static float ref_silu(float x) { return x / (1.0f + expf(-x)); }

static void ref_conv1x1_bn_silu(
    const act_t* in, act_t* out,
    const weight_t* w, const meta_t* scale, const meta_t* bias,
    int IC, int OC, int H, int W
) {
    for (int oc = 0; oc < OC; oc++) {
        for (int hw = 0; hw < H * W; hw++) {
            float acc = 0.0f;
            for (int ic = 0; ic < IC; ic++)
                acc += in[hw*IC+ic] * (float)w[oc*IC+ic];
            float y = acc * scale[oc] + bias[oc];
            out[oc*H*W+hw] = ref_silu(y);
        }
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "conv1x1_top  C-simulation testbench\n";
    std::cout << "Config: in_ch=" << TEST_IN_CH << " out_ch=" << TEST_OUT_CH
              << " H=" << TEST_H << " W=" << TEST_W << "\n";
    std::cout << "IC tiles: " << TEST_IN_CH/16 << " x 16\n";
    std::cout << "=================================================\n";

    static act_t    in   [TEST_IN_ELEMS];
    static act_t    out  [TEST_OUT_ELEMS];
    static act_t    ref  [TEST_OUT_ELEMS];
    static weight_t wc   [TEST_W_ELEMS];
    static meta_t   wm   [TEST_META_ELEMS];

    // Test 1: all-zero input -> expect all-zero output
    std::cout << "\n[Test 1] all-zero input -> expect all-zero output\n";
    memset(in,  0, sizeof(in));
    memset(out, 0, sizeof(out));
    memset(wc,  1, sizeof(wc));
    for (int i = 0; i < TEST_OUT_CH; i++) { wm[i] = 1.0f; wm[TEST_OUT_CH+i] = 0.0f; }

    conv1x1_top(in, out, wc, wm);

    float sum1 = 0.0f;
    for (int i = 0; i < TEST_OUT_ELEMS; i++) sum1 += fabsf(out[i]);
    std::cout << "  abs-sum = " << sum1 << "  (expected 0)\n";
    std::cout << (sum1 < 1e-3f ? "  PASS\n" : "  FAIL\n");

    // Test 2: random input, compare HLS vs reference
    std::cout << "\n[Test 2] random input vs CPU reference\n";
    for (int i = 0; i < TEST_IN_ELEMS; i++) in[i] = (act_t)((i % 7) - 3) * 0.5f;
    for (int i = 0; i < TEST_W_ELEMS;  i++) wc[i] = (weight_t)((i % 5) - 2);
    for (int i = 0; i < TEST_OUT_CH;   i++) { wm[i] = 0.5f; wm[TEST_OUT_CH+i] = 0.1f; }
    memset(out, 0, sizeof(out));
    memset(ref, 0, sizeof(ref));

    conv1x1_top(in, out, wc, wm);
    ref_conv1x1_bn_silu(in, ref, wc, wm, wm+TEST_OUT_CH, TEST_IN_CH, TEST_OUT_CH, TEST_H, TEST_W);

    float max_err = 0.0f;
    for (int i = 0; i < TEST_OUT_ELEMS; i++)
        max_err = fmaxf(max_err, fabsf(out[i] - ref[i]));
    std::cout << "  max error vs reference = " << max_err << "\n";
    std::cout << (max_err < 1e-4f ? "  PASS\n" : "  FAIL (check tiling logic)\n");

    std::cout << "\n=================================================\n";
    std::cout << "Testbench complete.\n";
    return 0;
}
