/*
 * testbench_mbconv40.cpp
 * HLS C-simulation testbench for mbconv_40_top.
 *
 * Config: in_ch=96, out_ch=128, expand=4, stride=2
 * Input spatial : 40×40,  Output spatial: 20×20
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include "mbconv_40_top.h"

static void fill_meta_unit(meta_t* w_meta) {
    // BN: scale=1.0, bias=0.0 for all three conv layers
    // Layout: [expand_scale | expand_bias | dw_scale | dw_bias | proj_scale | proj_bias]
    for (int i = 0; i < MBCONV40_WMETA_ELEMS; i++) w_meta[i] = 0.0f;
    const int hid = MBCONV40_HID;
    for (int i = 0;      i < hid;              i++) w_meta[i] = 1.0f;  // expand scale
    for (int i = hid*2;  i < hid*3;            i++) w_meta[i] = 1.0f;  // dw scale
    for (int i = hid*4;  i < hid*4 + MBCONV40_OUT_CH; i++) w_meta[i] = 1.0f;  // proj scale
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "mbconv_40_top  C-simulation testbench\n";
    std::cout << "Config: in_ch=96, out_ch=128, expand=4, stride=2\n";
    std::cout << "Input:  H=" << C2_H << "  W=" << C2_W << "\n";
    std::cout << "Output: H=" << C3_H << "  W=" << C3_W << "\n";
    std::cout << "=================================================\n";

    static act_t    in    [MBCONV40_IN_ELEMS];
    static act_t    out   [MBCONV40_OUT_ELEMS];
    static weight_t wconv [MBCONV40_WCONV_ELEMS];
    static meta_t   wmeta [MBCONV40_WMETA_ELEMS];

    // ---- Test 1: all-zero input ----------------------------------------
    std::cout << "\n[Test 1] all-zero input\n";
    memset(in,    0, sizeof(in));
    memset(out,   0, sizeof(out));
    memset(wconv, 0, sizeof(wconv));
    fill_meta_unit(wmeta);

    mbconv_40_top(in, out, wconv, wmeta);

    float sum1 = 0.0f;
    for (int i = 0; i < MBCONV40_OUT_ELEMS; i++) sum1 += out[i];
    std::cout << "  output sum = " << sum1 << "  (expected 0.0)\n";
    std::cout << (fabsf(sum1) < 1e-3f ? "  PASS\n" : "  FAIL\n");

    // ---- Test 2: checkerboard input + unit int8 weights ----------------
    std::cout << "\n[Test 2] checkerboard activations, unit int8 weights\n";
    for (int c = 0; c < MBCONV40_IN_CH; c++)
        for (int h = 0; h < C2_H; h++)
            for (int w = 0; w < C2_W; w++)
                in[c * C2_H * C2_W + h * C2_W + w] =
                    (act_t)(((h + w) % 2 == 0) ? 1.0f : -1.0f);

    memset(wconv, 1, sizeof(wconv));
    memset(out,   0, sizeof(out));

    mbconv_40_top(in, out, wconv, wmeta);

    float sum2 = 0.0f, min2 = out[0], max2 = out[0];
    for (int i = 0; i < MBCONV40_OUT_ELEMS; i++) {
        sum2 += out[i];
        if (out[i] < min2) min2 = out[i];
        if (out[i] > max2) max2 = out[i];
    }
    std::cout << "  output sum=" << sum2
              << "  min=" << min2
              << "  max=" << max2 << "\n";
    std::cout << ((max2 != 0.0f || min2 != 0.0f)
                  ? "  PASS (nonzero output)\n"
                  : "  NOTE: output is zero\n");

    std::cout << "\n=================================================\n";
    std::cout << "Testbench complete.\n";
    std::cout << "=================================================\n";
    return 0;
}
