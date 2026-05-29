/*
 * testbench_mbconv80.cpp
 * Minimal HLS C-simulation testbench for mbconv_80_top.
 *
 * Tests:
 *   1. All-zero input  → output should be all-zero (bias=0 in meta)
 *   2. Checkerboard input → output should be nonzero (exercises SiLU path)
 *
 * Run via:
 *   csim_design (in Vitis HLS) or standalone g++ compile:
 *     g++ -std=c++14 -I. testbench_mbconv80.cpp mbconv_80_top.cpp -lm -o tb_mbconv80
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include "mbconv_80_top.h"

static void fill_meta_unit(meta_t* w_meta) {
    // Set effective BN: scale=1.0, bias=0.0 for all layers
    // Layout (see mbconv_80_top.h):
    //   [0..255]    expand BN scale
    //   [256..511]  expand BN bias
    //   [512..767]  dw     BN scale
    //   [768..1023] dw     BN bias
    //   [1024..1087] proj  BN scale
    //   [1088..1151] proj  BN bias
    for (int i = 0; i < MBCONV80_WMETA_ELEMS; i++) w_meta[i] = 0.0f;
    // Set scales to 1 so activations pass through (not all zeroed by scale=0)
    const int hid = MBCONV80_HID;
    for (int i = 0;    i < hid;              i++) w_meta[i] = 1.0f;           // expand scale
    for (int i = hid*2; i < hid*3;           i++) w_meta[i] = 1.0f;           // dw scale
    for (int i = hid*4; i < hid*4 + MBCONV80_OUT_CH; i++) w_meta[i] = 1.0f;  // proj scale
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "mbconv_80_top  C-simulation testbench\n";
    std::cout << "Config: in_ch=64, out_ch=64, expand=4, stride=1\n";
    std::cout << "Spatial: H=" << C1_H << "  W=" << C1_W << "\n";
    std::cout << "=================================================\n";

    static act_t    in    [MBCONV80_IN_ELEMS];
    static act_t    out   [MBCONV80_OUT_ELEMS];
    static weight_t wconv [MBCONV80_WCONV_ELEMS];
    static meta_t   wmeta [MBCONV80_WMETA_ELEMS];

    // ---- Test 1: all-zero input ----------------------------------------
    std::cout << "\n[Test 1] all-zero input\n";
    memset(in,    0, sizeof(in));
    memset(out,   0, sizeof(out));
    memset(wconv, 0, sizeof(wconv));
    fill_meta_unit(wmeta);

    mbconv_80_top(in, out, wconv, wmeta);

    float sum1 = 0.0f;
    for (int i = 0; i < MBCONV80_OUT_ELEMS; i++) sum1 += out[i];
    std::cout << "  output sum = " << sum1 << "  (expected 0.0)\n";
    if (fabsf(sum1) < 1e-3f)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL (non-zero output for zero input)\n";

    // ---- Test 2: checkerboard input + unit weights ----------------------
    std::cout << "\n[Test 2] checkerboard activations, unit int8 weights\n";
    for (int c = 0; c < MBCONV80_IN_CH; c++)
        for (int h = 0; h < C1_H; h++)
            for (int w = 0; w < C1_W; w++)
                in[c * C1_H * C1_W + h * C1_W + w] =
                    (act_t)(((h + w) % 2 == 0) ? 1.0f : -1.0f);

    // int8 weights: all +1 (0x01)
    memset(wconv, 1, sizeof(wconv));
    memset(out,   0, sizeof(out));

    mbconv_80_top(in, out, wconv, wmeta);

    float sum2 = 0.0f, min2 = out[0], max2 = out[0];
    for (int i = 0; i < MBCONV80_OUT_ELEMS; i++) {
        sum2 += out[i];
        if (out[i] < min2) min2 = out[i];
        if (out[i] > max2) max2 = out[i];
    }
    std::cout << "  output sum=" << sum2
              << "  min=" << min2
              << "  max=" << max2 << "\n";
    // Residual adds input to output; with nonzero weights we expect nonzero
    if (max2 != 0.0f || min2 != 0.0f)
        std::cout << "  PASS (nonzero output for nonzero input)\n";
    else
        std::cout << "  NOTE: output is zero (check scale/bias config)\n";

    std::cout << "\n=================================================\n";
    std::cout << "Testbench complete.\n";
    std::cout << "=================================================\n";
    return 0;
}
