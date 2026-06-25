/*
 * c4_expand_top.cpp
 * Final backbone expansion: conv1×1(160→640, 10×10) + BN + SiLU
 * CHW layout (matches mobilevit_block_s4 output via conv3x3).
 *
 * Expected latency: trivially small (100 pixels).
 *   Outer trip = 640 × 100 = 64,000 iters at II=1 ≈ 64,000 cycles ≈ 0.32 ms
 *   (IC=160, UNROLL factor=16 → 10 IC groups per pipeline iteration)
 */

#include "c4_expand_top.h"
#include "fpga_utils.h"

void c4_expand_top(
    const act_t    in    [C4EXP_IN_ELEMS],
          act_t    out   [C4EXP_OUT_ELEMS],
    const weight_t w_conv[C4EXP_WCONV_ELEMS],
    const meta_t   w_meta[C4EXP_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    conv1x1_bn_silu(in, out,
                    w_conv,
                    w_meta,
                    w_meta + C4EXP_OUT_CH,
                    C4EXP_IN_CH, C4EXP_OUT_CH, C4_H, C4_W);
}
