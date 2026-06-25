/*
 * stem_top.cpp
 * HLS top: MobileViT-S stem — Conv3×3(3→16, stride=2, 320×320) + BN + SiLU
 * CHW layout throughout (matches backbone output feeding into mbconv_s0).
 *
 * Key loop structure (from conv3x3_bn_silu in fpga_utils.h):
 *   OC (16) × OH (160) × OW (160) pipelined, IC(3) × KH(3) × KW(3) unrolled
 *   Trip count = 16 × 160 × 160 = 409,600
 *   IC=3 is small — 3×9 = 27 MACs per output pixel
 *
 * Partition:
 *   in     cyclic factor=16 — no urem (3 < 16, factor covers stride-2 reads)
 *   w_conv cyclic factor=16
 *
 * Expected latency (II=1, trip=409,600):
 *   ~409,600 cycles ≈ 2 ms @ 200 MHz  (IC=3 too small to dominate)
 */

#include "stem_top.h"
#include "fpga_utils.h"

void stem_top(
    const act_t    in    [STEM_IN_ELEMS],
          act_t    out   [STEM_OUT_ELEMS],
    const weight_t w_conv[STEM_WCONV_ELEMS],
    const meta_t   w_meta[STEM_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    conv3x3_bn_silu(in, out,
                    w_conv,
                    w_meta,
                    w_meta + STEM_CH,
                    INPUT_C, STEM_CH, INPUT_H, INPUT_W, /*stride=*/2);
}
