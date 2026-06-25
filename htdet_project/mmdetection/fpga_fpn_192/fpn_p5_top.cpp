/*
 * fpn_p5_top.cpp
 * Phase 1: FPN P5+P6 — lateral 1×1 (C4→192) + output 3×3 + maxpool.
 *
 * All arrays in CHW layout: [ch][H][W] → flat index [c*H*W + h*W + w]
 *
 * BRAM budget (Phase 1, float32):
 *   c4     : 640*10*10 * 4 =  256 KB  (input,  BRAM port)
 *   w_conv : 454,656   * 1 =  444 KB  (int8,   BRAM port)
 *   w_meta :     384   * 4 =    2 KB  (float,  BRAM port)
 *   lat4   : 192*10*10 * 4 =   75 KB  (on-chip BRAM)
 *   p5     : 192*10*10 * 4 =   75 KB  (output, BRAM port)
 *   p6     : 192* 5* 5 * 4 =   19 KB  (output, BRAM port)
 *
 * Partitioning note (Phase 1 — no UNROLL in FPN functions):
 *   No array partitioning on c4 or w_conv: conv1x1_plain and fpn_conv3x3
 *   are pipelined at the innermost single-MAC loop with no UNROLL, so there
 *   is only 1 read per clock — no bank-conflict benefit from partitioning.
 *   lat4: no partition needed for same reason.
 *   Phase-2 performance optimisation will re-introduce UNROLL + partition.
 */

#include "fpn_p5_top.h"
#include "fpga_utils.h"

void fpn_p5_top(
    const act_t    c4    [FPN_P5_C4_ELEMS],
    const weight_t w_conv[FPN_P5_WCONV_ELEMS],
    const meta_t   w_meta[FPN_P5_WMETA_ELEMS],
          act_t    p5    [FPN_P5_P5_ELEMS],
          act_t    p6    [FPN_P5_P6_ELEMS]
) {
    #pragma HLS INTERFACE bram port=c4
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE bram port=p5
    #pragma HLS INTERFACE bram port=p6
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // No array partition pragmas: 1 read/write per clock (single-MAC pipeline).
    // Partitioning will be added in Phase 2 alongside UNROLL re-introduction.

    static act_t lat4[FPN_P5_P5_ELEMS];    // 192*10*10 = 19,200 floats
    #pragma HLS bind_storage variable=lat4 type=RAM_T2P impl=BRAM

    int wc = 0;   // int8 weight offset
    int wm = 0;   // float meta offset

    // ----------------------------------------------------------
    // Step 1: Lateral 1×1 convolution  (C4_CH=640 → FPN_OUT_CH=192)
    // Weight: w_conv[0 .. 122879]  (FPN_OUT_CH × C4_CH = 192×640 int8)
    // Bias:   w_meta[0 .. 191]     (FPN_OUT_CH floats)
    // ----------------------------------------------------------
    conv1x1_plain(c4, lat4,
                  w_conv + wc,
                  w_meta + wm,
                  C4_CH, FPN_OUT_CH, P5_H, P5_W);
    wc += FPN_P5_LAT4_W;   // +122,880
    wm += FPN_OUT_CH;       // +192

    // ----------------------------------------------------------
    // Step 2: Output 3×3 convolution (192→192, no activation)
    // Weight: w_conv[122880 .. 454655]  (FPN_OUT_CH × FPN_OUT_CH × 9 int8)
    // Bias:   w_meta[192 .. 383]        (FPN_OUT_CH floats)
    // ----------------------------------------------------------
    fpn_conv3x3(lat4, p5,
                w_conv + wc,
                w_meta + wm,
                P5_H, P5_W);
    wc += FPN_P5_OUT4_W;    // +331,776
    wm += FPN_OUT_CH;        // +192

    // ----------------------------------------------------------
    // Step 3: P6 = maxpool2×2(P5)
    // No weights. P6 spatial: 10×10 → 5×5
    // ----------------------------------------------------------
    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);
}
