/*
 * fpn_p4p5_top.cpp
 * Phase 2: FPN P4+P5+P6 — adds the C3 lateral path to Phase 1.
 *
 * All arrays: CHW layout [ch][H][W] → flat index c*H*W + h*W + w
 *
 * Computation order:
 *   1. lat4 = conv1x1(c4, 640→192)            [10×10]
 *   2. lat3 = conv1x1(c3, 128→192)            [20×20]
 *   3. up4  = upsample_2x(lat4)               [10×10 → 20×20]
 *   4. lat3 += up4   (top-down add)            [20×20]
 *   5. p5   = conv3x3(lat4, no act)            [10×10]
 *   6. p4   = conv3x3(lat3, no act)            [20×20]
 *   7. p6   = maxpool2x2(p5)                   [10×10 → 5×5]
 *
 * BRAM budget (float32):
 *   c4     :  640*10*10 *4 =  250 KB  (BRAM input port)
 *   c3     :  128*20*20 *4 =  200 KB  (BRAM input port)
 *   w_conv :  811,008   *1 =  793 KB  (BRAM input port, int8)
 *   w_meta :      768   *4 =    3 KB  (BRAM input port, float)
 *   lat4   :  192*10*10 *4 =   75 KB  (on-chip BRAM)
 *   lat3   :  192*20*20 *4 =  300 KB  (on-chip BRAM)
 *   up4    :  192*20*20 *4 =  300 KB  (on-chip BRAM — eliminated in Phase 3)
 *   p5     :  192*10*10 *4 =   75 KB  (BRAM output port)
 *   p4     :  192*20*20 *4 =  300 KB  (BRAM output port)
 *   p6     :  192* 5* 5 *4 =   19 KB  (BRAM output port)
 *
 * Latency estimate (II=5 from Phase 1 calibration):
 *   lat4 conv1x1  :  192 OC × (640*100 trip × II5) ≈  307 ms
 *   lat3 conv1x1  :  192 OC × (128*400 trip × II5) ≈  246 ms
 *   upsample+add  :  trivial (< 1 ms)
 *   P5 conv3x3    :  192*10*10 outer × 8656 inner   ≈  832 ms
 *   P4 conv3x3    :  192*20*20 outer × 8656 inner   ≈ 3328 ms  (4× P5)
 *   P6 maxpool    :  negligible
 *   Total estimate: ~4.7 s  (validation phase — latency optimisation in Phase 4)
 */

#include "fpn_p4p5_top.h"
#include "fpga_utils.h"

void fpn_p4p5_top(
    const act_t    c4    [FPN_P4P5_C4_ELEMS],
    const act_t    c3    [FPN_P4P5_C3_ELEMS],
    const weight_t w_conv[FPN_P4P5_WCONV_ELEMS],
    const meta_t   w_meta[FPN_P4P5_WMETA_ELEMS],
          act_t    p5    [FPN_P4P5_P5_ELEMS],
          act_t    p4    [FPN_P4P5_P4_ELEMS],
          act_t    p6    [FPN_P4P5_P6_ELEMS]
) {
    #pragma HLS INTERFACE bram port=c4
    #pragma HLS INTERFACE bram port=c3
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE bram port=p5
    #pragma HLS INTERFACE bram port=p4
    #pragma HLS INTERFACE bram port=p6
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // lat4: [192, 10, 10] = 19,200 floats (75 KB)
    static act_t lat4[FPN_P4P5_P5_ELEMS];
    #pragma HLS bind_storage variable=lat4 type=RAM_T2P impl=BRAM

    // lat3: [192, 20, 20] = 76,800 floats (300 KB) — modified in-place during top-down
    static act_t lat3[FPN_P4P5_P4_ELEMS];
    #pragma HLS bind_storage variable=lat3 type=RAM_T2P impl=BRAM

    // up4: upsample of lat4 from 10×10 → 20×20, added to lat3
    static act_t up4[FPN_P4P5_P4_ELEMS];
    #pragma HLS bind_storage variable=up4 type=RAM_T2P impl=BRAM

    int wc = 0, wm = 0;

    // ----------------------------------------------------------------
    // Step 1: Lateral 1×1 for C4 (640→192, 10×10)
    // w_conv[0 .. 122879], w_meta[0 .. 191]
    // ----------------------------------------------------------------
    conv1x1_plain(c4, lat4,
                  w_conv + wc, w_meta + wm,
                  C4_CH, FPN_OUT_CH, P5_H, P5_W);
    wc += FPN_P4P5_LAT4_W;   // +122,880
    wm += FPN_OUT_CH;          // +192

    // ----------------------------------------------------------------
    // Step 2: Lateral 1×1 for C3 (128→192, 20×20)
    // w_conv[122880 .. 147455], w_meta[192 .. 383]
    // ----------------------------------------------------------------
    conv1x1_plain(c3, lat3,
                  w_conv + wc, w_meta + wm,
                  C3_CH, FPN_OUT_CH, P4_H, P4_W);
    wc += FPN_P4P5_LAT3_W;    // +24,576
    wm += FPN_OUT_CH;           // +192

    // ----------------------------------------------------------------
    // Step 3: Top-down pathway — upsample lat4 and add to lat3
    //   up4[c, 2h+dh, 2w+dw] = lat4[c, h, w]  for dh,dw in {0,1}
    //   lat3 += up4
    // ----------------------------------------------------------------
    upsample_2x(lat4, up4, FPN_OUT_CH, P5_H, P5_W);
    elem_add_inplace(lat3, up4, FPN_P4P5_P4_ELEMS);

    // ----------------------------------------------------------------
    // Step 4: Output 3×3 for P5 (lat4 → p5, no activation)
    // w_conv[147456 .. 479231], w_meta[384 .. 575]
    // ----------------------------------------------------------------
    fpn_conv3x3(lat4, p5,
                w_conv + wc, w_meta + wm,
                P5_H, P5_W);
    wc += FPN_P4P5_OUT4_W;    // +331,776
    wm += FPN_OUT_CH;           // +192

    // ----------------------------------------------------------------
    // Step 5: Output 3×3 for P4 (lat3_merged → p4, no activation)
    // w_conv[479232 .. 811007], w_meta[576 .. 767]
    // ----------------------------------------------------------------
    fpn_conv3x3(lat3, p4,
                w_conv + wc, w_meta + wm,
                P4_H, P4_W);
    wc += FPN_P4P5_OUT3_W;    // +331,776
    wm += FPN_OUT_CH;           // +192

    // ----------------------------------------------------------------
    // Step 6: P6 = maxpool2×2(P5)   [10×10 → 5×5]
    // ----------------------------------------------------------------
    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);
}
