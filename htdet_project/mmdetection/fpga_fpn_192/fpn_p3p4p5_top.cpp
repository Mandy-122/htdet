/*
 * fpn_p3p4p5_top.cpp
 * Phase 3: FPN P3+P4+P5+P6.
 *
 * All arrays CHW: [ch][H][W] → c*H*W + h*W + w
 *
 * Computation order:
 *   1. lat4 = conv1x1(c4, 640→192, 10×10)
 *   2. lat3 = conv1x1(c3, 128→192, 20×20)
 *   3. lat2 = conv1x1(c2,  96→192, 40×40)
 *   4. lat3 += add_upsampled_2x(lat4)     [10×10 upsampled to 20×20, added in-place]
 *   5. lat2 += add_upsampled_2x(lat3)     [20×20 upsampled to 40×40, added in-place]
 *   6. p5   = conv3x3(lat4, no act)       [10×10]
 *   7. p4   = conv3x3(lat3, no act)       [20×20]  ← uses merged lat3
 *   8. p3   = conv3x3(lat2, no act)       [40×40]  ← uses merged lat2
 *   9. p6   = maxpool2×2(p5)              [5×5]
 *
 * No temporary upsample buffers: add_upsampled_2x_inplace reads src and
 * modifies dst in a single pass, saving 300 KB (lat3 level) + 0 KB for lat4
 * (compared to fpn_p4p5 which used a separate up4 buffer).
 *
 * BRAM budget (float32):
 *   c4    :  640*10*10 *4 =   250 KB  (BRAM port)
 *   c3    :  128*20*20 *4 =   200 KB  (BRAM port)
 *   c2    :   96*40*40 *4 =   600 KB  (BRAM port)
 *   w_conv: 1,161,216 *1 = 1,134 KB  (BRAM port, int8)
 *   w_meta:     1,152 *4 =     5 KB  (BRAM port)
 *   lat4  :  192*10*10 *4 =    75 KB  (on-chip BRAM)
 *   lat3  :  192*20*20 *4 =   300 KB  (on-chip BRAM, modified in-place)
 *   lat2  :  192*40*40 *4 = 1,200 KB  (on-chip BRAM, modified in-place)
 *   p5    :  192*10*10 *4 =    75 KB  (BRAM port)
 *   p4    :  192*20*20 *4 =   300 KB  (BRAM port)
 *   p3    :  192*40*40 *4 = 1,200 KB  (BRAM port)
 *   p6    :  192* 5* 5 *4 =    19 KB  (BRAM port)
 *
 * On-chip internal: lat4+lat3+lat2 = 75+300+1200 = 1,575 KB ≈ 788 BRAM18K (43%)
 *
 * Latency estimate (OC_TILE=4, II=5 from Phase 2 calibration):
 *   lat4 conv1×1 : 77 ms
 *   lat3 conv1×1 : 61 ms
 *   lat2 conv1×1 :  96*40*40=153,600 trip × II=5 × C1P_OC/4=48 ≈ 184 ms
 *   add_up steps :  negligible (II=1 each)
 *   P5 conv3×3   : 208 ms
 *   P4 conv3×3   : 832 ms
 *   P3 conv3×3   : 48×40×40 outer × 8672 inner ≈ 3.33 s
 *   Total estimate: ~4.7 s
 */

#include "fpn_p3p4p5_top.h"
#include "fpga_utils.h"

void fpn_p3p4p5_top(
    const act_t    c4    [FPN_P3P4P5_C4_ELEMS],
    const act_t    c3    [FPN_P3P4P5_C3_ELEMS],
    const act_t    c2    [FPN_P3P4P5_C2_ELEMS],
    const weight_t w_conv[FPN_P3P4P5_WCONV_ELEMS],
    const meta_t   w_meta[FPN_P3P4P5_WMETA_ELEMS],
          act_t    p5    [FPN_P3P4P5_P5_ELEMS],
          act_t    p4    [FPN_P3P4P5_P4_ELEMS],
          act_t    p3    [FPN_P3P4P5_P3_ELEMS],
          act_t    p6    [FPN_P3P4P5_P6_ELEMS]
) {
    #pragma HLS INTERFACE bram port=c4
    #pragma HLS INTERFACE bram port=c3
    #pragma HLS INTERFACE bram port=c2
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE bram port=p5
    #pragma HLS INTERFACE bram port=p4
    #pragma HLS INTERFACE bram port=p3
    #pragma HLS INTERFACE bram port=p6
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // Internal lateral feature map buffers
    static act_t lat4[FPN_P3P4P5_P5_ELEMS];   // 192*10*10 =   19,200 floats
    static act_t lat3[FPN_P3P4P5_P4_ELEMS];   // 192*20*20 =   76,800 floats
    static act_t lat2[FPN_P3P4P5_P3_ELEMS];   // 192*40*40 =  307,200 floats
    #pragma HLS bind_storage variable=lat4 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat3 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat2 type=RAM_T2P impl=BRAM

    // add_upsampled_2x_inplace modifies lat3 and lat2 via RMW.
    // Each element is written exactly once → no inter-iteration dependency.
    #pragma HLS DEPENDENCE variable=lat3 inter false
    #pragma HLS DEPENDENCE variable=lat2 inter false

    int wc = 0, wm = 0;

    // ----------------------------------------------------------------
    // Step 1–3: Lateral 1×1 convolutions (CHW, no activation)
    // ----------------------------------------------------------------
    // lat4 = conv1x1(c4)  640→192  10×10
    conv1x1_plain(c4, lat4, w_conv+wc, w_meta+wm, C4_CH, FPN_OUT_CH, P5_H, P5_W);
    wc += FPN_P3P4P5_LAT4_W;   wm += FPN_OUT_CH;

    // lat3 = conv1x1(c3)  128→192  20×20
    conv1x1_plain(c3, lat3, w_conv+wc, w_meta+wm, C3_CH, FPN_OUT_CH, P4_H, P4_W);
    wc += FPN_P3P4P5_LAT3_W;   wm += FPN_OUT_CH;

    // lat2 = conv1x1(c2)   96→192  40×40
    conv1x1_plain(c2, lat2, w_conv+wc, w_meta+wm, C2_CH, FPN_OUT_CH, P3_H, P3_W);
    wc += FPN_P3P4P5_LAT2_W;   wm += FPN_OUT_CH;

    // ----------------------------------------------------------------
    // Step 4–5: Top-down pathway (no temporary buffer)
    //   lat3 += nearest-neighbour upsample(lat4, 10×10 → 20×20)
    //   lat2 += nearest-neighbour upsample(lat3_merged, 20×20 → 40×40)
    // ----------------------------------------------------------------
    add_upsampled_2x_inplace(lat4, lat3, FPN_OUT_CH, P5_H, P5_W);
    add_upsampled_2x_inplace(lat3, lat2, FPN_OUT_CH, P4_H, P4_W);

    // ----------------------------------------------------------------
    // Step 6–8: Output 3×3 convolutions (OC_TILE=4, no activation)
    // ----------------------------------------------------------------
    // P5 = conv3x3(lat4)   w_conv[165888..497663]  w_meta[576..767]
    fpn_conv3x3(lat4, p5, w_conv+wc, w_meta+wm, P5_H, P5_W);
    wc += FPN_P3P4P5_OUT_W;   wm += FPN_OUT_CH;

    // P4 = conv3x3(lat3_merged)   w_conv[497664..829439]  w_meta[768..959]
    fpn_conv3x3(lat3, p4, w_conv+wc, w_meta+wm, P4_H, P4_W);
    wc += FPN_P3P4P5_OUT_W;   wm += FPN_OUT_CH;

    // P3 = conv3x3(lat2_merged)   w_conv[829440..1161215]  w_meta[960..1151]
    fpn_conv3x3(lat2, p3, w_conv+wc, w_meta+wm, P3_H, P3_W);
    wc += FPN_P3P4P5_OUT_W;   wm += FPN_OUT_CH;

    // ----------------------------------------------------------------
    // Step 9: P6 = maxpool2×2(P5)   [10×10 → 5×5]
    // ----------------------------------------------------------------
    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);
}
