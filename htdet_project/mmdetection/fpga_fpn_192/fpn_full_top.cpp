/*
 * fpn_full_top.cpp
 * Phase 4: Complete FPN — P2+P3+P4+P5+P6.
 *
 * Computation order:
 *   1. lat4 = conv1x1(c4, 640→192, 10×10)
 *   2. lat3 = conv1x1(c3, 128→192, 20×20)
 *   3. lat2 = conv1x1(c2,  96→192, 40×40)
 *   4. lat1 = conv1x1(c1,  64→192, 80×80)   ← new
 *   5. lat3 += add_upsampled_2x(lat4)         [10→20]
 *   6. lat2 += add_upsampled_2x(lat3)         [20→40]
 *   7. lat1 += add_upsampled_2x(lat2)         [40→80]  ← new
 *   8. p5   = conv3x3(lat4)  [10×10]
 *   9. p4   = conv3x3(lat3)  [20×20]
 *  10. p3   = conv3x3(lat2)  [40×40]
 *  11. p2   = conv3x3(lat1)  [80×80]           ← new
 *  12. p6   = maxpool2×2(p5) [5×5]
 *
 * ⚠ BRAM: lat1 (4.8 MB) exceeds xczu9eg on-chip memory.
 *   HLS synthesis completes with >100% BRAM utilization.
 *   Latency estimate is valid; P&R requires xczu28dr or streaming redesign.
 *
 * Latency estimate (OC_TILE=16, II=5, w_conv cyclic-17 partition):
 *   lat4 conv1x1   :  12 OC × 100 HW × 640 IC × II5  ≈  19 ms
 *   lat3 conv1x1   :  12 OC × 400 HW × 128 IC × II5  ≈  15 ms
 *   lat2 conv1x1   :  12 OC × 1600 HW × 96 IC × II5  ≈  46 ms
 *   lat1 conv1x1   :  12 OC × 6400 HW × 64 IC × II5  ≈ 123 ms
 *   add_ups        :  77K + 307K + 1229K trips, II=1  ≈   8 ms
 *   P5 conv3x3     :  12×100 outer  × 8659 inner      ≈  52 ms
 *   P4 conv3x3     :  12×400 outer  × 8659 inner      ≈ 208 ms
 *   P3 conv3x3     :  12×1600 outer × 8659 inner      ≈ 833 ms
 *   P2 conv3x3     :  12×6400 outer × 8659 inner      ≈ 3.33 s
 *   Total estimate :                                   ≈ 4.63 s
 */

#include "fpn_full_top.h"
#include "fpga_utils.h"

void fpn_full_top(
    const act_t    c4    [FPN_FULL_C4_ELEMS],
    const act_t    c3    [FPN_FULL_C3_ELEMS],
    const act_t    c2    [FPN_FULL_C2_ELEMS],
    const act_t    c1    [FPN_FULL_C1_ELEMS],
    const weight_t w_conv[FPN_FULL_WCONV_ELEMS],
    const meta_t   w_meta[FPN_FULL_WMETA_ELEMS],
          act_t    p5    [FPN_FULL_P5_ELEMS],
          act_t    p4    [FPN_FULL_P4_ELEMS],
          act_t    p3    [FPN_FULL_P3_ELEMS],
          act_t    p2    [FPN_FULL_P2_ELEMS],
          act_t    p6    [FPN_FULL_P6_ELEMS]
) {
    #pragma HLS INTERFACE bram port=c4
    #pragma HLS INTERFACE bram port=c3
    #pragma HLS INTERFACE bram port=c2
    #pragma HLS INTERFACE bram port=c1
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE bram port=p5
    #pragma HLS INTERFACE bram port=p4
    #pragma HLS INTERFACE bram port=p3
    #pragma HLS INTERFACE bram port=p2
    #pragma HLS INTERFACE bram port=p6
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    static act_t lat4[FPN_FULL_P5_ELEMS];   // 192*10*10 =    19,200  f  (75 KB)
    static act_t lat3[FPN_FULL_P4_ELEMS];   // 192*20*20 =    76,800  f  (300 KB)
    static act_t lat2[FPN_FULL_P3_ELEMS];   // 192*40*40 =   307,200  f  (1.2 MB)
    static act_t lat1[FPN_FULL_P2_ELEMS];   // 192*80*80 = 1,228,800  f  (4.8 MB ⚠)
    #pragma HLS bind_storage variable=lat4 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat3 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat2 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat1 type=RAM_T2P impl=BRAM

    // add_upsampled_2x_inplace modifies lat3, lat2, lat1 via RMW
    #pragma HLS DEPENDENCE variable=lat3 inter false
    #pragma HLS DEPENDENCE variable=lat2 inter false
    #pragma HLS DEPENDENCE variable=lat1 inter false

    int wc = 0, wm = 0;

    // ----------------------------------------------------------------
    // Step 1–4: Lateral 1×1 convolutions
    // ----------------------------------------------------------------
    conv1x1_plain(c4, lat4, w_conv+wc, w_meta+wm, C4_CH, FPN_OUT_CH, P5_H, P5_W);
    wc += FPN_FULL_LAT4_W;   wm += FPN_OUT_CH;

    conv1x1_plain(c3, lat3, w_conv+wc, w_meta+wm, C3_CH, FPN_OUT_CH, P4_H, P4_W);
    wc += FPN_FULL_LAT3_W;   wm += FPN_OUT_CH;

    conv1x1_plain(c2, lat2, w_conv+wc, w_meta+wm, C2_CH, FPN_OUT_CH, P3_H, P3_W);
    wc += FPN_FULL_LAT2_W;   wm += FPN_OUT_CH;

    conv1x1_plain(c1, lat1, w_conv+wc, w_meta+wm, C1_CH, FPN_OUT_CH, P2_H, P2_W);
    wc += FPN_FULL_LAT1_W;   wm += FPN_OUT_CH;

    // ----------------------------------------------------------------
    // Step 5–7: Top-down merges (no temporary buffers)
    // ----------------------------------------------------------------
    add_upsampled_2x_inplace(lat4, lat3, FPN_OUT_CH, P5_H, P5_W);  // 10→20
    add_upsampled_2x_inplace(lat3, lat2, FPN_OUT_CH, P4_H, P4_W);  // 20→40
    add_upsampled_2x_inplace(lat2, lat1, FPN_OUT_CH, P3_H, P3_W);  // 40→80

    // ----------------------------------------------------------------
    // Step 8–11: Output 3×3 convolutions (OC_TILE=4, no activation)
    // ----------------------------------------------------------------
    fpn_conv3x3(lat4, p5, w_conv+wc, w_meta+wm, P5_H, P5_W);
    wc += FPN_FULL_OUT_W;   wm += FPN_OUT_CH;

    fpn_conv3x3(lat3, p4, w_conv+wc, w_meta+wm, P4_H, P4_W);
    wc += FPN_FULL_OUT_W;   wm += FPN_OUT_CH;

    fpn_conv3x3(lat2, p3, w_conv+wc, w_meta+wm, P3_H, P3_W);
    wc += FPN_FULL_OUT_W;   wm += FPN_OUT_CH;

    fpn_conv3x3(lat1, p2, w_conv+wc, w_meta+wm, P2_H, P2_W);
    wc += FPN_FULL_OUT_W;   wm += FPN_OUT_CH;

    // ----------------------------------------------------------------
    // Step 12: P6 = maxpool2×2(P5)
    // ----------------------------------------------------------------
    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);
}
