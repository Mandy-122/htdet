/*
 * fpn_p5_top.h
 * Phase 1: FPN P5+P6 levels only — smallest module for initial HLS validation.
 *
 * Computes: C4[640,10,10] → lateral 1×1 → lat4[192,10,10]
 *                         → output 3×3  → P5 [192,10,10]
 *                         → maxpool 2×2 → P6 [192, 5, 5]
 *
 * Why start here: P5 uses the smallest BRAM budget and fastest C-sim (~1 min).
 * After this passes, add P4 (fpn_p4p5_top), then P3, then full FPN.
 *
 * Weight layout (w_conv, int8, row-major):
 *   [0 .. LAT4_W-1]          lat4 lateral 1×1 weights [FPN_OUT_CH * C4_CH]
 *   [LAT4_W .. LAT4_W+OUT4_W-1] out4 output 3×3 weights [FPN_OUT_CH * FPN_OUT_CH * 9]
 *
 * Meta layout (w_meta, float):
 *   [0 .. FPN_OUT_CH-1]           lat4 bias
 *   [FPN_OUT_CH .. 2*FPN_OUT_CH-1] out4 bias
 *
 * Total weight elements:
 *   WCONV = 192*640 + 192*192*9 = 122,880 + 331,776 = 454,656  (int8)
 *   WMETA = 192 + 192           = 384                           (float)
 */

#ifndef FPN_P5_TOP_H
#define FPN_P5_TOP_H

#include "fpga_types.h"

// ---- Weight element counts for P5 phase ----
#define FPN_P5_LAT4_W   (FPN_OUT_CH * C4_CH)                       // 192*640 = 122,880
#define FPN_P5_OUT4_W   (FPN_OUT_CH * FPN_OUT_CH * 9)              // 192*192*9 = 331,776
#define FPN_P5_WCONV_ELEMS  (FPN_P5_LAT4_W + FPN_P5_OUT4_W)       // 454,656
#define FPN_P5_WMETA_ELEMS  (FPN_OUT_CH + FPN_OUT_CH)              // 384

// ---- I/O element counts ----
#define FPN_P5_C4_ELEMS   (C4_CH     * P5_H * P5_W)   // 640*10*10 = 64,000
#define FPN_P5_P5_ELEMS   (FPN_OUT_CH * P5_H * P5_W)  // 192*10*10 = 19,200
#define FPN_P5_P6_ELEMS   (FPN_OUT_CH * P6_H * P6_W)  // 192* 5* 5 =  4,800

void fpn_p5_top(
    const act_t    c4    [FPN_P5_C4_ELEMS],
    const weight_t w_conv[FPN_P5_WCONV_ELEMS],
    const meta_t   w_meta[FPN_P5_WMETA_ELEMS],
          act_t    p5    [FPN_P5_P5_ELEMS],
          act_t    p6    [FPN_P5_P6_ELEMS]
);

#endif // FPN_P5_TOP_H
