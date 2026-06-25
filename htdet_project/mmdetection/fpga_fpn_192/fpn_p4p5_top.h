/*
 * fpn_p4p5_top.h
 * Phase 2: FPN P4+P5+P6 levels.
 *
 * Adds the C3 path on top of Phase 1 (P5+P6):
 *   C4[640,10,10] → lat4[192,10,10]          ─── conv3x3 ─── P5[192,10,10] → P6[192,5,5]
 *   C3[128,20,20] → lat3[192,20,20] ← up(lat4) + add ─── conv3x3 ─── P4[192,20,20]
 *
 * Weight layout (w_conv, int8, row-major):
 *   [0         ..  122879]  lat4_w  FPN_OUT_CH × C4_CH      = 192×640  = 122,880
 *   [122880     ..  147455]  lat3_w  FPN_OUT_CH × C3_CH      = 192×128  =  24,576
 *   [147456     ..  479231]  out4_w  FPN_OUT_CH × FPN_OUT_CH × 9        = 331,776  (P5)
 *   [479232     ..  811007]  out3_w  FPN_OUT_CH × FPN_OUT_CH × 9        = 331,776  (P4)
 *   Total WCONV: 811,008 int8
 *
 * Meta layout (w_meta, float):
 *   [0   .. 191]   lat4_b  [192]
 *   [192 .. 383]   lat3_b  [192]
 *   [384 .. 575]   out4_b  [192]  (P5 output conv bias)
 *   [576 .. 767]   out3_b  [192]  (P4 output conv bias)
 *   Total WMETA: 768 floats
 */

#ifndef FPN_P4P5_TOP_H
#define FPN_P4P5_TOP_H

#include "fpga_types.h"

// ---- Weight element counts ----
#define FPN_P4P5_LAT4_W       (FPN_OUT_CH * C4_CH)               // 122,880
#define FPN_P4P5_LAT3_W       (FPN_OUT_CH * C3_CH)               //  24,576
#define FPN_P4P5_OUT4_W       (FPN_OUT_CH * FPN_OUT_CH * 9)      // 331,776
#define FPN_P4P5_OUT3_W       (FPN_OUT_CH * FPN_OUT_CH * 9)      // 331,776
#define FPN_P4P5_WCONV_ELEMS  (FPN_P4P5_LAT4_W  \
                              + FPN_P4P5_LAT3_W  \
                              + FPN_P4P5_OUT4_W  \
                              + FPN_P4P5_OUT3_W)                  // 811,008
#define FPN_P4P5_WMETA_ELEMS  (4 * FPN_OUT_CH)                   //     768

// ---- I/O element counts ----
#define FPN_P4P5_C4_ELEMS   (C4_CH      * P5_H * P5_W)   // 640*10*10 =  64,000
#define FPN_P4P5_C3_ELEMS   (C3_CH      * P4_H * P4_W)   // 128*20*20 =  51,200
#define FPN_P4P5_P5_ELEMS   (FPN_OUT_CH * P5_H * P5_W)   // 192*10*10 =  19,200
#define FPN_P4P5_P4_ELEMS   (FPN_OUT_CH * P4_H * P4_W)   // 192*20*20 =  76,800
#define FPN_P4P5_P6_ELEMS   (FPN_OUT_CH * P6_H * P6_W)   // 192* 5* 5 =   4,800

void fpn_p4p5_top(
    const act_t    c4    [FPN_P4P5_C4_ELEMS],
    const act_t    c3    [FPN_P4P5_C3_ELEMS],
    const weight_t w_conv[FPN_P4P5_WCONV_ELEMS],
    const meta_t   w_meta[FPN_P4P5_WMETA_ELEMS],
          act_t    p5    [FPN_P4P5_P5_ELEMS],
          act_t    p4    [FPN_P4P5_P4_ELEMS],
          act_t    p6    [FPN_P4P5_P6_ELEMS]
);

#endif // FPN_P4P5_TOP_H
