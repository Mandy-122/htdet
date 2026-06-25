/*
 * fpn_p3p4p5_top.h
 * Phase 3: FPN P3+P4+P5+P6.
 *
 * Adds C2[96,40,40] on top of Phase 2 (P4+P5+P6).
 *
 * Top-down pathway:
 *   C4[640,10,10] → lat4[192,10,10] ────────────────── conv3x3 → P5 → maxpool → P6
 *   C3[128,20,20] → lat3[192,20,20] ← add_up(lat4)  ─ conv3x3 → P4
 *   C2[ 96,40,40] → lat2[192,40,40] ← add_up(lat3)  ─ conv3x3 → P3
 *
 * add_up = add_upsampled_2x_inplace (no temporary up-buffer needed).
 *
 * Weight layout (w_conv, int8):
 *   [0          ..  122879]  lat4_w  192×640          =  122,880
 *   [122880      ..  147455]  lat3_w  192×128          =   24,576
 *   [147456      ..  165887]  lat2_w  192× 96          =   18,432
 *   [165888      ..  497663]  out4_w  192×192×9        =  331,776  (P5)
 *   [497664      ..  829439]  out3_w  192×192×9        =  331,776  (P4)
 *   [829440      .. 1161215]  out2_w  192×192×9        =  331,776  (P3)
 *   Total WCONV = 1,161,216
 *
 * Weight layout (w_meta, float):
 *   [0   ..  191]  lat4_b  [192]
 *   [192 ..  383]  lat3_b  [192]
 *   [384 ..  575]  lat2_b  [192]
 *   [576 ..  767]  out4_b  [192]
 *   [768 ..  959]  out3_b  [192]
 *   [960 .. 1151]  out2_b  [192]
 *   Total WMETA = 1,152
 */

#ifndef FPN_P3P4P5_TOP_H
#define FPN_P3P4P5_TOP_H

#include "fpga_types.h"

#define FPN_P3P4P5_LAT4_W    (FPN_OUT_CH * C4_CH)           // 122,880
#define FPN_P3P4P5_LAT3_W    (FPN_OUT_CH * C3_CH)           //  24,576
#define FPN_P3P4P5_LAT2_W    (FPN_OUT_CH * C2_CH)           //  18,432
#define FPN_P3P4P5_OUT_W     (FPN_OUT_CH * FPN_OUT_CH * 9)  // 331,776
#define FPN_P3P4P5_WCONV_ELEMS  (FPN_P3P4P5_LAT4_W      \
                               + FPN_P3P4P5_LAT3_W      \
                               + FPN_P3P4P5_LAT2_W      \
                               + 3 * FPN_P3P4P5_OUT_W)      // 1,161,216
#define FPN_P3P4P5_WMETA_ELEMS  (6 * FPN_OUT_CH)            //     1,152

#define FPN_P3P4P5_C4_ELEMS   (C4_CH      * P5_H * P5_W)   // 640*10*10 =   64,000
#define FPN_P3P4P5_C3_ELEMS   (C3_CH      * P4_H * P4_W)   // 128*20*20 =   51,200
#define FPN_P3P4P5_C2_ELEMS   (C2_CH      * P3_H * P3_W)   //  96*40*40 =  153,600
#define FPN_P3P4P5_P5_ELEMS   (FPN_OUT_CH * P5_H * P5_W)   // 192*10*10 =   19,200
#define FPN_P3P4P5_P4_ELEMS   (FPN_OUT_CH * P4_H * P4_W)   // 192*20*20 =   76,800
#define FPN_P3P4P5_P3_ELEMS   (FPN_OUT_CH * P3_H * P3_W)   // 192*40*40 =  307,200
#define FPN_P3P4P5_P6_ELEMS   (FPN_OUT_CH * P6_H * P6_W)   // 192* 5* 5 =    4,800

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
);

#endif // FPN_P3P4P5_TOP_H
