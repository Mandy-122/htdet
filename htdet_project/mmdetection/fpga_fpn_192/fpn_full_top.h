/*
 * fpn_full_top.h
 * Phase 4: Complete FPN — P2+P3+P4+P5+P6 (all 5 levels).
 *
 * Adds C1[64,80,80] on top of Phase 3 (P3+P4+P5+P6).
 *
 * ⚠ BRAM NOTE: lat1[192,80,80] = 4.8 MB requires ~2,401 BRAM18K.
 *   ZCU9EG has 1,824 BRAM18K total → ~171% utilization on synthesis.
 *   Synthesis completes (latency/correctness valid); P&R will fail on xczu9eg.
 *   Target xczu28dr or redesign P2 as streaming to deploy.
 *
 * Top-down pathway:
 *   C4[640,10,10] → lat4[192,10,10] ──────────────────────── conv3x3 → P5 → maxpool → P6
 *   C3[128,20,20] → lat3[192,20,20] ← add_up(lat4) ──────── conv3x3 → P4
 *   C2[ 96,40,40] → lat2[192,40,40] ← add_up(lat3) ──────── conv3x3 → P3
 *   C1[ 64,80,80] → lat1[192,80,80] ← add_up(lat2) ──────── conv3x3 → P2
 *
 * Weight layout (w_conv, int8):
 *   [0         ..  122879]  lat4_w  192×640   = 122,880
 *   [122880     ..  147455]  lat3_w  192×128   =  24,576
 *   [147456     ..  165887]  lat2_w  192× 96   =  18,432
 *   [165888     ..  178175]  lat1_w  192× 64   =  12,288
 *   [178176     ..  509951]  out4_w  192×192×9 = 331,776  (P5)
 *   [509952     ..  841727]  out3_w  192×192×9 = 331,776  (P4)
 *   [841728     .. 1173503]  out2_w  192×192×9 = 331,776  (P3)
 *   [1173504    .. 1505279]  out1_w  192×192×9 = 331,776  (P2)
 *   Total WCONV = 1,505,280
 *
 * Weight layout (w_meta, float):
 *   [0   ..  191]  lat4_b  [0   ..  383]  lat3_b
 *   [384 ..  575]  lat2_b  [576 ..  767]  lat1_b
 *   [768 ..  959]  out4_b  [960 .. 1151]  out3_b
 *   [1152.. 1343]  out2_b  [1344.. 1535]  out1_b
 *   Total WMETA = 1,536
 */

#ifndef FPN_FULL_TOP_H
#define FPN_FULL_TOP_H

#include "fpga_types.h"

#define FPN_FULL_LAT4_W    (FPN_OUT_CH * C4_CH)           // 122,880
#define FPN_FULL_LAT3_W    (FPN_OUT_CH * C3_CH)           //  24,576
#define FPN_FULL_LAT2_W    (FPN_OUT_CH * C2_CH)           //  18,432
#define FPN_FULL_LAT1_W    (FPN_OUT_CH * C1_CH)           //  12,288
#define FPN_FULL_OUT_W     (FPN_OUT_CH * FPN_OUT_CH * 9)  // 331,776
#define FPN_FULL_WCONV_ELEMS  (FPN_FULL_LAT4_W          \
                             + FPN_FULL_LAT3_W           \
                             + FPN_FULL_LAT2_W           \
                             + FPN_FULL_LAT1_W           \
                             + 4 * FPN_FULL_OUT_W)        // 1,505,280
#define FPN_FULL_WMETA_ELEMS  (8 * FPN_OUT_CH)            //     1,536

#define FPN_FULL_C4_ELEMS  (C4_CH      * P5_H * P5_W)   //   64,000
#define FPN_FULL_C3_ELEMS  (C3_CH      * P4_H * P4_W)   //   51,200
#define FPN_FULL_C2_ELEMS  (C2_CH      * P3_H * P3_W)   //  153,600
#define FPN_FULL_C1_ELEMS  (C1_CH      * P2_H * P2_W)   //  409,600
#define FPN_FULL_P5_ELEMS  (FPN_OUT_CH * P5_H * P5_W)   //   19,200
#define FPN_FULL_P4_ELEMS  (FPN_OUT_CH * P4_H * P4_W)   //   76,800
#define FPN_FULL_P3_ELEMS  (FPN_OUT_CH * P3_H * P3_W)   //  307,200
#define FPN_FULL_P2_ELEMS  (FPN_OUT_CH * P2_H * P2_W)   // 1,228,800
#define FPN_FULL_P6_ELEMS  (FPN_OUT_CH * P6_H * P6_W)   //    4,800

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
);

#endif // FPN_FULL_TOP_H
