/*
 * mbconv_s0_top.h  (v4 — HWC layout)
 * HLS top for Stage-0 MBConv: STEM_CH→STAGE0_CH (16→32, expand=4, stride=1)
 * Input : stem_out[160, 160, 16]  (HWC)
 * Output: s0_out  [160, 160, 32]  (HWC)
 * hid = 16 × 4 = 64
 *
 * stride=1, in_ch(16) ≠ out_ch(32) → NO residual add
 *
 * Weight layout (w_conv, int8):
 *   expand 1×1 : [64  * 16]   =  1,024 int8
 *   dw     3×3 : [64  *  9]   =    576 int8
 *   proj   1×1 : [32  * 64]   =  2,048 int8
 *   Total                       3,648 int8
 *
 * Meta layout (w_meta, float):
 *   expand BN scale/bias : [64+64]   = 128 float
 *   dw     BN scale/bias : [64+64]   = 128 float
 *   proj   BN scale/bias : [32+32]   =  64 float
 *   Total                             320 float
 *
 * ⚠ URAM WARNING: ex_buf [160×160×64 × float] = 6.25 MB > device URAM (2.88 MB)
 *   dw_buf = same. Synthesis completes; P&R requires tiling.
 */

#ifndef MBCONV_S0_TOP_H
#define MBCONV_S0_TOP_H

#include "fpga_types.h"

#define MBCONVS0_IN_CH    STEM_CH           // 16
#define MBCONVS0_OUT_CH   STAGE0_CH         // 32
#define MBCONVS0_HID      64               // 16 * 4

#define MBCONVS0_IN_ELEMS   (STEM_H * STEM_W * MBCONVS0_IN_CH)    // 160*160*16 = 409,600
#define MBCONVS0_OUT_ELEMS  (STEM_H * STEM_W * MBCONVS0_OUT_CH)   // 160*160*32 = 819,200
#define MBCONVS0_WCONV_ELEMS (MBCONVS0_HID * MBCONVS0_IN_CH       \
                            + MBCONVS0_HID * 9                     \
                            + MBCONVS0_OUT_CH * MBCONVS0_HID)      // 3,648
#define MBCONVS0_WMETA_ELEMS ((MBCONVS0_HID  + MBCONVS0_HID)      \
                            + (MBCONVS0_HID  + MBCONVS0_HID)      \
                            + (MBCONVS0_OUT_CH + MBCONVS0_OUT_CH)) // 320

void mbconv_s0_top(
    const act_t    in    [MBCONVS0_IN_ELEMS],
          act_t    out   [MBCONVS0_OUT_ELEMS],
    const weight_t w_conv[MBCONVS0_WCONV_ELEMS],
    const meta_t   w_meta[MBCONVS0_WMETA_ELEMS]
);

#endif // MBCONV_S0_TOP_H
