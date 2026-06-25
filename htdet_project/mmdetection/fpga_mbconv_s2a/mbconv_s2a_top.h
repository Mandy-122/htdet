/*
 * mbconv_s2a_top.h  (v4 — HWC layout)
 * HLS top for Stage-2a MBConv: C1_CH→C2_CH downsampler (64→96, expand=4, stride=2)
 * Input : c1[80, 80, 64]   (HWC, C1 spatial)
 * Output: s2_mb[40, 40, 96] (HWC, C2 spatial)
 * hid = 64 × 4 = 256
 *
 * stride=2, in_ch(64) ≠ out_ch(96) → NO residual add
 *
 * Weight layout (w_conv, int8):
 *   expand 1×1 : [256 *  64]  = 16,384 int8
 *   dw     3×3 : [256 *   9]  =  2,304 int8
 *   proj   1×1 : [ 96 * 256]  = 24,576 int8
 *   Total                       43,264 int8
 *
 * Meta layout (w_meta, float):
 *   expand BN scale/bias : [256+256]  = 512 float
 *   dw     BN scale/bias : [256+256]  = 512 float
 *   proj   BN scale/bias : [ 96+ 96]  = 192 float
 *   Total                             1,216 float
 *
 * ⚠ URAM: ex_buf [80×80×256 × float] = 6.25 MB > device URAM (2.88 MB)
 *   dw_buf [40×40×256 × float] = 1.5625 MB < 2.88 MB ✓
 *   ex_buf overflows; dw_buf fits.
 */

#ifndef MBCONV_S2A_TOP_H
#define MBCONV_S2A_TOP_H

#include "fpga_types.h"

#define MBCONVS2A_IN_CH    C1_CH        // 64
#define MBCONVS2A_OUT_CH   C2_CH        // 96
#define MBCONVS2A_HID      256          // 64 * 4

#define MBCONVS2A_IN_ELEMS   (C1_H * C1_W * MBCONVS2A_IN_CH)     // 80*80*64 = 409,600
#define MBCONVS2A_OUT_ELEMS  (C2_H * C2_W * MBCONVS2A_OUT_CH)    // 40*40*96 = 153,600
#define MBCONVS2A_WCONV_ELEMS (MBCONVS2A_HID * MBCONVS2A_IN_CH   \
                             + MBCONVS2A_HID * 9                   \
                             + MBCONVS2A_OUT_CH * MBCONVS2A_HID)   // 43,264
#define MBCONVS2A_WMETA_ELEMS ((MBCONVS2A_HID  + MBCONVS2A_HID)  \
                             + (MBCONVS2A_HID  + MBCONVS2A_HID)  \
                             + (MBCONVS2A_OUT_CH + MBCONVS2A_OUT_CH)) // 1,216

void mbconv_s2a_top(
    const act_t    in    [MBCONVS2A_IN_ELEMS],
          act_t    out   [MBCONVS2A_OUT_ELEMS],
    const weight_t w_conv[MBCONVS2A_WCONV_ELEMS],
    const meta_t   w_meta[MBCONVS2A_WMETA_ELEMS]
);

#endif // MBCONV_S2A_TOP_H
