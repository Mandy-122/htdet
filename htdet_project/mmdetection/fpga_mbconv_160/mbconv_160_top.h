/*
 * mbconv_160_top.h  (v4 — HWC layout)
 * HLS top for mbconv_160: STAGE0→C1 downsampling block
 * in_ch=32 (STAGE0_CH), out_ch=64 (C1_CH), expand=4, stride=2
 * Input: 160×160×32 (HWC, STEM spatial)
 * Output:  80×80×64 (HWC, C1 spatial)
 * hid = 128
 *
 * NOTE: This module has very large intermediate buffers at 160×160 spatial.
 *       ex_buf  [160×160×128] = 12.5 MB — far exceeds device URAM (2.88 MB).
 *       Intended for latency characterisation only; tiling/streaming required
 *       for physical implementation.
 *
 * Weight layout (coff):
 *   expand 1×1 : [128 * 32]      =  4,096 int8
 *   dw     3×3 : [128 * 9]       =  1,152 int8
 *   proj   1×1 : [ 64 * 128]     =  8,192 int8
 *   Total                          13,440 int8
 *
 * Meta layout (moff):
 *   expand BN scale/bias : [128+128] = 256 float
 *   dw     BN scale/bias : [128+128] = 256 float
 *   proj   BN scale/bias : [ 64+ 64] = 128 float
 *   Total                              640 float
 */

#ifndef MBCONV_160_TOP_H
#define MBCONV_160_TOP_H

#include "fpga_types.h"

#define MBCONV160_IN_CH       STAGE0_CH             // 32
#define MBCONV160_OUT_CH      C1_CH                 // 64
#define MBCONV160_HID         128                   // 32 * 4

// HWC element counts
#define MBCONV160_IN_ELEMS    (STEM_H * STEM_W * MBCONV160_IN_CH)   // 160*160*32  = 819,200
#define MBCONV160_OUT_ELEMS   (C1_H   * C1_W   * MBCONV160_OUT_CH)  //  80*80*64   = 409,600
#define MBCONV160_WCONV_ELEMS (MBCONV160_HID * MBCONV160_IN_CH        \
                             + MBCONV160_HID * 9                       \
                             + MBCONV160_OUT_CH * MBCONV160_HID)       // 13,440
#define MBCONV160_WMETA_ELEMS ((MBCONV160_HID + MBCONV160_HID)         \
                             + (MBCONV160_HID + MBCONV160_HID)         \
                             + (MBCONV160_OUT_CH + MBCONV160_OUT_CH))  // 640

void mbconv_160_top(
    const act_t    in     [MBCONV160_IN_ELEMS],
          act_t    out    [MBCONV160_OUT_ELEMS],
    const weight_t w_conv [MBCONV160_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV160_WMETA_ELEMS]
);

#endif // MBCONV_160_TOP_H
