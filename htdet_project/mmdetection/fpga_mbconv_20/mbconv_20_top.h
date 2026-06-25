/*
 * mbconv_20_top.h  (v4 — HWC layout)
 * HLS top for mbconv_20: in_ch=128, out_ch=128, expand=4, stride=1
 * Spatial: 20×20 (C3 stage residual block)
 * hid = 512
 *
 * Weight layout (coff):
 *   expand 1×1 : [512 * 128]     = 65,536 int8
 *   dw     3×3 : [512 * 9]       =  4,608 int8
 *   proj   1×1 : [128 * 512]     = 65,536 int8
 *   Total                         135,680 int8
 *
 * Meta layout (moff):
 *   expand BN scale : [512]  expand BN bias : [512]  → 1,024 float
 *   dw     BN scale : [512]  dw     BN bias : [512]  → 1,024 float
 *   proj   BN scale : [128]  proj   BN bias : [128]  →   256 float
 *   Total                                             2,304 float
 */

#ifndef MBCONV_20_TOP_H
#define MBCONV_20_TOP_H

#include "fpga_types.h"

#define MBCONV20_IN_CH       C3_CH               // 128
#define MBCONV20_OUT_CH      C3_CH               // 128
#define MBCONV20_HID         512                 // 128 * 4

// HWC element counts
#define MBCONV20_IN_ELEMS    (C3_H * C3_W * MBCONV20_IN_CH)    // 20*20*128 =  51,200
#define MBCONV20_OUT_ELEMS   (C3_H * C3_W * MBCONV20_OUT_CH)   // 20*20*128 =  51,200
#define MBCONV20_WCONV_ELEMS (MBCONV20_HID * MBCONV20_IN_CH      \
                            + MBCONV20_HID * 9                    \
                            + MBCONV20_OUT_CH * MBCONV20_HID)     // 135,680
#define MBCONV20_WMETA_ELEMS ((MBCONV20_HID + MBCONV20_HID)       \
                            + (MBCONV20_HID + MBCONV20_HID)       \
                            + (MBCONV20_OUT_CH + MBCONV20_OUT_CH)) // 2,304

void mbconv_20_top(
    const act_t    in     [MBCONV20_IN_ELEMS],
          act_t    out    [MBCONV20_OUT_ELEMS],
    const weight_t w_conv [MBCONV20_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV20_WMETA_ELEMS]
);

#endif // MBCONV_20_TOP_H
