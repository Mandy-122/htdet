/*
 * mbconv_40_top.h
 * HLS top function for isolated C-synthesis of mbconv_40.
 *
 * Spatial  : H=C2_H=40, W=C2_W=40  (PTQ INT8, W8A32, 320×320 input)
 * Test config: in_ch=96, out_ch=128, expand=4, stride=2
 *   → hid=384, oH=20, oW=20, use_res=false (stride≠1)
 *
 * Weight layout (coff stride):
 *   w_conv[0          .. hid*in_ch-1]      expand 1×1  (int8, 36,864 elem)
 *   w_conv[36864      .. +hid*9-1]         dw     3×3  (int8,  3,456 elem)
 *   w_conv[40320      .. +out*hid-1]       proj   1×1  (int8, 49,152 elem)
 *   Total: MBCONV40_WCONV_ELEMS = 89,472
 *
 * Meta layout (moff stride):
 *   w_meta[0    .. hid-1]           expand BN scale  (float, 384)
 *   w_meta[384  .. 767]             expand BN bias   (float, 384)
 *   w_meta[768  .. 1151]            dw     BN scale  (float, 384)
 *   w_meta[1152 .. 1535]            dw     BN bias   (float, 384)
 *   w_meta[1536 .. 1663]            proj   BN scale  (float, 128)
 *   w_meta[1664 .. 1791]            proj   BN bias   (float, 128)
 *   Total: MBCONV40_WMETA_ELEMS = 1,792
 */

#ifndef MBCONV_40_TOP_H
#define MBCONV_40_TOP_H

#include "fpga_types.h"

// in_ch=96, out_ch=128, expand=4  →  hid=384
#define MBCONV40_IN_CH       C2_CH           // 96
#define MBCONV40_OUT_CH      C3_CH           // 128
#define MBCONV40_HID         384             // 96 * 4
// stride=2: output spatial shrinks to C3_H × C3_W = 20×20
#define MBCONV40_IN_ELEMS    (MBCONV40_IN_CH  * C2_H * C2_W)   // 96*40*40 = 153,600
#define MBCONV40_OUT_ELEMS   (MBCONV40_OUT_CH * C3_H * C3_W)   // 128*20*20 = 51,200
#define MBCONV40_WCONV_ELEMS (MBCONV40_HID * MBCONV40_IN_CH     \
                            + MBCONV40_HID * 9                  \
                            + MBCONV40_OUT_CH * MBCONV40_HID)   // 89,472
#define MBCONV40_WMETA_ELEMS ((MBCONV40_HID + MBCONV40_HID)     \
                            + (MBCONV40_HID + MBCONV40_HID)     \
                            + (MBCONV40_OUT_CH + MBCONV40_OUT_CH)) // 1,792

void mbconv_40_top(
    const act_t    in     [MBCONV40_IN_ELEMS],
          act_t    out    [MBCONV40_OUT_ELEMS],
    const weight_t w_conv [MBCONV40_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV40_WMETA_ELEMS]
);

#endif // MBCONV_40_TOP_H
