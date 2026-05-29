/*
 * mbconv_80_top.h
 * Declares the HLS top function for isolated C-synthesis of mbconv_80.
 *
 * Fixed spatial: H=C1_H=80, W=C1_W=80  (PTQ INT8, W8A32, 320×320 input)
 * Test config  : in_ch=64, out_ch=64, expand=4, stride=1
 *
 * Weight layout (coff stride):
 *   w_conv[0 .. hid*in_ch-1]       expand 1x1 kernel  (int8, 16384 elem)
 *   w_conv[16384 .. +hid*9-1]      dw 3x3 kernel      (int8,  2304 elem)
 *   w_conv[18688 .. +out*hid-1]    proj 1x1 kernel    (int8, 16384 elem)
 *   Total: MBCONV80_WCONV_ELEMS = 35072
 *
 * Meta layout (moff stride):
 *   w_meta[0    .. hid-1]          expand BN scale    (float, 256 elem)
 *   w_meta[256  .. 511]            expand BN bias     (float, 256 elem)
 *   w_meta[512  .. 767]            dw     BN scale    (float, 256 elem)
 *   w_meta[768  .. 1023]           dw     BN bias     (float, 256 elem)
 *   w_meta[1024 .. 1087]           proj   BN scale    (float,  64 elem)
 *   w_meta[1088 .. 1151]           proj   BN bias     (float,  64 elem)
 *   Total: MBCONV80_WMETA_ELEMS = 1152
 */

#ifndef MBCONV_80_TOP_H
#define MBCONV_80_TOP_H

#include "fpga_types.h"

// in_ch=64, out_ch=64, expand=4  →  hid=256
#define MBCONV80_HID         256
#define MBCONV80_IN_CH       C1_CH          // 64
#define MBCONV80_OUT_CH      C1_CH          // 64
#define MBCONV80_IN_ELEMS    (MBCONV80_IN_CH  * C1_H * C1_W)  // 64*160*160 = 1,638,400
#define MBCONV80_OUT_ELEMS   (MBCONV80_OUT_CH * C1_H * C1_W)  // 64*160*160 = 1,638,400
#define MBCONV80_WCONV_ELEMS (MBCONV80_HID * MBCONV80_IN_CH      \
                            + MBCONV80_HID * 9                    \
                            + MBCONV80_OUT_CH * MBCONV80_HID)     // 35,072
#define MBCONV80_WMETA_ELEMS ((MBCONV80_HID + MBCONV80_HID)       \
                            + (MBCONV80_HID + MBCONV80_HID)       \
                            + (MBCONV80_OUT_CH + MBCONV80_OUT_CH)) // 1,152

void mbconv_80_top(
    const act_t    in     [MBCONV80_IN_ELEMS],
          act_t    out    [MBCONV80_OUT_ELEMS],
    const weight_t w_conv [MBCONV80_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV80_WMETA_ELEMS]
);

#endif // MBCONV_80_TOP_H
