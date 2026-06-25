/*
 * stem_top.h
 * HLS top for the MobileViT-S stem conv:
 *   Conv3×3(3→16, stride=2, 320×320) + BN + SiLU
 * Input : image[3, 320, 320]  (CHW, float — reinterpret_cast of input_t)
 * Output: stem_out[16, 160, 160]  (CHW, float)
 *
 * Weight layout (w_conv, int8):
 *   [0 .. STEM_CH*INPUT_C*9-1]  kernel  16×3×3×3 = 432 int8
 *
 * Meta layout (w_meta, float):
 *   [0        .. STEM_CH-1]        BN effective scale  [16]
 *   [STEM_CH  .. 2*STEM_CH-1]      BN effective bias   [16]
 *   Total: 32 float
 */

#ifndef STEM_TOP_H
#define STEM_TOP_H

#include "fpga_types.h"

#define STEM_WCONV_ELEMS  (STEM_CH * INPUT_C * 9)            // 432
#define STEM_WMETA_ELEMS  (STEM_CH + STEM_CH)                // 32
#define STEM_IN_ELEMS     (INPUT_C  * INPUT_H * INPUT_W)     // 307,200
#define STEM_OUT_ELEMS    (STEM_CH  * STEM_H  * STEM_W)      // 409,600

void stem_top(
    const act_t    in    [STEM_IN_ELEMS],
          act_t    out   [STEM_OUT_ELEMS],
    const weight_t w_conv[STEM_WCONV_ELEMS],
    const meta_t   w_meta[STEM_WMETA_ELEMS]
);

#endif // STEM_TOP_H
