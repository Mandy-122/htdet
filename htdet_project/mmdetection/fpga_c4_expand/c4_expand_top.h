/*
 * c4_expand_top.h
 * HLS top: final backbone expansion conv1×1(160→640, 10×10) + BN + SiLU
 * Input : s4_mvit[160, 10, 10]  (CHW, from mobilevit_block_s4)
 * Output: c4     [640, 10, 10]  (CHW, backbone C4 output)
 *
 * Weight layout (w_conv, int8):
 *   conv1×1 kernel : [640 * 160]  = 102,400 int8
 *
 * Meta layout (w_meta, float):
 *   BN effective scale : [640]
 *   BN effective bias  : [640]
 *   Total: 1,280 float
 */

#ifndef C4_EXPAND_TOP_H
#define C4_EXPAND_TOP_H

#include "fpga_types.h"

#define C4EXP_IN_CH      STAGE4_PRE_CH      // 160
#define C4EXP_OUT_CH     C4_CH              // 640

#define C4EXP_IN_ELEMS   (C4EXP_IN_CH  * C4_H * C4_W)   // 160*10*10 = 16,000
#define C4EXP_OUT_ELEMS  (C4EXP_OUT_CH * C4_H * C4_W)   // 640*10*10 = 64,000
#define C4EXP_WCONV_ELEMS (C4EXP_OUT_CH * C4EXP_IN_CH)  // 102,400
#define C4EXP_WMETA_ELEMS (C4EXP_OUT_CH + C4EXP_OUT_CH) // 1,280

void c4_expand_top(
    const act_t    in    [C4EXP_IN_ELEMS],
          act_t    out   [C4EXP_OUT_ELEMS],
    const weight_t w_conv[C4EXP_WCONV_ELEMS],
    const meta_t   w_meta[C4EXP_WMETA_ELEMS]
);

#endif // C4_EXPAND_TOP_H
