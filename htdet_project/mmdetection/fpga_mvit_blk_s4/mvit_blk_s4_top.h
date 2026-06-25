/*
 * mvit_blk_s4_top.h
 * HLS top for the full MobileViT Block Stage-4.
 *
 * in_ch=160 (STAGE4_PRE_CH), d=240 (MVIT_S4_DIM), H=W=10 (C4_H/C4_W)
 * depth=3, patch=2, N_patches=25 per view, p²=4 views
 *
 * Weight layout (w_conv, int8):
 *   local   conv3×3 : 160×160×9   = 230,400
 *   proj    1×1     : 240×160     =  38,400
 *   back    1×1     : 160×240     =  38,400
 *   fusion  conv3×3 : 160×320×9  = 460,800
 *   Total wconv                    768,000
 *
 * Weight layout (w_meta, float):
 *   local BN s+b       : 2×160    =    320
 *   proj dequant scale : 240      =    240
 *   transformer × 3    : 3×463,440 = 1,390,320
 *   final LN w+b       : 2×240    =    480
 *   back-proj BN s+b   : 2×160    =    320
 *   fusion BN s+b      : 2×160    =    320
 *   Total wmeta                   1,392,000
 */

#ifndef MVIT_BLK_S4_TOP_H
#define MVIT_BLK_S4_TOP_H

#include "fpga_types.h"

#define MVIT_S4_WCONV_ELEMS   768000
#define MVIT_S4_WMETA_ELEMS   1392000
#define MVIT_S4_IN_ELEMS      (STAGE4_PRE_CH * C4_H * C4_W)   // 160*10*10 = 16,000
#define MVIT_S4_OUT_ELEMS     (STAGE4_PRE_CH * C4_H * C4_W)   // 16,000

void mvit_blk_s4_top(
    const act_t    in    [MVIT_S4_IN_ELEMS],
          act_t    out   [MVIT_S4_OUT_ELEMS],
    const weight_t w_conv[MVIT_S4_WCONV_ELEMS],
    const meta_t   w_meta[MVIT_S4_WMETA_ELEMS]
);

#endif // MVIT_BLK_S4_TOP_H
