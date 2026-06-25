/*
 * mvit_blk_s2_top.h
 * HLS top for the full MobileViT Block Stage-2.
 *
 * in_ch=96 (C2_CH), d=144 (MVIT_S2_DIM), H=W=40 (C2_H/C2_W)
 * depth=2, patch=2, N_patches=400 (per view), p²=4 views
 *
 * Includes:
 *   local conv3×3(96→96, s=1) + BN + SiLU
 *   proj 1×1(96→144)           + dequant scale
 *   4 views × [DEPTH=2 × transformer_block_s2 + LayerNorm]
 *   back-proj 1×1(144→96)      + BN + SiLU
 *   concat(input, proj_back)
 *   fusion conv3×3(192→96, s=1) + BN + SiLU
 *
 * Weight layout (w_conv, int8):
 *   local   conv3×3 : 96×96×9      =  82,944
 *   proj    1×1     : 144×96        =  13,824
 *   back    1×1     : 96×144        =  13,824
 *   fusion  conv3×3 : 96×192×9     = 165,888
 *   Total wconv                      276,480
 *
 * Weight layout (w_meta, float):
 *   local BN scale+bias : 96+96  =    192
 *   proj dequant scale  : 144    =    144
 *   transformer×2       : 2×167,472 = 334,944
 *   final LN w+b        : 144+144 =   288
 *   back-proj BN s+b    : 96+96  =    192
 *   fusion BN s+b       : 96+96  =    192
 *   Total wmeta                    335,952
 */

#ifndef MVIT_BLK_S2_TOP_H
#define MVIT_BLK_S2_TOP_H

#include "fpga_types.h"

#define MVIT_S2_WCONV_ELEMS   276480
#define MVIT_S2_WMETA_ELEMS   335952
#define MVIT_S2_IN_ELEMS      (C2_CH * C2_H * C2_W)    // 96*40*40 = 153,600
#define MVIT_S2_OUT_ELEMS     (C2_CH * C2_H * C2_W)    // 153,600

void mvit_blk_s2_top(
    const act_t    in    [MVIT_S2_IN_ELEMS],
          act_t    out   [MVIT_S2_OUT_ELEMS],
    const weight_t w_conv[MVIT_S2_WCONV_ELEMS],
    const meta_t   w_meta[MVIT_S2_WMETA_ELEMS]
);

#endif // MVIT_BLK_S2_TOP_H
