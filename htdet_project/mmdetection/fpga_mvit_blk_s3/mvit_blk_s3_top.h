/*
 * mvit_blk_s3_top.h
 * HLS top for the full MobileViT Block Stage-3.
 *
 * in_ch=128 (C3_CH), d=192 (MVIT_S3_DIM), H=W=20 (C3_H/C3_W)
 * depth=4, patch=2, N_patches=100 per view, p²=4 views
 *
 * Weight layout (w_conv, int8):
 *   local   conv3×3 : 128×128×9   = 147,456
 *   proj    1×1     : 192×128     =  24,576
 *   back    1×1     : 128×192     =  24,576
 *   fusion  conv3×3 : 128×256×9  = 294,912
 *   Total wconv                    491,520
 *
 * Weight layout (w_meta, float):
 *   local BN s+b       : 2×128    =    256
 *   proj dequant scale : 192      =    192
 *   transformer × 4    : 4×297,024 = 1,188,096
 *   final LN w+b       : 2×192    =    384
 *   back-proj BN s+b   : 2×128    =    256
 *   fusion BN s+b      : 2×128    =    256
 *   Total wmeta                   1,189,440
 */

#ifndef MVIT_BLK_S3_TOP_H
#define MVIT_BLK_S3_TOP_H

#include "fpga_types.h"

#define MVIT_S3_WCONV_ELEMS   491520
#define MVIT_S3_WMETA_ELEMS   1189440
#define MVIT_S3_IN_ELEMS      (C3_CH * C3_H * C3_W)    // 128*20*20 = 51,200
#define MVIT_S3_OUT_ELEMS     (C3_CH * C3_H * C3_W)    // 51,200

void mvit_blk_s3_top(
    const act_t    in    [MVIT_S3_IN_ELEMS],
          act_t    out   [MVIT_S3_OUT_ELEMS],
    const weight_t w_conv[MVIT_S3_WCONV_ELEMS],
    const meta_t   w_meta[MVIT_S3_WMETA_ELEMS]
);

#endif // MVIT_BLK_S3_TOP_H
