/*
 * t_blk_s2_top.h
 * HLS top-function declaration for isolated C-synthesis of transformer_block_s2.
 *
 * Corresponds to MobileViTBlock.transformer (the TransformerBlock stack) at
 * stage-2 of HTDet MobileViT.  All weights are float (meta_t); no int8 here.
 *
 * Weight flat-array layout (per block, then repeated TB_DEPTH times):
 *
 *   Offset              | Field        | Shape         | #elems
 *   --------------------|--------------|---------------|-------
 *   BLK_Q_W_OFF  =    0 | Q proj w     | [DIM, DIM]    |  256
 *   BLK_Q_B_OFF  =  256 | Q proj b     | [DIM]         |   16
 *   BLK_K_W_OFF  =  272 | K proj w     | [DIM, DIM]    |  256
 *   BLK_K_B_OFF  =  528 | K proj b     | [DIM]         |   16
 *   BLK_V_W_OFF  =  544 | V proj w     | [DIM, DIM]    |  256
 *   BLK_V_B_OFF  =  800 | V proj b     | [DIM]         |   16
 *   BLK_O_W_OFF  =  816 | O proj w     | [DIM, DIM]    |  256
 *   BLK_O_B_OFF  = 1072 | O proj b     | [DIM]         |   16
 *   BLK_LN1_W_OFF= 1088 | LN1 scale    | [DIM]         |   16
 *   BLK_LN1_B_OFF= 1104 | LN1 bias     | [DIM]         |   16
 *   BLK_LN2_W_OFF= 1120 | LN2 scale    | [DIM]         |   16
 *   BLK_LN2_B_OFF= 1136 | LN2 bias     | [DIM]         |   16
 *   BLK_FC1_W_OFF= 1152 | MLP fc1 w    | [MLP_HID,DIM] |  512
 *   BLK_FC1_B_OFF= 1664 | MLP fc1 b    | [MLP_HID]     |   32
 *   BLK_FC2_W_OFF= 1696 | MLP fc2 w    | [DIM,MLP_HID] |  512
 *   BLK_FC2_B_OFF= 2208 | MLP fc2 b    | [DIM]         |   16
 *   --  per-block total: 2224 floats
 */

#ifndef T_BLK_S2_TOP_H
#define T_BLK_S2_TOP_H

#include "t_blk_s2_types.h"

// ── Weight offsets within one block (derived, do not edit manually) ──────────
#define BLK_Q_W_OFF    0
#define BLK_Q_B_OFF   (BLK_Q_W_OFF   + TB_Q_W)
#define BLK_K_W_OFF   (BLK_Q_B_OFF   + TB_Q_B)
#define BLK_K_B_OFF   (BLK_K_W_OFF   + TB_K_W)
#define BLK_V_W_OFF   (BLK_K_B_OFF   + TB_K_B)
#define BLK_V_B_OFF   (BLK_V_W_OFF   + TB_V_W)
#define BLK_O_W_OFF   (BLK_V_B_OFF   + TB_V_B)
#define BLK_O_B_OFF   (BLK_O_W_OFF   + TB_O_W)
#define BLK_LN1_W_OFF (BLK_O_B_OFF   + TB_O_B)
#define BLK_LN1_B_OFF (BLK_LN1_W_OFF + TB_LN1_W)
#define BLK_LN2_W_OFF (BLK_LN1_B_OFF + TB_LN1_B)
#define BLK_LN2_B_OFF (BLK_LN2_W_OFF + TB_LN2_W)
#define BLK_FC1_W_OFF (BLK_LN2_B_OFF + TB_LN2_B)
#define BLK_FC1_B_OFF (BLK_FC1_W_OFF + TB_FC1_W)
#define BLK_FC2_W_OFF (BLK_FC1_B_OFF + TB_FC1_B)
#define BLK_FC2_B_OFF (BLK_FC2_W_OFF + TB_FC2_W)
// BLK_FC2_B_OFF + TB_FC2_B == TB_BLK_W_ELEMS == 2224  ✓

// ── Top function ─────────────────────────────────────────────────────────────
// in/out : [TB_SEQ * TB_DIM]  (token sequence, row-major: token s at in[s*DIM])
// w      : [TB_TOTAL_W_ELEMS] (all block weights concatenated)
void transformer_blk_s2_top(
    const act_t  in  [TB_IN_ELEMS],
          act_t  out [TB_OUT_ELEMS],
    const meta_t w   [TB_TOTAL_W_ELEMS]
);

#endif // T_BLK_S2_TOP_H
