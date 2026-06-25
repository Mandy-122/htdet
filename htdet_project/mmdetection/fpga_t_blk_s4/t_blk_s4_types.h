/*
 * t_blk_s4_types.h
 * Type aliases and dimension constants for transformer_block_s4 C-synthesis.
 *
 * Full S4 target (HTDet MobileViT stage-4):
 *   in_ch=160, transformer_dim=240, depth=3, patch=2, seq=25 (320x320 input)
 *   Feature map at S4: 10x10, patch=2x2 -> N=(10/2)*(10/2)=25 patches
 *   MLP hidden = 2*dim = 480, HEADS=4, HEAD_DIM=60
 *
 * KEY DIFFERENCE from S2/S3: DIM=240=2^4*3*5, so gcd(DIM,5)=5 != 1.
 *   v_buf: factor=7  -- gcd(240,7)=1; (kt*240)%7={0,2,4,6,1} for kt=0..4 distinct
 *   SM_TILE: 5  (25 not divisible by 4; 25/5=5 tiles OK)
 *   a_buf: factor=5  -- a_buf accesses use seq indices, not DIM, so factor=5 is fine
 *
 * Scaling ladder -- change ONLY the block below and rerun csynth:
 *
 *   Step | DIM | SEQ |HEADS|HD |MLP |DEPTH| BLK_W  | Focus
 *   -----|-----|-----|-----|---|----|-----|--------|----------------------
 *    1   |  16 |   4 |  2  |  8|  32|  1  |   2224 | Baseline
 *    2<- | 240 |  25 |  4  | 60| 480|  3  | Full S4 (SEQ=25 is tiny!)
 *
 * Linear II: in_dim=240 -> II=121; fc2 in_dim=480 -> II=241
 * MHSA: SEQ=25 tiny -- MHSA_SM II~5, total MHSA < 100K cycles per depth
 * Synthesis time estimate: 20-40 min (small SEQ, simple address decode)
 *
 * Weight layout (all meta_t / float):
 *   Per block: Q_w|Q_b | K_w|K_b | V_w|V_b | O_w|O_b |
 *              LN1_w|LN1_b | LN2_w|LN2_b | FC1_w|FC1_b | FC2_w|FC2_b
 */

#ifndef T_BLK_S4_TYPES_H
#define T_BLK_S4_TYPES_H

#include <cmath>
#include <cstdint>

typedef float act_t;    // activations (float32, W8A32 scheme)
typedef float meta_t;   // transformer weights (float -- not int8)
typedef float acc_t;    // accumulators

// ── Full S4: DIM=240, SEQ=25, HEADS=4, DEPTH=3 ───────────────────────────────
#define TB_DIM      240   // S4 transformer_dim=240
#define TB_SEQ       25   // S4 seq=25 (10x10 feature, patch=2x2, 320x320 input)
#define TB_HEADS      4   // 4 attention heads -> HEAD_DIM=60
#define TB_HEAD_DIM  (TB_DIM / TB_HEADS)       // 60
#define TB_MLP_HID  480   // MLP hidden = 2 * transformer_dim
#define TB_DEPTH      3   // S4 depth=3 transformer blocks
// ─────────────────────────────────────────────────────────────────────────────

// v_buf partition: MHSA_OK_T reads TB_SM_TILE=5 elements per tile at stride DIM.
// DIM=240: gcd(240,5)=5 (CANNOT use 5!), gcd(240,7)=1.
// (kt*240)%7 = {0,2,4,6,1} for kt=0..4 -- all distinct with factor=7 ✓
#define TB_VBUF_FACTOR 7

// SM_SUM tile: TB_SM_TILE must divide TB_SEQ. 25/5=5 ✓ (25 not divisible by 4)
// Also check v_buf: 5 reads at stride DIM=240, factor=7: {0,240,480,720,960}%7={0,2,4,6,1} distinct ✓
#define TB_SM_TILE    5
#define TB_SM_TILES   (TB_SEQ / TB_SM_TILE)    // 5

// Per-block weight element counts
#define TB_Q_W   (TB_DIM * TB_DIM)
#define TB_Q_B    TB_DIM
#define TB_K_W   (TB_DIM * TB_DIM)
#define TB_K_B    TB_DIM
#define TB_V_W   (TB_DIM * TB_DIM)
#define TB_V_B    TB_DIM
#define TB_O_W   (TB_DIM * TB_DIM)
#define TB_O_B    TB_DIM
#define TB_LN1_W  TB_DIM
#define TB_LN1_B  TB_DIM
#define TB_LN2_W  TB_DIM
#define TB_LN2_B  TB_DIM
#define TB_FC1_W (TB_MLP_HID * TB_DIM)
#define TB_FC1_B  TB_MLP_HID
#define TB_FC2_W (TB_DIM    * TB_MLP_HID)
#define TB_FC2_B  TB_DIM

// Per-block total: 4*(240*240+240) + 4*240 + (480*240+480) + (240*480+240) = 463,440
#define TB_BLK_W_ELEMS ( \
    TB_Q_W + TB_Q_B + TB_K_W + TB_K_B + \
    TB_V_W + TB_V_B + TB_O_W + TB_O_B + \
    TB_LN1_W + TB_LN1_B + TB_LN2_W + TB_LN2_B + \
    TB_FC1_W + TB_FC1_B + TB_FC2_W + TB_FC2_B)

#define TB_TOTAL_W_ELEMS  (TB_BLK_W_ELEMS * TB_DEPTH)
#define TB_IN_ELEMS   (TB_SEQ * TB_DIM)
#define TB_OUT_ELEMS  (TB_SEQ * TB_DIM)

#endif // T_BLK_S4_TYPES_H
