/*
 * t_blk_s3_types.h
 * Type aliases and dimension constants for transformer_block_s3 C-synthesis.
 *
 * Full S3 target (HTDet MobileViT stage-3):
 *   in_ch=128, transformer_dim=192, depth=4, patch=2, seq=100 (320x320 input)
 *   Feature map at S3: 20x20, patch=2x2 → N=(20/2)*(20/2)=100 patches
 *   MLP hidden = 2*dim = 384
 *   HEADS=4, HEAD_DIM=48
 *
 * All HLS pragmas proven in S2 (Steps 1-5) apply here unchanged.
 * Key partition factors for S3 (DIM=192):
 *   v_buf: factor=5  — gcd(192,5)=1; (kt*192)%5={0,2,4,1} for kt=0..3, all distinct ✓
 *   a_buf: factor=5  — MHSA_OK_T reads consecutive a_buf indices, always distinct ✓
 *   SM_TILE: 4 (100 divisible by 4 → SM_TILES=25)
 *
 * Scaling ladder — change ONLY the block below and rerun csynth:
 *
 *   Step | DIM | SEQ |HEADS|HD |MLP |DEPTH| BLK_W  | Focus
 *   -----|-----|-----|-----|---|----|-----|--------|-------------------------
 *    1   |  16 |   4 |  2  |  8|  32|  1  |   2224 | Baseline; II tuning
 *    2   |  64 |  16 |  4  | 16| 128|  1  |  33920 | DIM+SEQ scale
 *    3<- | 192 | 100 |  4  | 48| 384|  4  | 297024 | Full S3 target
 *
 * Linear II at full scale: in_dim=192 → II=97; fc2 in_dim=384 → II=193
 * MHSA: SEQ=100 same as S2-Step4 (proven to work).
 *
 * Flags to watch:
 *   - LF_S_LF_OD II: in_dim/2+1 = 97 for DIM=192, 193 for MLP=384
 *   - MHSA_SM: PIPELINE+DEPENDENCE (SEQ=100, same approach as S2-Step4)
 *   - BRAM: weight array 297024×4B=1.14MB, needs many BRAMs — OK on xczu28dr
 *
 * Weight layout (all meta_t / float):
 *   Per block: Q_w|Q_b | K_w|K_b | V_w|V_b | O_w|O_b |
 *              LN1_w|LN1_b | LN2_w|LN2_b | FC1_w|FC1_b | FC2_w|FC2_b
 */

#ifndef T_BLK_S3_TYPES_H
#define T_BLK_S3_TYPES_H

#include <cmath>
#include <cstdint>

typedef float act_t;    // activations (float32, W8A32 scheme)
typedef float meta_t;   // transformer weights (float — not int8)
typedef float acc_t;    // accumulators

// ── Full S3: DIM=192, SEQ=100, HEADS=4, DEPTH=4 ──────────────────────────────
#define TB_DIM      192   // S3 transformer_dim=192
#define TB_SEQ      100   // S3 seq=100 (20x20 feature, patch=2x2 for 320x320 input)
#define TB_HEADS      4   // 4 attention heads → HEAD_DIM=48
#define TB_HEAD_DIM  (TB_DIM / TB_HEADS)      // 48
#define TB_MLP_HID  384   // MLP hidden = 2 * transformer_dim
#define TB_DEPTH      4   // S3 depth=4 transformer blocks
// ─────────────────────────────────────────────────────────────────────────────

// v_buf partition: MHSA_OK_T reads TB_SM_TILE=4 elements per tile at stride DIM.
// DIM=192: gcd(192,5)=1; (kt*192)%5 = {0,2,4,1} for kt=0..3 → all distinct ✓
#define TB_VBUF_FACTOR 5

// SM_SUM tile: TB_SM_TILE must divide TB_SEQ. 100/4=25 ✓
#define TB_SM_TILE    4
#define TB_SM_TILES   (TB_SEQ / TB_SM_TILE)    // 25

// Per-block weight element counts
#define TB_Q_W   (TB_DIM * TB_DIM)            // Q proj weight : 256
#define TB_Q_B    TB_DIM                      // Q proj bias   :  16
#define TB_K_W   (TB_DIM * TB_DIM)            // K proj weight : 256
#define TB_K_B    TB_DIM                      // K proj bias   :  16
#define TB_V_W   (TB_DIM * TB_DIM)            // V proj weight : 256
#define TB_V_B    TB_DIM                      // V proj bias   :  16
#define TB_O_W   (TB_DIM * TB_DIM)            // O proj weight : 256
#define TB_O_B    TB_DIM                      // O proj bias   :  16
#define TB_LN1_W  TB_DIM                      // LN1 scale     :  16
#define TB_LN1_B  TB_DIM                      // LN1 bias      :  16
#define TB_LN2_W  TB_DIM                      // LN2 scale     :  16
#define TB_LN2_B  TB_DIM                      // LN2 bias      :  16
#define TB_FC1_W (TB_MLP_HID * TB_DIM)        // fc1 weight    : 512
#define TB_FC1_B  TB_MLP_HID                  // fc1 bias      :  32
#define TB_FC2_W (TB_DIM    * TB_MLP_HID)     // fc2 weight    : 512
#define TB_FC2_B  TB_DIM                      // fc2 bias      :  16

// Per-block total (S3): 4*(192*192+192) + 4*192 + (384*192+384) + (192*384+192) = 297,024
#define TB_BLK_W_ELEMS ( \
    TB_Q_W + TB_Q_B + TB_K_W + TB_K_B + \
    TB_V_W + TB_V_B + TB_O_W + TB_O_B + \
    TB_LN1_W + TB_LN1_B + TB_LN2_W + TB_LN2_B + \
    TB_FC1_W + TB_FC1_B + TB_FC2_W + TB_FC2_B)

// Total across all depth layers
#define TB_TOTAL_W_ELEMS  (TB_BLK_W_ELEMS * TB_DEPTH)

// Input / output sizes
#define TB_IN_ELEMS   (TB_SEQ * TB_DIM)       // 16 * 64 = 1024
#define TB_OUT_ELEMS  (TB_SEQ * TB_DIM)        // 1024

#endif // T_BLK_S3_TYPES_H
