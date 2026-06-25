/*
 * t_blk_s2_types.h
 * Type aliases and dimension constants for transformer_block_s2 C-synthesis.
 *
 * Full S2 target (HTDet MobileViT stage-2):
 *   in_ch=96, transformer_dim=144, depth=2, patch=2, seq=400 (320×320 input)
 *
 * Scaling ladder — change ONLY the block below and rerun csynth:
 *
 *   Step | DIM | SEQ |HEADS|HD |MLP |DEPTH| BLK_W  | Focus
 *   -----|-----|-----|-----|---|----|-----|--------|-------------------------
 *    1   |  16 |   4 |  2  |  8|  32|  1  |   2224 | Baseline; II tuning
 *    2   |  32 |   4 |  2  | 16|  64|  1  |   8544 | DIM scale; linear II
 *    3   |  64 |  16 |  4  | 16| 128|  1  |  33920 | SEQ+HEADS; SM_EXP PIPELINE
 *    4<- | 144 | 100 |  4  | 36| 288|  1  | 167472 | Full DIM; MHSA_SM PIPELINE
 *    5   | 144 | 400 |  4  | 36| 288|  2  | 334944 | Full S2
 *
 * Flags to watch at each step:
 *   - LF_S_LF_OD II: grows as in_dim/2+1 (9->17->33->73)
 *   - Timing slack on LF_S_LF_OD: -0.21 ns at Step 1 -- gets tighter
 *   - MHSA_QI_MHSA_KI latency: O(SEQ^2) -- explodes at Step 3+
 *   - MHSA_SM: UNROLL at Step 1-3; PIPELINE+DEPENDENCE at Step 4+ (SEQ>16)
 *   - BRAM: at Step 4+ weight array >single BRAM; check memory report
 *
 * Weight layout (all meta_t / float):
 *   Per block: Q_w|Q_b | K_w|K_b | V_w|V_b | O_w|O_b |
 *              LN1_w|LN1_b | LN2_w|LN2_b | FC1_w|FC1_b | FC2_w|FC2_b
 */

#ifndef T_BLK_S2_TYPES_H
#define T_BLK_S2_TYPES_H

#include <cmath>
#include <cstdint>

typedef float act_t;    // activations (float32, W8A32 scheme)
typedef float meta_t;   // transformer weights (float — not int8)
typedef float acc_t;    // accumulators

// ── Step 5 (FULL S2): DIM=144, SEQ=400, DEPTH=2 ─────────────────────────────
#define TB_DIM      144   // Step1=16 → Step2=32 → Step3=64 → Step4/5=144
#define TB_SEQ      400   // Step1/2=4 → Step3=16 → Step4=100 → Step5=400
#define TB_HEADS      4   // Step1/2=2 → Step3/4/5=4
#define TB_HEAD_DIM  (TB_DIM / TB_HEADS)      // 36  (full S2: 36)
#define TB_MLP_HID  288   // Step1=32 → Step2=64 → Step3=128 → Step4/5=288
#define TB_DEPTH      2   // Step5 full S2 depth=2
// ─────────────────────────────────────────────────────────────────────────────

// v_buf partition factor for MHSA_OQ_MHSA_OD:
//   Reads TB_SEQ elements at stride TB_DIM (ki=0..SEQ-1, unrolled).
//   Need {0, DIM, 2*DIM, ..., (SEQ-1)*DIM} % factor all distinct.
//   Use smallest prime > TB_SEQ with gcd(TB_DIM, prime) = 1.
//   TB_SEQ+1 is prime for all our steps and satisfies gcd condition:
//     Step1: 4+1=5   → {0,16,32,48}%5={0,1,2,3}     ✓ for DIM=16
//     Step2: 4+1=5   → {0,32,64,96}%5={0,2,4,1}     ✓ for DIM=32
//     Step3: 16+1=17 → all 16 ki*64 mod 17 distinct  ✓ for DIM=64
//     Step4: 100+1=101 → all 100 ki*144 mod 101 distinct ✓
//     Step5: 400+1=401 → same                          ✓
// v_buf partition: MHSA_OK_T reads 4 elements per tile at stride DIM.
// Need 4 consecutive ki*DIM values to hit distinct banks.
// With DIM=144 and factor=5: (kt*144)%5 = {0,4,3,2} for kt=0..3 → all distinct ✓
// Factor=5 works for all steps with tile-based MHSA_OK_T (replaces old TB_SEQ+1 formula
// which was for full-UNROLL MHSA_OK; with tile=4 we only need 4 unique banks, not SEQ+1).
#define TB_VBUF_FACTOR 5                       // Fixed prime; sufficient for tile-based MHSA_OK_T

// Tile size for SM_SUM (must divide TB_SEQ evenly: 4/16/100/400 all divisible by 4)
#define TB_SM_TILE    4
#define TB_SM_TILES   (TB_SEQ / TB_SM_TILE)    // 25 for Step4

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

// Per-block total (Step3): 4*(64²+64) + 4*64 + (128*64+128) + (64*128+64) ≈ 33,920
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

#endif // T_BLK_S2_TYPES_H
