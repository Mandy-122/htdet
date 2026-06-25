/*
 * fpga_types.h  (PTQ INT8 variant)
 * Data types and model constants for HTDet FPGA (Vitis HLS)
 *
 * Model: RetinaNet with TIMMBackbone(mobilevit_s, out_indices=(1,2,3,4))
 *        + FPN(in=[64,96,128,640], out=192, num_outs=5)
 *        + RetinaHead(num_classes=4, stacked_convs=4)
 * Input: 640x640x3 (CHW, normalized with ImageNet mean/std,
 *        values approx in [-2.1, 2.6])
 *
 * PTQ INT8 weight layout:
 *   weight_t = int8_t   — conv kernel weights (quantized)
 *   meta_t   = float    — BN effective scales, BN biases, transformer weights (all float)
 *   bias_t, act_t, acc_t, score_t, bbox_t stay float
 */

#ifndef FPGA_TYPES_H
#define FPGA_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <cstdint>

// ============================================================
// DATA TYPES (W8A32: INT8 weights, float32 activations/accumulators)
// ============================================================

typedef int8_t  weight_t;   // INT8 conv kernel weights (PTQ)
typedef float   meta_t;     // BN effective scales, BN biases, transformer weights (all float)
typedef float   bias_t;     // convolution biases (float)
typedef float   act_t;      // activations (float32 for W8A32)
typedef float   acc_t;      // accumulators (float32)
typedef float   input_t;    // input image pixels
typedef float   bbox_t;     // bounding box coordinates
typedef float   score_t;    // classification scores


// ============================================================
// INPUT
// Update INPUT_H/INPUT_W here if changing resolution; all
// downstream spatial macros and AXI depths derive from these.
// ============================================================
#define INPUT_C   3
#define INPUT_H   640
#define INPUT_W   640
#define IMAGE_ELEMS  (INPUT_C * INPUT_H * INPUT_W)  // 3×640×640 = 1,228,800

// ============================================================
// MOBILEVIT-S BACKBONE CHANNELS
// TIMM mobilevit_s with out_indices=(1,2,3,4)
// Confirmed by config: in_channels=[64, 96, 128, 640]
// ============================================================

// Internal channels per stage
#define STEM_CH        16
#define STAGE0_CH      32    // Stage 0 (not exported to FPN)

// Exported backbone output channels → FPN inputs
#define C1_CH          64    // out_index=1, stride 4
#define C2_CH          96    // out_index=2, stride 8
#define C3_CH         128    // out_index=3, stride 16
#define C4_CH         640    // out_index=4, stride 32

// Stage 4 internal channel before final expansion conv
#define STAGE4_PRE_CH  160

// MobileViT transformer dimensions (TIMM 1.0.x mobilevit_s, verified from weights)
#define MVIT_S2_DIM    144   // stage2 in_ch=96  → d=144
#define MVIT_S2_DEPTH    2
#define MVIT_S3_DIM    192   // stage3 in_ch=128 → d=192
#define MVIT_S3_DEPTH    4
#define MVIT_S4_DIM    240   // stage4 in_ch=160 → d=240
#define MVIT_S4_DEPTH    3
#define MVIT_HEADS       4   // Attention heads
#define MVIT_PATCH       2   // Patch size p (P = p*p = 4 independent views)

// Patch token counts per stage
#define MVIT_N_S2  ((C2_H / MVIT_PATCH) * (C2_W / MVIT_PATCH))  // 1600 for 640 input
#define MVIT_N_S3  ((C3_H / MVIT_PATCH) * (C3_W / MVIT_PATCH))  // 400 for 640 input
#define MVIT_N_S4  ((C4_H / MVIT_PATCH) * (C4_W / MVIT_PATCH))  // 100 for 640 input
#define MVIT_N_MAX MVIT_N_S2

// MBConv expand ratio
#define MBCONV_EXPAND    4

// ============================================================
// BACKBONE OUTPUT SPATIAL DIMENSIONS  (640x640 input)
// ============================================================
#define STEM_H    (INPUT_H / 2)    // 320
#define STEM_W    (INPUT_W / 2)    // 320

#define C1_H      (INPUT_H / 4)    // 160
#define C1_W      (INPUT_W / 4)    // 160

#define C2_H      (INPUT_H / 8)    // 80
#define C2_W      (INPUT_W / 8)    // 80

#define C3_H      (INPUT_H / 16)   // 40
#define C3_W      (INPUT_W / 16)   // 40

#define C4_H      (INPUT_H / 32)   // 20
#define C4_W      (INPUT_W / 32)   // 20

// ============================================================
// FPN NECK  (in_channels=[64,96,128,640], out_channels=192, num_outs=5)
// ============================================================
#define FPN_OUT_CH    192
#define HEAD_FEAT_CH  192

#define P2_H    C1_H    // 160  (INPUT_H/4)
#define P2_W    C1_W    // 160
#define P3_H    C2_H    // 80   (INPUT_H/8)
#define P3_W    C2_W    // 80
#define P4_H    C3_H    // 40   (INPUT_H/16)
#define P4_W    C3_W    // 40
#define P5_H    C4_H    // 20   (INPUT_H/32)
#define P5_W    C4_W    // 20
#define P6_H    (C4_H / 2)  // 10  (INPUT_H/64)
#define P6_W    (C4_W / 2)  // 10

// Element counts for AXI depth pragmas
#define P2_ELEMS  (FPN_OUT_CH * P2_H * P2_W)   // 192*160*160 = 4,915,200
#define P3_ELEMS  (FPN_OUT_CH * P3_H * P3_W)   // 192*80*80   = 1,228,800
#define P4_ELEMS  (FPN_OUT_CH * P4_H * P4_W)   // 192*40*40   =   307,200
#define P5_ELEMS  (FPN_OUT_CH * P5_H * P5_W)   // 192*20*20   =    76,800
#define P6_ELEMS  (FPN_OUT_CH * P6_H * P6_W)   // 192*10*10   =    19,200

// ============================================================
// RETINANET HEAD
// num_classes=4, stacked_convs=4, feat_channels=192
// anchor scales=[4,6,8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]
// ============================================================

#define HEAD_STACKED_CONVS   4
#define NUM_CLASSES          4   // holothurian, echinus, scallop, starfish (URPC)
#define ANCHOR_SCALES_N      3   // {4, 6, 8}
#define ANCHOR_RATIOS_N      3   // {0.5, 1.0, 2.0}
#define ANCHORS_PER_LOC      (ANCHOR_SCALES_N * ANCHOR_RATIOS_N)   // 9

#define MAX_DETS       100
#define NMS_PRE       1000   // Max proposals kept per level after top-K by score
#define CAND_BUF_SIZE 5000   // Candidate collection buffer per level
#define SCORE_THR     0.20f
#define NMS_IOU_THR   0.5f

// ============================================================
// DETECTION OUTPUT
// ============================================================
typedef struct {
    bbox_t  x1, y1, x2, y2;
    score_t score;
    int     class_id;
} Detection;

// ============================================================
// UTILITY MACROS
// ============================================================
#define DIV_CEIL(a,b)   (((a) + (b) - 1) / (b))
#ifndef MIN
#define MIN(a,b)        ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#endif

#endif // FPGA_TYPES_H
