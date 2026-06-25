/*
 * fpga_types.h  (PTQ INT8 variant — shared with fpga_mbconv_80/40/20)
 * Data types and model constants for HTDet FPGA (Vitis HLS)
 *
 * Model: RetinaNet with TIMMBackbone(mobilevit_s, out_indices=(1,2,3,4))
 *        + FPN(in=[64,96,128,640], out=192, num_outs=5)
 *        + RetinaHead(num_classes=4, stacked_convs=4)
 * Input: 320x320x3 (CHW, normalized with ImageNet mean/std)
 *
 * PTQ INT8:  weight_t = int8_t  (conv kernels)
 *            meta_t   = float   (BN scales, BN biases, plain biases, transformer weights)
 */

#ifndef FPGA_TYPES_H
#define FPGA_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <cstdint>

// ============================================================
// DATA TYPES
// ============================================================
typedef int8_t  weight_t;   // INT8 conv kernel weights
typedef float   meta_t;     // BN scales, biases (float)
typedef float   bias_t;     // convolution biases
typedef float   act_t;      // activations (float, W8A32)
typedef float   acc_t;      // accumulators
typedef float   input_t;    // image pixels
typedef float   bbox_t;     // bounding box coords
typedef float   score_t;    // classification scores

// ============================================================
// INPUT  (320×320)
// ============================================================
#define INPUT_C   3
#define INPUT_H   320
#define INPUT_W   320
#define IMAGE_ELEMS  (INPUT_C * INPUT_H * INPUT_W)   // 307,200

// ============================================================
// BACKBONE CHANNEL WIDTHS  (MobileViT-S, TIMM out_indices=(1,2,3,4))
// ============================================================
#define STEM_CH        16
#define STAGE0_CH      32

#define C1_CH          64    // stride-4  output
#define C2_CH          96    // stride-8  output
#define C3_CH         128    // stride-16 output
#define C4_CH         640    // stride-32 output

#define STAGE4_PRE_CH  160

// MobileViT transformer dims (verified from checkpoint)
#define MVIT_S2_DIM    144
#define MVIT_S2_DEPTH    2
#define MVIT_S3_DIM    192
#define MVIT_S3_DEPTH    4
#define MVIT_S4_DIM    240
#define MVIT_S4_DEPTH    3
#define MVIT_HEADS       4
#define MVIT_PATCH       2

#define MBCONV_EXPAND    4

// ============================================================
// BACKBONE SPATIAL DIMENSIONS  (320×320 input)
// ============================================================
#define STEM_H    (INPUT_H / 2)    // 160
#define STEM_W    (INPUT_W / 2)    // 160

#define C1_H      (INPUT_H / 4)    //  80
#define C1_W      (INPUT_W / 4)    //  80

#define C2_H      (INPUT_H / 8)    //  40
#define C2_W      (INPUT_W / 8)    //  40

#define C3_H      (INPUT_H / 16)   //  20
#define C3_W      (INPUT_W / 16)   //  20

#define C4_H      (INPUT_H / 32)   //  10
#define C4_W      (INPUT_W / 32)   //  10

// Patch token counts
#define MVIT_N_S2  ((C2_H / MVIT_PATCH) * (C2_W / MVIT_PATCH))  // 400
#define MVIT_N_S3  ((C3_H / MVIT_PATCH) * (C3_W / MVIT_PATCH))  // 100
#define MVIT_N_S4  ((C4_H / MVIT_PATCH) * (C4_W / MVIT_PATCH))  // 25
#define MVIT_N_MAX MVIT_N_S2

// ============================================================
// FPN NECK  (out_channels=192, num_outs=5)
// ============================================================
#define FPN_OUT_CH    192
#define HEAD_FEAT_CH  192

// FPN output spatial dims — same as backbone strides
#define P2_H    C1_H    //  80
#define P2_W    C1_W    //  80
#define P3_H    C2_H    //  40
#define P3_W    C2_W    //  40
#define P4_H    C3_H    //  20
#define P4_W    C3_W    //  20
#define P5_H    C4_H    //  10
#define P5_W    C4_W    //  10
#define P6_H    (C4_H / 2)   //   5
#define P6_W    (C4_W / 2)   //   5

// Element counts per FPN level
#define P2_ELEMS  (FPN_OUT_CH * P2_H * P2_W)   // 192*80*80 = 1,228,800
#define P3_ELEMS  (FPN_OUT_CH * P3_H * P3_W)   // 192*40*40 =   307,200
#define P4_ELEMS  (FPN_OUT_CH * P4_H * P4_W)   // 192*20*20 =    76,800
#define P5_ELEMS  (FPN_OUT_CH * P5_H * P5_W)   // 192*10*10 =    19,200
#define P6_ELEMS  (FPN_OUT_CH * P6_H * P6_W)   // 192* 5* 5 =     4,800

// ============================================================
// RETINANET HEAD
// ============================================================
#define HEAD_STACKED_CONVS   4
#define NUM_CLASSES          4
#define ANCHOR_SCALES_N      3
#define ANCHOR_RATIOS_N      3
#define ANCHORS_PER_LOC      (ANCHOR_SCALES_N * ANCHOR_RATIOS_N)   // 9

#define MAX_DETS       100
#define NMS_PRE       1000
#define CAND_BUF_SIZE 5000
#define SCORE_THR     0.20f
#define NMS_IOU_THR   0.5f

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
#define MIN(a,b)  ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b)  ((a) > (b) ? (a) : (b))
#endif

#endif // FPGA_TYPES_H
