/*
 * fpga_types.h
 * Data types and model constants for HTDet FPGA (Vitis HLS)
 *
 * Model: RetinaNet with TIMMBackbone(mobilevit_s, out_indices=(1,2,3,4))
 *        + FPN(in=[64,96,128,640], out=256, num_outs=5)
 *        + RetinaHead(num_classes=4, stacked_convs=4)
 * Input: 640x640x3 (CHW, normalized with ImageNet mean/std,
 *        values approx in [-2.1, 2.6])
 */

#ifndef FPGA_TYPES_H
#define FPGA_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// ============================================================
// FIXED-POINT PRECISION
// ============================================================

//// Q8.8  – weights/biases: range ±127, step 1/256
//typedef ap_fixed<16,  8, AP_RND, AP_SAT> weight_t;
//typedef ap_fixed<16,  8, AP_RND, AP_SAT> bias_t;
//
//// Q12.4 – activations: range ±2047, step 1/16
//typedef ap_fixed<16, 12, AP_RND, AP_SAT> act_t;
//
//// Q16.16 – accumulators: prevents overflow during MAC
//typedef ap_fixed<32, 16, AP_RND, AP_SAT> acc_t;
//
//// Q8.8  – input image pixels (values in [-1, 1])
//typedef ap_fixed<16,  8, AP_RND, AP_SAT> input_t;
//
//// Q16.8 – bounding box coordinates (pixel space up to ~32767)
//typedef ap_fixed<24, 16, AP_RND, AP_SAT> bbox_t;
//
//// Q4.12 – classification scores in [0, 1]
//typedef ap_fixed<16,  4, AP_RND, AP_SAT> score_t;



// ============================================================
// PTQ-DERIVED PRECISION TABLE  (W8A32 calibration, 200 URPC val images)
// Source: fpga_support/ptq_calibrate.py  →  ptq_results/ptq_report.txt
//
// W8A32 result: +3.3% detection delta vs float (negligible).
// All 53 Conv2d layers: SQNR 36.8–49.1 dB, CosSim ≥ 0.9999.
//
// For W8A8 (INT8 weights + INT8 activations) use:
//   ap_fixed<8, INT_BITS, AP_RND, AP_SAT>
//   INT_BITS = ceil(log2(ActMax+1)) + 1  (sign bit included)
//   Fractional bits = 8 − INT_BITS
//
//  Layer                                    ActMax  INT_BITS  ap_fixed<8,N>
//  ------------------------------------------------------------------------
//  backbone.model.stem.conv                   12.1     5      ap_fixed<8,5>
//  backbone.model.stages_0.0.conv1_1x1        18.6     6      ap_fixed<8,6>
//  backbone.model.stages_0.0.conv2_kxk (DW) 232.0     9      ap_fixed<8,9> (*)
//  backbone.model.stages_0.0.conv3_1x1        50.4     7      ap_fixed<8,7>
//  backbone.model.stages_1.0.conv1_1x1        27.6     6      ap_fixed<8,6>
//  backbone.model.stages_1.0.conv2_kxk (DW) 196.0     9      ap_fixed<8,9> (*)
//  backbone.model.stages_1.0.conv3_1x1        15.0     5      ap_fixed<8,5>
//  backbone.model.stages_1.1.conv1_1x1        13.6     5      ap_fixed<8,5>
//  backbone.model.stages_1.1.conv2_kxk        34.4     7      ap_fixed<8,7>
//  backbone.model.stages_1.1.conv3_1x1        15.9     6      ap_fixed<8,6>
//  backbone.model.stages_1.2.conv1_1x1        13.1     5      ap_fixed<8,5>
//  backbone.model.stages_1.2.conv2_kxk        20.0     6      ap_fixed<8,6>
//  backbone.model.stages_1.2.conv3_1x1         8.1     5      ap_fixed<8,5>
//  backbone.model.stages_2.0.conv1_1x1        17.0     6      ap_fixed<8,6>
//  backbone.model.stages_2.0.conv2_kxk        11.9     5      ap_fixed<8,5>
//  backbone.model.stages_2.0.conv3_1x1         8.2     5      ap_fixed<8,5>
//  backbone.model.stages_2.1.conv_kxk         23.5     6      ap_fixed<8,6>
//  backbone.model.stages_2.1.conv_1x1         10.9     5      ap_fixed<8,5>
//  backbone.model.stages_2.1.conv_proj         2.9     3      ap_fixed<8,3>
//  backbone.model.stages_2.1.conv_fusion      25.6     6      ap_fixed<8,6>
//  backbone.model.stages_3.0.conv1_1x1         5.5     4      ap_fixed<8,4>
//  backbone.model.stages_3.0.conv2_kxk         7.9     5      ap_fixed<8,5>
//  backbone.model.stages_3.0.conv3_1x1         9.9     5      ap_fixed<8,5>
//  backbone.model.stages_3.1.conv_kxk         22.0     6      ap_fixed<8,6>
//  backbone.model.stages_3.1.conv_1x1         10.3     5      ap_fixed<8,5>
//  backbone.model.stages_3.1.conv_proj         3.3     4      ap_fixed<8,4>
//  backbone.model.stages_3.1.conv_fusion      27.1     6      ap_fixed<8,6>
//  backbone.model.stages_4.0.conv1_1x1         6.6     4      ap_fixed<8,4>
//  backbone.model.stages_4.0.conv2_kxk         6.8     4      ap_fixed<8,4>
//  backbone.model.stages_4.0.conv3_1x1         8.7     5      ap_fixed<8,5>
//  backbone.model.stages_4.1.conv_kxk         23.2     6      ap_fixed<8,6>
//  backbone.model.stages_4.1.conv_1x1         12.8     5      ap_fixed<8,5>
//  backbone.model.stages_4.1.conv_proj         4.1     4      ap_fixed<8,4>
//  backbone.model.stages_4.1.conv_fusion      21.4     6      ap_fixed<8,6>
//  backbone.model.final_conv                   3.6     4      ap_fixed<8,4>
//  neck.lateral_convs.0.conv                   5.9     4      ap_fixed<8,4>
//  neck.lateral_convs.1.conv                   1.8     3      ap_fixed<8,3>
//  neck.lateral_convs.2.conv                   2.2     3      ap_fixed<8,3>
//  neck.lateral_convs.3.conv                   3.9     4      ap_fixed<8,4>
//  neck.fpn_convs.0..3.conv                2.4–6.4   3–4     ap_fixed<8,4>
//  bbox_head.cls_convs.0..3.conv           4.7–11.8  4–5     ap_fixed<8,5>
//  bbox_head.reg_convs.0..3.conv           0.7–8.2   2–5     ap_fixed<8,5>
//  bbox_head.retina_cls                       21.5     6      ap_fixed<8,6>
//  bbox_head.retina_reg                        0.9     2      ap_fixed<8,2>
//
//  (*) Early depthwise convs (stages_0.0, stages_1.0 conv2_kxk) reach ±232/±196
//      before BN. These need 9 integer bits, leaving only 1 fractional bit in 8b.
//      Consider per-channel input clipping or keeping these two layers in FP32.
// ============================================================
//
// ============================================================
// W8A32 ACTIVE TYPES  (this deployment folder targets W8A32 synthesis)
// INT8 conv weights  +  FP32 activations
// ============================================================

// INT8 conv weight type (BN-fused, per-channel symmetric) — used for weight loading.
typedef ap_int<8>   wint8_t;

// Activations remain FP32; no per-layer ap_fixed aliases needed.
typedef float  act_t;

// Accumulator: FP32 (no overflow risk with float accum).
typedef float  acc_t;

// Input image: FP32 (ImageNet-normalised, values ≈ −2.1 to +2.6)
typedef float  input_t;

// Transformer and conv layers: FP32 (weights dequantized before MAC)
typedef float  weight_t;
typedef float  bias_t;

// Bounding box and score types: FP32
typedef float  bbox_t;
typedef float  score_t;




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
// d = transformer_dim per stage (NOT in_ch * patch_area — TIMM uses no token_proj/unproj)
// The FPGA runs depth × transformer_block on P=4 independent views of [N, d] tokens.
#define MVIT_S2_DIM    144   // stage2 in_ch=96  → d=144
#define MVIT_S2_DEPTH    2
#define MVIT_S3_DIM    192   // stage3 in_ch=128 → d=192
#define MVIT_S3_DEPTH    4
#define MVIT_S4_DIM    240   // stage4 in_ch=160 → d=240
#define MVIT_S4_DEPTH    3
#define MVIT_HEADS       4   // Attention heads (qkv combined in TIMM; split at export)
#define MVIT_PATCH       2   // Patch size p (P = p*p = 4 independent views)

// Patch token counts per stage: N = (H/p)*(W/p), one value per view.
// These depend on INPUT_H/W via C*_H/W macros — change automatically with resolution.
#define MVIT_N_S2  ((C2_H / MVIT_PATCH) * (C2_W / MVIT_PATCH))  // 1600 for 640 input
#define MVIT_N_S3  ((C3_H / MVIT_PATCH) * (C3_W / MVIT_PATCH))  // 400 for 640 input
#define MVIT_N_S4  ((C4_H / MVIT_PATCH) * (C4_W / MVIT_PATCH))  // 100 for 640 input
// Largest token count (S2); drives transformer_block static buffer allocation.
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
// FPN NECK  (in_channels=[64,96,128,640], out_channels=256, num_outs=5)
// ============================================================
//#define FPN_OUT_CH    256
//#define HEAD_FEAT_CH       256
#define FPN_OUT_CH    256
#define HEAD_FEAT_CH  256

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

// Element counts for AXI depth pragmas — update with INPUT_H/W automatically
#define P2_ELEMS  (FPN_OUT_CH * P2_H * P2_W)   // 256*160*160 = 6,553,600
#define P3_ELEMS  (FPN_OUT_CH * P3_H * P3_W)   // 256*80*80   = 1,638,400
#define P4_ELEMS  (FPN_OUT_CH * P4_H * P4_W)   // 256*40*40   =   409,600
#define P5_ELEMS  (FPN_OUT_CH * P5_H * P5_W)   // 256*20*20   =   102,400
#define P6_ELEMS  (FPN_OUT_CH * P6_H * P6_W)   // 256*10*10   =    25,600

// ============================================================
// RETINANET HEAD
// num_classes=4, stacked_convs=4, feat_channels=256
// anchor scales=[4,6,8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]
// ============================================================

#define HEAD_STACKED_CONVS   4
#define NUM_CLASSES          4   // holothurian, echinus, scallop, starfish (URPC)
#define ANCHOR_SCALES_N      3   // {4, 6, 8}
#define ANCHOR_RATIOS_N      3   // {0.5, 1.0, 2.0}
#define ANCHORS_PER_LOC      (ANCHOR_SCALES_N * ANCHOR_RATIOS_N)   // 9

#define MAX_DETS       100
#define NMS_PRE       1000   // Max proposals kept per level after top-K by score
#define CAND_BUF_SIZE 5000   // Candidate collection buffer per level (> NMS_PRE to avoid spatial truncation)
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
// WEIGHT BLOCK SIZES (float32 elements per submodule)
// Used by export_weights.py and weights_loader.h
// Fused BN: each conv layer stores (scale[oc], bias[oc]) post-fusion
// ============================================================

// Convenience: fused BN adds 2*out_ch params per conv layer
#define FUSED_BN(ch)   (2 * (ch))

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
