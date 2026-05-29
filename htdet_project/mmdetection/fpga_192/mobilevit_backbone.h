/*
 * mobilevit_backbone.h
 * MobileViT-S backbone for HTDet FPGA (Vitis HLS)
 *
 * Architecture (TIMM mobilevit_s, out_indices=(1,2,3,4)):
 *
 *   Input  [3, 640, 640]  (INPUT_H=640, INPUT_W=640 — change in fpga_types.h)
 *   Stem   Conv3x3(3→16, s=2) + BN + SiLU            → [16, 320, 320]
 *   Stg0   MBConv(16→32, expand=4, s=1)               → [32, 320, 320]  (not exported)
 *   Stg1   MBConv(32→64, expand=4, s=2)               → [64, 160, 160]
 *          MBConv(64→64, expand=4, s=1) × 2            → C1 = [64, 160, 160]  ← FPN input 0
 *   Stg2   MBConv(64→96, expand=4, s=2)               → [96,  80,  80]
 *          MobileViTBlock(96, d=144, depth=2)           → C2 = [96,  80,  80]  ← FPN input 1
 *   Stg3   MBConv(96→128, expand=4, s=2)              → [128, 40,  40]
 *          MobileViTBlock(128, d=192, depth=4)          → C3 = [128, 40,  40]  ← FPN input 2
 *   Stg4   MBConv(128→160, expand=4, s=2)             → [160, 20,  20]
 *          MobileViTBlock(160, d=240, depth=3)          → [160, 20,  20]
 *          Conv1x1(160→640) + BN + SiLU               → C4 = [640, 20,  20]  ← FPN input 3
 *
 * Weight layout (flat float32 / weight_t array, passed from DDR):
 *   See WEIGHT_OFFSETS section at the bottom of this file.
 *   All BN parameters are pre-fused into conv weights at export time:
 *     - bn_scale[oc] = gamma[oc] / sqrt(var[oc] + eps)
 *     - bn_bias[oc]  = beta[oc]  - mean[oc] * bn_scale[oc]
 *
 * MobileViT patch/transformer convention:
 *   patch_size = 2×2.  For an H×W feature map, we extract (H/2)×(W/2) = N patches.
 *   Each patch is proj_dim = d × patch_size² values → linearly projected to d.
 *   TransformerBlock: LayerNorm → MHSA (4 heads) → residual → LayerNorm → MLP → residual.
 */

#ifndef MOBILEVIT_BACKBONE_H
#define MOBILEVIT_BACKBONE_H

#include "fpga_types.h"
#include "fpga_utils.h"

// Definitions are in mobilevit_backbone.cpp — add that file to the Vitis HLS project.

void mbconv_160(const act_t* in, act_t* out, const weight_t* w, int in_ch, int out_ch, int expand, int stride);
void mbconv_80 (const act_t* in, act_t* out, const weight_t* w, int in_ch, int out_ch, int expand, int stride);
void mbconv_40 (const act_t* in, act_t* out, const weight_t* w, int in_ch, int out_ch, int expand, int stride);
void mbconv_20 (const act_t* in, act_t* out, const weight_t* w, int in_ch, int out_ch, int expand, int stride);
void mbconv_10 (const act_t* in, act_t* out, const weight_t* w, int in_ch, int out_ch, int expand, int stride);

void transformer_block(act_t* tokens, const weight_t* w, int N, int dim);

// Stage-specific transformer blocks — fixed N/dim, right-sized BRAM.
void transformer_block_s2(act_t* tokens, const weight_t* w);
void transformer_block_s3(act_t* tokens, const weight_t* w);
void transformer_block_s4(act_t* tokens, const weight_t* w);

void mobilevit_block_s2(const act_t* input, act_t* output, const weight_t* w);
void mobilevit_block_s3(const act_t* input, act_t* output, const weight_t* w);
void mobilevit_block_s4(const act_t* input, act_t* output, const weight_t* w);

// Top-level backbone: image [INPUT_C*H*W] + flat weights → C1..C4 feature maps.
void mobilevit_backbone(
    const input_t*  image,
    const weight_t* weights,
    act_t* c1,   // [C1_CH * C1_H * C1_W]
    act_t* c2,   // [C2_CH * C2_H * C2_W]
    act_t* c3,   // [C3_CH * C3_H * C3_W]
    act_t* c4    // [C4_CH * C4_H * C4_W]
);

#endif // MOBILEVIT_BACKBONE_H
