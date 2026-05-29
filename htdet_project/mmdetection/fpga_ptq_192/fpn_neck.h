/*
 * fpn_neck.h  (PTQ INT8 variant — FPN dequant scale fix applied)
 * Feature Pyramid Network for HTDet FPGA (Vitis HLS)
 *
 * Config: in_channels=[64, 96, 128, 640], out_channels=192, num_outs=5
 *
 * Architecture (standard MMDetection FPN, top-down pathway):
 *   C1 [64,  160, 160] → lateral_conv1 (1×1, 64→192)  ──────────────── P2 [192, 160, 160]
 *   C2 [96,   80,  80] → lateral_conv2 (1×1, 96→192)  ← upsample+add ─ P3 [192,  80,  80]
 *   C3 [128,  40,  40] → lateral_conv3 (1×1, 128→192) ← upsample+add ─ P4 [192,  40,  40]
 *   C4 [640,  20,  20] → lateral_conv4 (1×1, 640→192) ← (top)         P5 [192,  20,  20]
 *   P5 ──── maxpool(2×2) ──────────────────────────── P6 [192,  10,  10]
 *   After add: each level gets a 3×3 output conv (192→192), no activation.
 *
 * W8A32 PTQ split:
 *   fpn_conv (int8_t): all conv kernel weights
 *   fpn_meta (float):  per-conv [eff_scale[192], eff_bias[192]] (scale+bias interleaved)
 *
 * Weight layout:
 *   fpn_conv: lat1_w[192*64], lat2_w[192*96], lat3_w[192*128], lat4_w[192*640],
 *             out1_w[192*192*9], out2_w, out3_w, out4_w
 *   fpn_meta: lat1_s[192], lat1_b[192],
 *             lat2_s[192], lat2_b[192],
 *             lat3_s[192], lat3_b[192],
 *             lat4_s[192], lat4_b[192],
 *             out1_s[192], out1_b[192],
 *             out2_s[192], out2_b[192],
 *             out3_s[192], out3_b[192],
 *             out4_s[192], out4_b[192]
 *
 * Total fpn_conv elements: (64+96+128+640)*192 + 4*(192*192*9) = 1,503,744
 * Total fpn_meta elements: 8 * (192+192) = 3072
 */

#ifndef FPN_NECK_V2_H
#define FPN_NECK_V2_H

#include "fpga_types.h"
#include "fpga_utils.h"

// 3×3 conv + BN (fused dequant scale + bias), no activation.  in_ch = out_ch = FPN_OUT_CH = 192.
// weights: int8, scale: meta_t (float eff_scale), bias: meta_t (float eff_bias)
void fpn_conv3x3_v2(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const meta_t*   scale,
    const meta_t*   bias,
    int H, int W
);

// FPN forward pass: C1–C4 backbone features → P2–P6 pyramid outputs.
// fpn_conv: int8_t conv weights (fpn_int8.bin)
// fpn_meta: float [scale, bias] per conv (fpn_meta_float.bin, 3072 elements)
void fpn_neck(
    const act_t* c1,        // [C1_CH=64,  160, 160]
    const act_t* c2,        // [C2_CH=96,   80,  80]
    const act_t* c3,        // [C3_CH=128,  40,  40]
    const act_t* c4,        // [C4_CH=640,  20,  20]
    const weight_t* fpn_conv,   // fpn_int8.bin
    const meta_t*   fpn_meta,   // fpn_meta_float.bin ([scale,bias] × 8 convs)
    act_t* p2,              // [FPN_OUT_CH=192, 160, 160]
    act_t* p3,              // [FPN_OUT_CH=192,  80,  80]
    act_t* p4,              // [FPN_OUT_CH=192,  40,  40]
    act_t* p5,              // [FPN_OUT_CH=192,  20,  20]
    act_t* p6               // [FPN_OUT_CH=192,  10,  10]
);

#endif // FPN_NECK_V2_H
