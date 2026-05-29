/*
 * fpn_neck.cpp  (PTQ INT8 variant — FPN dequant scale fix applied)
 * Feature Pyramid Network — function definitions (Vitis HLS).
 *
 * Fixes vs fpn_neck.cpp:
 *   - lateral 1×1 convs now use conv1x1_bn (applies eff_scale + eff_bias)
 *   - fpn_conv3x3_v2 applies eff_scale[oc] × acc + eff_bias[oc] (was bias-only init, no scale)
 *   - fpn_meta layout changed to [scale[192], bias[192]] per conv (3072 total, was 1536)
 *   - moff advances by FPN_OUT_CH*2 per conv (was FPN_OUT_CH)
 */

#include "fpn_neck.h"

#define FPN3_IC_TILE 16

void fpn_conv3x3_v2(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const meta_t*   scale,
    const meta_t*   bias,
    int H, int W
) {
    FPN3_OC: for (int oc = 0; oc < FPN_OUT_CH; oc++) {
        FPN3_OH: for (int oh = 0; oh < H; oh++) {
            FPN3_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                FPN3_ICT: for (int icb = 0; icb < FPN_OUT_CH; icb += FPN3_IC_TILE) {
                    FPN3_IC: for (int ic = icb; ic < icb + FPN3_IC_TILE; ic++) {
                        #pragma HLS UNROLL
                        FPN3_KH: for (int kh = 0; kh < 3; kh++) {
                            #pragma HLS UNROLL
                            FPN3_KW: for (int kw = 0; kw < 3; kw++) {
                                #pragma HLS UNROLL
                                int ih = oh + kh - 1;
                                int iw = ow + kw - 1;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    acc += (acc_t)input[ic*H*W + ih*W + iw]
                                         * (acc_t)(float)weights[(oc*FPN_OUT_CH+ic)*9 + kh*3+kw];
                                }
                            }
                        }
                    }
                }
                output[oc*H*W + oh*W + ow] = (act_t)((acc_t)acc * (acc_t)scale[oc] + (acc_t)bias[oc]);
            }
        }
    }
}

void fpn_neck(
    const act_t* c1,
    const act_t* c2,
    const act_t* c3,
    const act_t* c4,
    const weight_t* fpn_conv,
    const meta_t*   fpn_meta,
    act_t* p2,
    act_t* p3,
    act_t* p4,
    act_t* p5,
    act_t* p6
) {
    static act_t lat1[FPN_OUT_CH * P2_H * P2_W];
    static act_t lat2[FPN_OUT_CH * P3_H * P3_W];
    static act_t lat3[FPN_OUT_CH * P4_H * P4_W];
    static act_t lat4[FPN_OUT_CH * P5_H * P5_W];
    #pragma HLS bind_storage variable=lat1 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat2 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat3 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=lat4 type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=lat1 cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat2 cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat3 cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat4 cyclic factor=16 dim=1

    int coff = 0;  // int8 weight element index
    int moff = 0;  // float meta element index (each conv: scale[192] then bias[192])

    // ── Lateral 1×1 projections: int8 weights + BN (eff_scale, eff_bias) ─────
    // fpn_meta layout: [scale[FPN_OUT_CH], bias[FPN_OUT_CH]] per conv
    conv1x1_bn(c1, lat1, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH,
               C1_CH, FPN_OUT_CH, P2_H, P2_W);
    coff += FPN_OUT_CH * C1_CH;
    moff += FPN_OUT_CH * 2;

    conv1x1_bn(c2, lat2, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH,
               C2_CH, FPN_OUT_CH, P3_H, P3_W);
    coff += FPN_OUT_CH * C2_CH;
    moff += FPN_OUT_CH * 2;

    conv1x1_bn(c3, lat3, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH,
               C3_CH, FPN_OUT_CH, P4_H, P4_W);
    coff += FPN_OUT_CH * C3_CH;
    moff += FPN_OUT_CH * 2;

    conv1x1_bn(c4, lat4, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH,
               C4_CH, FPN_OUT_CH, P5_H, P5_W);
    coff += FPN_OUT_CH * C4_CH;
    moff += FPN_OUT_CH * 2;

    // ── Top-down pathway: fused upsample + in-place add ─────────────────────
    FPN_UP43: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P4_H; h++) {
            for (int w = 0; w < P4_W; w++) {
                #pragma HLS PIPELINE II=1
                lat3[c*P4_H*P4_W + h*P4_W + w] +=
                    lat4[c*P5_H*P5_W + (h>>1)*P5_W + (w>>1)];
            }
        }
    }

    FPN_UP32: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P3_H; h++) {
            for (int w = 0; w < P3_W; w++) {
                #pragma HLS PIPELINE II=1
                lat2[c*P3_H*P3_W + h*P3_W + w] +=
                    lat3[c*P4_H*P4_W + (h>>1)*P4_W + (w>>1)];
            }
        }
    }

    FPN_UP21: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P2_H; h++) {
            for (int w = 0; w < P2_W; w++) {
                #pragma HLS PIPELINE II=1
                lat1[c*P2_H*P2_W + h*P2_W + w] +=
                    lat2[c*P3_H*P3_W + (h>>1)*P3_W + (w>>1)];
            }
        }
    }

    // ── Output 3×3 smoothing convolutions: int8 weights + BN (scale, bias) ──
    fpn_conv3x3_v2(lat1, p2, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH, P2_H, P2_W);
    coff += FPN_OUT_CH*FPN_OUT_CH*9;
    moff += FPN_OUT_CH * 2;

    fpn_conv3x3_v2(lat2, p3, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH, P3_H, P3_W);
    coff += FPN_OUT_CH*FPN_OUT_CH*9;
    moff += FPN_OUT_CH * 2;

    fpn_conv3x3_v2(lat3, p4, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH, P4_H, P4_W);
    coff += FPN_OUT_CH*FPN_OUT_CH*9;
    moff += FPN_OUT_CH * 2;

    fpn_conv3x3_v2(lat4, p5, fpn_conv+coff, fpn_meta+moff, fpn_meta+moff+FPN_OUT_CH, P5_H, P5_W);
    coff += FPN_OUT_CH*FPN_OUT_CH*9;
    moff += FPN_OUT_CH * 2;

    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);

    (void)coff;
    (void)moff;
}
