/*
 * fpn_neck.cpp
 * Feature Pyramid Network — function definitions (Vitis HLS).
 * Add this file to the Vitis HLS project alongside fpn_neck.h.
 *
 * Changes vs original:
 *   - fpn_conv3x3: channel-tiled inner loop (TILE=16) instead of full UNROLL of 256
 *   - fpn_neck: upsample+add fused in-place — eliminates up2/up3/up4 buffers (~34 MB)
 */

#include "fpn_neck.h"

// Tile size for input-channel dimension in fpn_conv3x3.
// UNROLL of the full 256-channel inner loop produces 2304 adders per pixel;
// tiling to 16 reduces the adder tree to manageable size for synthesis.
#define FPN3_IC_TILE 16

void fpn_conv3x3(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const bias_t*   bias,
    int H, int W
) {
    FPN3_OC: for (int oc = 0; oc < FPN_OUT_CH; oc++) {
        FPN3_OH: for (int oh = 0; oh < H; oh++) {
            FPN3_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = (acc_t)bias[oc];
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
                                         * (acc_t)weights[(oc*FPN_OUT_CH+ic)*9 + kh*3+kw];
                                }
                            }
                        }
                    }
                }
                output[oc*H*W + oh*W + ow] = (act_t)acc;
            }
        }
    }
}

void fpn_neck(
    const act_t* c1,
    const act_t* c2,
    const act_t* c3,
    const act_t* c4,
    const weight_t* fpn_w,
    act_t* p2,
    act_t* p3,
    act_t* p4,
    act_t* p5,
    act_t* p6
) {
    // Lateral outputs (4 levels).  No separate upsample buffers needed —
    // the top-down adds are done by reading the parent level directly.
    static act_t lat1[FPN_OUT_CH * P2_H * P2_W];
    static act_t lat2[FPN_OUT_CH * P3_H * P3_W];
    static act_t lat3[FPN_OUT_CH * P4_H * P4_W];
    static act_t lat4[FPN_OUT_CH * P5_H * P5_W];
    #pragma HLS RESOURCE variable=lat1 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=lat2 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=lat3 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=lat4 core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=lat1 cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat2 cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat3 cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=lat4 cyclic factor=8 dim=1

    int off = 0;

    // ── Lateral 1×1 projections ──────────────────────────────────────────────
    conv1x1_plain(c1, lat1, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*C1_CH),
                  C1_CH, FPN_OUT_CH, P2_H, P2_W);
    off += FPN_OUT_CH*C1_CH + FPN_OUT_CH;

    conv1x1_plain(c2, lat2, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*C2_CH),
                  C2_CH, FPN_OUT_CH, P3_H, P3_W);
    off += FPN_OUT_CH*C2_CH + FPN_OUT_CH;

    conv1x1_plain(c3, lat3, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*C3_CH),
                  C3_CH, FPN_OUT_CH, P4_H, P4_W);
    off += FPN_OUT_CH*C3_CH + FPN_OUT_CH;

    conv1x1_plain(c4, lat4, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*C4_CH),
                  C4_CH, FPN_OUT_CH, P5_H, P5_W);
    off += FPN_OUT_CH*C4_CH + FPN_OUT_CH;

    // ── Top-down pathway: fused upsample + in-place add (no temporary buffers) ─
    // P4-level: lat3 += nearest-neighbour 2× upsample of lat4
    FPN_UP43: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P4_H; h++) {
            for (int w = 0; w < P4_W; w++) {
                #pragma HLS PIPELINE II=1
                lat3[c*P4_H*P4_W + h*P4_W + w] +=
                    lat4[c*P5_H*P5_W + (h>>1)*P5_W + (w>>1)];
            }
        }
    }

    // P3-level: lat2 += nearest-neighbour 2× upsample of (post-add) lat3
    FPN_UP32: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P3_H; h++) {
            for (int w = 0; w < P3_W; w++) {
                #pragma HLS PIPELINE II=1
                lat2[c*P3_H*P3_W + h*P3_W + w] +=
                    lat3[c*P4_H*P4_W + (h>>1)*P4_W + (w>>1)];
            }
        }
    }

    // P2-level: lat1 += nearest-neighbour 2× upsample of (post-add) lat2
    FPN_UP21: for (int c = 0; c < FPN_OUT_CH; c++) {
        for (int h = 0; h < P2_H; h++) {
            for (int w = 0; w < P2_W; w++) {
                #pragma HLS PIPELINE II=1
                lat1[c*P2_H*P2_W + h*P2_W + w] +=
                    lat2[c*P3_H*P3_W + (h>>1)*P3_W + (w>>1)];
            }
        }
    }

    // ── Output 3×3 smoothing convolutions ────────────────────────────────────
    const int out_w_stride = FPN_OUT_CH * FPN_OUT_CH * 9 + FPN_OUT_CH;

    fpn_conv3x3(lat1, p2, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*FPN_OUT_CH*9), P2_H, P2_W);
    off += out_w_stride;

    fpn_conv3x3(lat2, p3, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*FPN_OUT_CH*9), P3_H, P3_W);
    off += out_w_stride;

    fpn_conv3x3(lat3, p4, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*FPN_OUT_CH*9), P4_H, P4_W);
    off += out_w_stride;

    fpn_conv3x3(lat4, p5, fpn_w+off, (const bias_t*)(fpn_w+off+FPN_OUT_CH*FPN_OUT_CH*9), P5_H, P5_W);
    off += out_w_stride;

    maxpool2x2(p5, p6, FPN_OUT_CH, P5_H, P5_W);
}
