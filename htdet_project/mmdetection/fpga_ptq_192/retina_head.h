/*
 * retina_head.h  (PTQ INT8 variant)
 * RetinaNet detection head for HTDet FPGA (Vitis HLS)
 *
 * Config:
 *   num_classes=4, stacked_convs=4, feat_channels=192
 *   anchor scales=[4, 6, 8], ratios=[0.5, 1.0, 2.0]
 *   strides=[4, 8, 16, 32, 64]   (for P2…P6)
 *
 * W8A32 PTQ changes:
 *   - retina_conv_once: int8 weights + float w_dequant_scale + float bias
 *   - retina_stacked_convs: separate w_conv (int8) and w_meta (float)
 *   - retina_pred_conv: int8 weights + float scale + float bias (acc = raw * scale + bias)
 *   - retina_head top-level: split cls/reg into _int8 and _meta pointers
 *   - ARRAY_PARTITION cyclic factor=8 → 16
 *   - #pragma HLS RESOURCE → #pragma HLS bind_storage
 *
 * Head stacked conv weight block per layer:
 *   w_conv: HEAD_FEAT_CH * HEAD_FEAT_CH * 9  (int8)
 *   w_meta: HEAD_FEAT_CH scales (w_dequant_scale) + HEAD_FEAT_CH biases (float)
 *
 * Pred conv weight block:
 *   w_conv: out_ch * HEAD_FEAT_CH * 9  (int8)
 *   w_scale: out_ch  (float, per-channel dequant scale)
 *   w_bias: out_ch   (float)
 */

#ifndef RETINA_HEAD_H
#define RETINA_HEAD_H

#include "fpga_types.h"
#include "fpga_utils.h"
#include <fstream>
#include <iostream>
#include <cmath>

// Set to 1 to save P3 intermediate arrays to csim_validation/ for comparison
#ifndef DEBUG_HEAD_DUMP
#define DEBUG_HEAD_DUMP 0
#endif

// ============================================================
// ANCHOR CONFIGURATION
// ============================================================
static const float ANCHOR_SCALES_V[ANCHOR_SCALES_N] = {4.0f, 6.0f, 8.0f};
static const float ANCHOR_RATIOS_V[ANCHOR_RATIOS_N] = {0.5f, 1.0f, 2.0f};
static const int   FPN_STRIDES[5]                    = {4,    8,   16,  32, 64};

// ============================================================
// STACKED 3×3 CONVOLUTIONS (shared across FPN levels)
// Single conv3x3 + per-channel w_dequant_scale + per-channel bias + ReLU.
// ============================================================

// Single stacked conv: int8 weights, float scale, float bias, ReLU
// Defined in retina_head.cpp
void retina_conv_once(
    const act_t*    src,
    act_t*          dst,
    const weight_t* lw,    // int8 conv weights: [HEAD_FEAT_CH * HEAD_FEAT_CH * 9]
    const meta_t*   bs,    // float per-channel w_dequant_scale
    const meta_t*   bb,    // float per-channel bias
    int H, int W
);

inline void retina_stacked_convs(
    const act_t*    input,
    act_t*          output,
    const weight_t* w_conv,    // int8: 4 × [HEAD_FEAT_CH*HEAD_FEAT_CH*9]
    const meta_t*   w_meta,    // float: 4 × [scale[192] + bias[192]]
    int H, int W
) {
    // Ping-pong buffers: layers 0,2 → tmp; layers 1,3 → output.
    static act_t tmp[HEAD_FEAT_CH * P2_H * P2_W];
    #pragma HLS bind_storage variable=tmp type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=1

    // Stride between consecutive layer weight blocks (int8 weights only).
    const int S_conv = HEAD_FEAT_CH * HEAD_FEAT_CH * 9;
    // Stride between consecutive layer meta blocks (scale + bias).
    const int S_meta = HEAD_FEAT_CH * 2;

    // Precompute per-layer weight/meta base pointers.
    const weight_t* w0c = w_conv;
    const weight_t* w1c = w_conv +   S_conv;
    const weight_t* w2c = w_conv + 2*S_conv;
    const weight_t* w3c = w_conv + 3*S_conv;

    const meta_t* w0m = w_meta;
    const meta_t* w1m = w_meta +   S_meta;
    const meta_t* w2m = w_meta + 2*S_meta;
    const meta_t* w3m = w_meta + 3*S_meta;

    // Layer 0: input  → tmp
    retina_conv_once(input,  tmp,    w0c, w0m, w0m+HEAD_FEAT_CH, H, W);
    // Layer 1: tmp    → output
    retina_conv_once(tmp,    output, w1c, w1m, w1m+HEAD_FEAT_CH, H, W);
    // Layer 2: output → tmp
    retina_conv_once(output, tmp,    w2c, w2m, w2m+HEAD_FEAT_CH, H, W);
    // Layer 3: tmp    → output
    retina_conv_once(tmp,    output, w3c, w3m, w3m+HEAD_FEAT_CH, H, W);
}

// Channel tile size for 3×3 convs in the head.
#define HEAD_IC_TILE 16

// ============================================================
// PREDICTION HEAD: 3×3 conv (192 → out_ch), int8 weights, float scale+bias
// acc = sum(input * int8_weight); output = acc * scale[oc] + bias[oc]
// ============================================================
inline void retina_pred_conv(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,   // int8: [out_ch * HEAD_FEAT_CH * 9]
    const meta_t*   w_scale,   // float: [out_ch] per-channel dequant scale
    const meta_t*   bias,      // float: [out_ch]
    int out_ch, int H, int W
) {
    RP_OC: for (int oc = 0; oc < out_ch; oc++) {
        RP_OH: for (int oh = 0; oh < H; oh++) {
            RP_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t raw_acc = 0;
                RP_ICT: for (int icb = 0; icb < HEAD_FEAT_CH; icb += HEAD_IC_TILE) {
                    RP_IC: for (int ic = icb; ic < icb + HEAD_IC_TILE; ic++) {
                        #pragma HLS UNROLL
                        RP_KH: for (int kh = 0; kh < 3; kh++) {
                            #pragma HLS UNROLL
                            RP_KW: for (int kw = 0; kw < 3; kw++) {
                                #pragma HLS UNROLL
                                int ih = oh + kh - 1;
                                int iw = ow + kw - 1;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    raw_acc += (acc_t)input[ic*H*W + ih*W + iw]
                                             * (acc_t)(float)weights[(oc*HEAD_FEAT_CH+ic)*9 + kh*3+kw];
                                }
                            }
                        }
                    }
                }
                output[oc*H*W + oh*W + ow] = (act_t)(raw_acc * (acc_t)w_scale[oc] + (acc_t)bias[oc]);
            }
        }
    }
}

// ============================================================
// ANCHOR GENERATION
// ============================================================
inline void gen_anchors(
    bbox_t* anchors,
    int H, int W, int stride
) {
    #pragma HLS INLINE
    int idx = 0;
    GA_H: for (int h = 0; h < H; h++) {
        GA_W: for (int w = 0; w < W; w++) {
            float cx = (w + 0.5f) * stride;
            float cy = (h + 0.5f) * stride;

            GA_R: for (int ri = 0; ri < ANCHOR_RATIOS_N; ri++) {
                GA_S: for (int si = 0; si < ANCHOR_SCALES_N; si++) {
                    #pragma HLS PIPELINE II=1
                    float scale = ANCHOR_SCALES_V[si] * stride;
                    float ratio = ANCHOR_RATIOS_V[ri];

                    float aw = scale / sqrtf(ratio);
                    float ah = scale * sqrtf(ratio);

                    anchors[idx*4 + 0] = (bbox_t)(cx - aw * 0.5f);
                    anchors[idx*4 + 1] = (bbox_t)(cy - ah * 0.5f);
                    anchors[idx*4 + 2] = (bbox_t)(cx + aw * 0.5f);
                    anchors[idx*4 + 3] = (bbox_t)(cy + ah * 0.5f);
                    idx++;
                }
            }
        }
    }
}

// ============================================================
// DELTA DECODING (DeltaXYWH bboxcoder, std=[1,1,1,1])
// ============================================================
inline void decode_boxes(
    const bbox_t* anchors,
    const act_t*  deltas,
    bbox_t*       boxes,
    int N
) {
    #pragma HLS INLINE
    DEC: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        bbox_t ax1 = anchors[i*4+0];
        bbox_t ay1 = anchors[i*4+1];
        bbox_t ax2 = anchors[i*4+2];
        bbox_t ay2 = anchors[i*4+3];

        bbox_t aw  = ax2 - ax1;
        bbox_t ah  = ay2 - ay1;
        bbox_t acx = ax1 + aw * (bbox_t)0.5f;
        bbox_t acy = ay1 + ah * (bbox_t)0.5f;

        float dx = (float)deltas[i*4+0];
        float dy = (float)deltas[i*4+1];
        float dw = (float)deltas[i*4+2];
        float dh = (float)deltas[i*4+3];

        if (dw > 4.135f) dw = 4.135f;
        if (dh > 4.135f) dh = 4.135f;

        float pred_cx = dx * (float)aw + (float)acx;
        float pred_cy = dy * (float)ah + (float)acy;
        float pred_w  = expf(dw) * (float)aw;
        float pred_h  = expf(dh) * (float)ah;

        boxes[i*4+0] = (bbox_t)(pred_cx - pred_w * 0.5f);
        boxes[i*4+1] = (bbox_t)(pred_cy - pred_h * 0.5f);
        boxes[i*4+2] = (bbox_t)(pred_cx + pred_w * 0.5f);
        boxes[i*4+3] = (bbox_t)(pred_cy + pred_h * 0.5f);
    }
}

// ============================================================
// NMS (greedy, sorted by score descending)
// ============================================================
static inline float iou_f(const bbox_t* a, const bbox_t* b) {
    #pragma HLS INLINE
    float x1 = (float)(a[0] > b[0] ? a[0] : b[0]);
    float y1 = (float)(a[1] > b[1] ? a[1] : b[1]);
    float x2 = (float)(a[2] < b[2] ? a[2] : b[2]);
    float y2 = (float)(a[3] < b[3] ? a[3] : b[3]);
    float inter_w = x2 - x1; if (inter_w <= 0) return 0;
    float inter_h = y2 - y1; if (inter_h <= 0) return 0;
    float inter = inter_w * inter_h;
    float area_a = (float)(a[2]-a[0]) * (float)(a[3]-a[1]);
    float area_b = (float)(b[2]-b[0]) * (float)(b[3]-b[1]);
    float uni = area_a + area_b - inter;
    return (uni > 0) ? (inter / uni) : 0;
}

// Defined in retina_head.cpp.
int nms(
    const bbox_t*  boxes,
    const score_t* scores,
    int*           order,
    int*           keep,
    const int*     class_ids,
    int N,
    float iou_thr
);

// Simple insertion sort (descending by score)
inline void sort_scores(const score_t* scores, int* order, int N) {
    #pragma HLS INLINE
    for (int i = 0; i < N; i++) order[i] = i;
    SS_I: for (int i = 1; i < N; i++) {
        int key = order[i];
        score_t key_s = scores[key];
        int j = i - 1;
        SS_J: while (j >= 0 && scores[order[j]] < key_s) {
            order[j+1] = order[j];
            j--;
        }
        order[j+1] = key;
    }
}

// ============================================================
// PROCESS ONE FPN LEVEL — defined in retina_head.cpp
// ============================================================
int process_level(
    const act_t*    feat,
    const weight_t* cls_conv_int8,
    const meta_t*   cls_conv_meta,
    const weight_t* reg_conv_int8,
    const meta_t*   reg_conv_meta,
    const weight_t* cls_pred_w,   const meta_t* cls_pred_scale, const meta_t* cls_pred_b,
    const weight_t* reg_pred_w,   const meta_t* reg_pred_scale, const meta_t* reg_pred_b,
    int H, int W, int stride,
    bbox_t*  cand_boxes,
    score_t* cand_scores,
    int*     cand_cls,
    int      max_cand
);

// ============================================================
// TOP-LEVEL RETINA HEAD
// ============================================================
inline void retina_head(
    const act_t* p2,
    const act_t* p3,
    const act_t* p4,
    const act_t* p5,
    const act_t* p6,

    const weight_t* cls_conv_int8,   // int8: 4 layers × [192*192*9]
    const meta_t*   cls_conv_meta,   // float: 4 layers × [scale[192] + bias[192]]
    const weight_t* reg_conv_int8,
    const meta_t*   reg_conv_meta,

    const weight_t* cls_pred_w,      // int8: [36*192*9]
    const meta_t*   cls_pred_scale,  // float: [36]
    const meta_t*   cls_pred_b,      // float: [36]
    const weight_t* reg_pred_w,      // int8: [36*192*9]
    const meta_t*   reg_pred_scale,  // float: [36]
    const meta_t*   reg_pred_b,      // float: [36]

    Detection* detections,
    int*       num_dets
) {
    static bbox_t  cand_boxes [CAND_BUF_SIZE * 4];
    static score_t cand_scores[CAND_BUF_SIZE];
    static int     cand_cls   [CAND_BUF_SIZE];
    static int     cand_order [CAND_BUF_SIZE];
    #pragma HLS bind_storage variable=cand_boxes  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=cand_scores type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=cand_cls    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=cand_order  type=RAM_T2P impl=BRAM

    static bbox_t  global_boxes [5 * NMS_PRE * 4];
    static score_t global_scores[5 * NMS_PRE];
    static int     global_cls   [5 * NMS_PRE];
    static int     global_order [5 * NMS_PRE];
    static int     global_keep  [MAX_DETS];
    #pragma HLS bind_storage variable=global_boxes  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=global_scores type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=global_cls    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=global_order  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=global_keep   type=RAM_T2P impl=BRAM

    int global_nc = 0;

#define COLLECT_LEVEL(FEAT, LH, LW, STR, LABEL) \
    { \
        int nc = process_level(FEAT, \
                               cls_conv_int8, cls_conv_meta, \
                               reg_conv_int8, reg_conv_meta, \
                               cls_pred_w, cls_pred_scale, cls_pred_b, \
                               reg_pred_w, reg_pred_scale, reg_pred_b, \
                               LH, LW, STR, \
                               cand_boxes, cand_scores, cand_cls, \
                               CAND_BUF_SIZE); \
        sort_scores(cand_scores, cand_order, nc); \
        int nc_topk = (nc < NMS_PRE) ? nc : NMS_PRE; \
        int space   = 5 * NMS_PRE - global_nc; \
        int copy_n  = (nc_topk < space) ? nc_topk : space; \
        LABEL: for (int k = 0; k < copy_n; k++) { \
            int ki = cand_order[k]; \
            global_boxes [(global_nc+k)*4+0] = cand_boxes[ki*4+0]; \
            global_boxes [(global_nc+k)*4+1] = cand_boxes[ki*4+1]; \
            global_boxes [(global_nc+k)*4+2] = cand_boxes[ki*4+2]; \
            global_boxes [(global_nc+k)*4+3] = cand_boxes[ki*4+3]; \
            global_scores[global_nc+k] = cand_scores[ki]; \
            global_cls   [global_nc+k] = cand_cls[ki]; \
        } \
        global_nc += copy_n; \
    }

    COLLECT_LEVEL(p2, P2_H, P2_W, FPN_STRIDES[0], CPY_L0)
    COLLECT_LEVEL(p3, P3_H, P3_W, FPN_STRIDES[1], CPY_L1)
    COLLECT_LEVEL(p4, P4_H, P4_W, FPN_STRIDES[2], CPY_L2)
    COLLECT_LEVEL(p5, P5_H, P5_W, FPN_STRIDES[3], CPY_L3)
    COLLECT_LEVEL(p6, P6_H, P6_W, FPN_STRIDES[4], CPY_L4)
#undef COLLECT_LEVEL

    sort_scores(global_scores, global_order, global_nc);
    int nk = nms(global_boxes, global_scores, global_order, global_keep, global_cls,
                 global_nc, NMS_IOU_THR);

    int total_dets = (nk < MAX_DETS) ? nk : MAX_DETS;

    for (int k = 0; k < total_dets; k++) {
        int ki = global_keep[k];
        detections[k].x1       = global_boxes[ki * 4 + 0];
        detections[k].y1       = global_boxes[ki * 4 + 1];
        detections[k].x2       = global_boxes[ki * 4 + 2];
        detections[k].y2       = global_boxes[ki * 4 + 3];
        detections[k].score    = global_scores[ki];
        detections[k].class_id = global_cls[ki];
    }

    *num_dets = total_dets;
}

#endif // RETINA_HEAD_H
