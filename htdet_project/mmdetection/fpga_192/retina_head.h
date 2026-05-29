/*
 * retina_head.h
 * RetinaNet detection head for HTDet FPGA (Vitis HLS)
 *
 * Config:
 *   num_classes=4, stacked_convs=4, feat_channels=256
 *   anchor scales=[4, 6, 8], ratios=[0.5, 1.0, 2.0]
 *   strides=[4, 8, 16, 32, 64]   (for P2…P6)
 *   bbox_coder: DeltaXYWH (target_stds=[1,1,1,1])
 *   score_thr=0.05, nms_iou=0.5, max_per_img=100, nms_pre=1000
 *
 * Two branches per FPN level (shared weights across levels in MMDetection):
 *   cls_branch: 4 × Conv3x3(256→256, GN, ReLU) → Conv3x3(256 → 9*4)  + sigmoid
 *   reg_branch: 4 × Conv3x3(256→256, GN, ReLU) → Conv3x3(256 → 9*4)
 *
 * Note: This model's RetinaHead has norm_cfg NOT SET — the stacked convs are
 *       plain Conv2d(bias=True)+ReLU with NO GroupNorm.  The exported binary
 *       therefore has scale=ones(256) and bias=conv.bias per layer.
 *
 * Weight layout (flat, cls head first then reg head):
 *   cls_conv_w[0..3]:  4 × [256*256*9, scale[256]=ones, bias[256]=conv.bias]
 *   cls_pred_w:        [9*4 * 256 * 9], cls_pred_b [9*4]  (final 3x3, no BN)
 *   reg_conv_w[0..3]:  4 × [256*256*9, scale[256]=ones, bias[256]=conv.bias]
 *   reg_pred_w:        [9*4 * 256 * 9], reg_pred_b [9*4]  (final 3x3, no BN)
 *
 * Total head elements ≈ (4+4)*(256*256*9 + 256+256) + 2*(9*4*256*9 + 9*4)
 *                     ≈ 8*589,824 + 8*512 + 2*82,944 + 2*36
 *                     ≈ 4,722,960 elements × 2B ≈ 9.5MB (stored in DDR)
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
// scales=[4,6,8], ratios=[0.5,1.0,2.0]
// Anchors are computed at runtime; no stored parameters needed.
// ============================================================
static const float ANCHOR_SCALES_V[ANCHOR_SCALES_N] = {4.0f, 6.0f, 8.0f};
static const float ANCHOR_RATIOS_V[ANCHOR_RATIOS_N] = {0.5f, 1.0f, 2.0f};
static const int   FPN_STRIDES[5]                    = {4,    8,   16,  32, 64};

// ============================================================
// STACKED 3×3 CONVOLUTIONS (shared across FPN levels)
// in_ch=256 for all layers; 4 stacked convs with plain Conv2d+ReLU.
// One set for cls branch, one set for reg branch.
// ============================================================

// Single conv3x3 + per-channel affine (scale=1) + ReLU.
// Defined in retina_head.cpp (non-static for linkage with inline callers).
void retina_conv_once(
    const act_t*    src,
    act_t*          dst,
    const weight_t* lw,    // conv weights: [HEAD_FEAT_CH * HEAD_FEAT_CH * 9]
    const weight_t* bs,    // per-channel scale (all ones for plain conv)
    const bias_t*   bb,    // per-channel bias
    int H, int W
);

inline void retina_stacked_convs(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,       // flat: 4 × [256*256*9 + 256+256]
    int H, int W
) {
    // Ping-pong buffers: layers 0,2 → tmp; layers 1,3 → output.
    // With HEAD_STACKED_CONVS=4, final write is always to output.
    static act_t tmp[HEAD_FEAT_CH * P2_H * P2_W];
    #pragma HLS RESOURCE variable=tmp core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=8 dim=1

    // Stride between consecutive layer weight blocks.
    const int S  = HEAD_FEAT_CH*HEAD_FEAT_CH*9 + HEAD_FEAT_CH*2;
    // Offset from layer base to scale array.
    const int WO = HEAD_FEAT_CH*HEAD_FEAT_CH*9;

    // Precompute per-layer weight base pointers (no ternary pointer selection).
    const weight_t* w0 = weights;
    const weight_t* w1 = weights +   S;
    const weight_t* w2 = weights + 2*S;
    const weight_t* w3 = weights + 3*S;

    // Layer 0: input  → tmp
    retina_conv_once(input,  tmp,    w0, w0+WO, (const bias_t*)(w0+WO+HEAD_FEAT_CH), H, W);
    // Layer 1: tmp    → output
    retina_conv_once(tmp,    output, w1, w1+WO, (const bias_t*)(w1+WO+HEAD_FEAT_CH), H, W);
    // Layer 2: output → tmp
    retina_conv_once(output, tmp,    w2, w2+WO, (const bias_t*)(w2+WO+HEAD_FEAT_CH), H, W);
    // Layer 3: tmp    → output
    retina_conv_once(tmp,    output, w3, w3+WO, (const bias_t*)(w3+WO+HEAD_FEAT_CH), H, W);
}

// Channel tile size for 3×3 convs in the head.
// Tiling to 16 reduces the adder tree from 256×9=2304 ops to 16×9=144 ops
// per pipeline iteration, making II=1 achievable for synthesis.
#define HEAD_IC_TILE 16

// ============================================================
// PREDICTION HEAD: 3×3 conv (256 → out_ch), no BN, plain bias
// Used for both cls_pred (out_ch=9*4=36) and reg_pred (out_ch=36).
// ============================================================
inline void retina_pred_conv(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,   // [out_ch * 256 * 9]
    const bias_t*   bias,      // [out_ch]
    int out_ch, int H, int W
) {
    RP_OC: for (int oc = 0; oc < out_ch; oc++) {
        RP_OH: for (int oh = 0; oh < H; oh++) {
            RP_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = (acc_t)bias[oc];
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
                                    acc += (acc_t)input[ic*H*W + ih*W + iw]
                                         * (acc_t)weights[(oc*HEAD_FEAT_CH+ic)*9 + kh*3+kw];
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

// ============================================================
// ANCHOR GENERATION
// For one FPN level: generate anchors in [x1,y1,x2,y2] format.
// anchors: [H * W * ANCHORS_PER_LOC * 4]
// ============================================================
inline void gen_anchors(
    bbox_t* anchors,
    int H, int W, int stride
) {
    #pragma HLS INLINE
    int idx = 0;
    GA_H: for (int h = 0; h < H; h++) {
        GA_W: for (int w = 0; w < W; w++) {
            // Anchor center in input image space
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
// boxes: [N * 4]  (x1,y1,x2,y2) decoded from anchor + delta
// ============================================================
inline void decode_boxes(
    const bbox_t* anchors,
    const act_t*  deltas,      // [N * 4]  (dx, dy, dw, dh)
    bbox_t*       boxes,       // [N * 4]  (x1, y1, x2, y2) output
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

        // Clamp dw/dh to avoid huge exp values
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
// Operates on a flat list of (box, score) pairs.
// keep[]: indices of kept boxes; returns num_keep.
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

// class_ids: per-candidate class label. Suppression only within same class
// (matches MMDetection multiclass_nms). Defined in retina_head.cpp.
int nms(
    const bbox_t*  boxes,
    const score_t* scores,
    int*           order,
    int*           keep,
    const int*     class_ids,
    int N,
    float iou_thr
);

// Simple insertion sort (descending by score) for ≤ NMS_PRE elements
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
// Runs stacked convs, pred convs, score-filters candidates.
// Returns number of candidates written to cand_boxes/cand_scores/cand_cls.
// ============================================================
int process_level(
    const act_t*    feat,
    const weight_t* cls_conv_w,
    const weight_t* reg_conv_w,
    const weight_t* cls_pred_w,   const bias_t* cls_pred_b,
    const weight_t* reg_pred_w,   const bias_t* reg_pred_b,
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
    // FPN feature maps (BRAM)
    const act_t* p2,    // [256, 160, 160]
    const act_t* p3,    // [256,  80,  80]
    const act_t* p4,    // [256,  40,  40]
    const act_t* p5,    // [256,  20,  20]
    const act_t* p6,    // [256,  10,  10]

    // Head weights from DDR
    const weight_t* head_cls_w,    // cls stacked convs weights (all 4 layers)
    const weight_t* head_reg_w,    // reg stacked convs weights (all 4 layers)
    const weight_t* cls_pred_w,    // cls pred conv weights [36*256*9]
    const bias_t*   cls_pred_b,    // cls pred conv bias   [36]
    const weight_t* reg_pred_w,    // reg pred conv weights [36*256*9]
    const bias_t*   reg_pred_b,    // reg pred conv bias   [36]

    // Outputs
    Detection* detections,
    int*       num_dets
) {
    // Per-level candidate collection buffer (reused each level)
    static bbox_t  cand_boxes [CAND_BUF_SIZE * 4];
    static score_t cand_scores[CAND_BUF_SIZE];
    static int     cand_cls   [CAND_BUF_SIZE];
    static int     cand_order [CAND_BUF_SIZE];
    #pragma HLS RESOURCE variable=cand_boxes  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=cand_scores core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=cand_cls    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=cand_order  core=RAM_T2P_BRAM

    // Global pool: accumulates topk candidates from all 5 FPN levels before one global NMS.
    // Matches MMDetection: merge all levels first, then multiclass_nms once.
    // Size = 5 * NMS_PRE (each level contributes at most NMS_PRE candidates).
    static bbox_t  global_boxes [5 * NMS_PRE * 4];
    static score_t global_scores[5 * NMS_PRE];
    static int     global_cls   [5 * NMS_PRE];
    static int     global_order [5 * NMS_PRE];
    static int     global_keep  [MAX_DETS];
    #pragma HLS RESOURCE variable=global_boxes  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=global_scores core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=global_cls    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=global_order  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=global_keep   core=RAM_T2P_BRAM

    int global_nc = 0;

    // ---- Pass 1: explicit per-level processing (no pointer-select loop) ────
    // Eliminates conditional lvl_feat pointer; HLS can analyze each call site.
#define COLLECT_LEVEL(FEAT, LH, LW, STR, LABEL) \
    { \
        int nc = process_level(FEAT, head_cls_w, head_reg_w, \
                               cls_pred_w, cls_pred_b, \
                               reg_pred_w, reg_pred_b, \
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

    // ---- Pass 2: one global NMS across all levels (matches MMDetection multiclass_nms) ----
    sort_scores(global_scores, global_order, global_nc);
    int nk = nms(global_boxes, global_scores, global_order, global_keep, global_cls,
                 global_nc, NMS_IOU_THR);

    int total_dets = (nk < MAX_DETS) ? nk : MAX_DETS;

    // NMS already processes in score-descending order, so output is sorted
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
