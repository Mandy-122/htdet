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
// GROUP NORM + RELU  (applied in-place after each stacked conv)
// G=32 groups, HEAD_FEAT_CH=256 → group_size=8.
// Weight layout: gamma[oc] and beta[oc] already extracted by caller.
// ============================================================
#define GN_GROUPS   32
#define GN_GS       (HEAD_FEAT_CH / GN_GROUPS)   // 8 channels per group
#define GN_EPS      1e-5f

inline void apply_group_norm(
    act_t* buf,
    const weight_t* gamma,
    const weight_t* beta,
    int H, int W
) {
    const int n_per_group = GN_GS * H * W;
    GN_G: for (int g = 0; g < GN_GROUPS; g++) {
        float sum = 0.0f;
        GN_M_C: for (int c = g*GN_GS; c < (g+1)*GN_GS; c++)
            GN_M_S: for (int i = 0; i < H*W; i++)
                sum += (float)buf[c*H*W + i];
        float mean = sum / (float)n_per_group;

        float var_sum = 0.0f;
        GN_V_C: for (int c = g*GN_GS; c < (g+1)*GN_GS; c++)
            GN_V_S: for (int i = 0; i < H*W; i++) {
                float d = (float)buf[c*H*W + i] - mean;
                var_sum += d * d;
            }
        float inv_std = 1.0f / sqrtf(var_sum / (float)n_per_group + GN_EPS);

        GN_N_C: for (int c = g*GN_GS; c < (g+1)*GN_GS; c++) {
            float gv = (float)gamma[c];
            float bv = (float)beta[c];
            GN_N_S: for (int i = 0; i < H*W; i++) {
                float v = ((float)buf[c*H*W + i] - mean) * inv_std;
                v = v * gv + bv;
                buf[c*H*W + i] = (act_t)(v > 0.0f ? v : 0.0f);  // ReLU
            }
        }
    }
}

// ============================================================
// STACKED 3×3 CONVOLUTIONS (shared across FPN levels)
// in_ch=256 for all layers; 4 stacked convs with GroupNorm+ReLU.
// One set for cls branch, one set for reg branch.
// ============================================================
inline void retina_stacked_convs(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,       // flat: 4 × [256*256*9 + 256+256]
    int H, int W
) {
    // Ping-pong: even layers (0,2) write to tmp; odd layers (1,3) write to output.
    // With HEAD_STACKED_CONVS=4, the final write (lyr=3) goes to output — no copy needed.
    static act_t tmp[HEAD_FEAT_CH * P2_H * P2_W];  // max size P2: 256*160*160
    #pragma HLS RESOURCE variable=tmp core=RAM_T2P_BRAM

    int off = 0;
    const act_t* cur = input;

    // 4 stacked 3x3 convolutions.
    // Model uses plain Conv2d(bias=True)+ReLU with no GroupNorm (norm_cfg not set).
    // Weight layout per layer: conv_w[256*256*9], scale[256]=ones, bias[256]=conv.bias
    // Correct computation: output = relu(conv(x) * scale[oc] + bias[oc])
    //                              = relu(conv(x) + conv_bias[oc])   (scale is always 1)
    RSC: for (int lyr = 0; lyr < HEAD_STACKED_CONVS; lyr++) {
        // Even layers write to tmp, odd layers write to output (ping-pong)
        act_t* nxt = (lyr % 2 == 0) ? tmp : output;

        const weight_t* lw = weights + off;
        const weight_t* bs = lw + HEAD_FEAT_CH * HEAD_FEAT_CH * 9;   // scale (=1 for plain conv)
        const bias_t*   bb = (const bias_t*)(bs + HEAD_FEAT_CH);      // conv bias

        int H2 = H, W2 = W;
        FPN3_SC_OC: for (int oc = 0; oc < HEAD_FEAT_CH; oc++) {
            FPN3_SC_OH: for (int oh = 0; oh < H2; oh++) {
                FPN3_SC_OW: for (int ow = 0; ow < W2; ow++) {
                    #pragma HLS PIPELINE II=1
                    acc_t acc = 0;
                    FPN3_SC_IC: for (int ic = 0; ic < HEAD_FEAT_CH; ic++) {
                        FPN3_SC_KH: for (int kh = 0; kh < 3; kh++) {
                            FPN3_SC_KW: for (int kw = 0; kw < 3; kw++) {
                                int ih = oh + kh - 1;
                                int iw = ow + kw - 1;
                                if (ih >= 0 && ih < H2 && iw >= 0 && iw < W2) {
                                    acc += (acc_t)cur[ic*H2*W2 + ih*W2 + iw]
                                         * (acc_t)lw[(oc*HEAD_FEAT_CH+ic)*9 + kh*3+kw];
                                }
                            }
                        }
                    }
                    // Per-channel affine + ReLU: relu(acc * scale[oc] + bias[oc])
                    act_t y = (act_t)((acc_t)acc * (acc_t)bs[oc] + (acc_t)bb[oc]);
                    nxt[oc*H2*W2 + oh*W2 + ow] = (y > (act_t)0) ? y : (act_t)0;
                }
            }
        }
        off += HEAD_FEAT_CH*HEAD_FEAT_CH*9 + HEAD_FEAT_CH + HEAD_FEAT_CH;
        cur = nxt;
    }
    // lyr=3 wrote to output; result already in place, no copy needed.
}

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
                RP_IC: for (int ic = 0; ic < HEAD_FEAT_CH; ic++) {
                    RP_KH: for (int kh = 0; kh < 3; kh++) {
                        RP_KW: for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += (acc_t)input[ic*H*W + ih*W + iw]
                                     * (acc_t)weights[(oc*HEAD_FEAT_CH+ic)*9 + kh*3+kw];
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

// class_ids: per-candidate class label. Suppression only applied within the same class,
// matching MMDetection's multiclass_nms behaviour.
int nms(
    const bbox_t*  boxes,      // [N * 4]
    const score_t* scores,     // [N]
    int*           order,      // [N]  – sorted indices (pre-sorted by caller)
    int*           keep,       // [MAX_DETS]  output
    const int*     class_ids,  // [N]  – class label per candidate
    int N,
    float iou_thr
) {
    #pragma HLS INLINE

    // Sized for global pool (5 levels × NMS_PRE candidates each)
    static bool suppressed[5 * NMS_PRE];
    #pragma HLS RESOURCE variable=suppressed core=RAM_T2P_BRAM

    for (int i = 0; i < N; i++) suppressed[i] = false;

    int num_keep = 0;
    NMS_I: for (int i = 0; i < N && num_keep < MAX_DETS; i++) {
        int ii = order[i];
        if (suppressed[ii]) continue;
        keep[num_keep++] = ii;
        NMS_J: for (int j = i + 1; j < N; j++) {
            int jj = order[j];
            if (suppressed[jj]) continue;
            if (class_ids[ii] != class_ids[jj]) continue;  // only suppress within same class
            float iou = iou_f(boxes + ii*4, boxes + jj*4);
            if (iou > iou_thr) suppressed[jj] = true;
        }
    }
    return num_keep;
}

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
// PROCESS ONE FPN LEVEL
// Runs stacked convs, pred convs, generates anchors, decodes
// boxes, applies score threshold, returns candidates.
// Returns number of candidates written to cand_boxes/cand_scores/cand_cls.
// ============================================================
int process_level(
    const act_t*    feat,         // [256, H, W]
    const weight_t* cls_conv_w,   // stacked cls conv weights
    const weight_t* reg_conv_w,   // stacked reg conv weights
    const weight_t* cls_pred_w,   const bias_t* cls_pred_b,
    const weight_t* reg_pred_w,   const bias_t* reg_pred_b,
    int H, int W, int stride,
    // Output candidates
    bbox_t*  cand_boxes,   // [NMS_PRE * 4]
    score_t* cand_scores,  // [NMS_PRE]
    int*     cand_cls,     // [NMS_PRE]
    int      max_cand
) {
    #pragma HLS INLINE

    // ---- Run cls and reg stacked convs ----
    static act_t cls_feat[HEAD_FEAT_CH * P2_H * P2_W];
    static act_t reg_feat[HEAD_FEAT_CH * P2_H * P2_W];
    #pragma HLS RESOURCE variable=cls_feat core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=reg_feat core=RAM_T2P_BRAM

    retina_stacked_convs(feat, cls_feat, cls_conv_w, H, W);
    retina_stacked_convs(feat, reg_feat, reg_conv_w, H, W);

    // ---- Prediction convs ----
    const int cls_out_ch = ANCHORS_PER_LOC * NUM_CLASSES;  // 9*4=36
    const int reg_out_ch = ANCHORS_PER_LOC * 4;            // 9*4=36

    static act_t cls_logits[ANCHORS_PER_LOC * NUM_CLASSES * P2_H * P2_W];
    static act_t reg_deltas[ANCHORS_PER_LOC * 4           * P2_H * P2_W];
    #pragma HLS RESOURCE variable=cls_logits core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=reg_deltas core=RAM_T2P_BRAM

    retina_pred_conv(cls_feat, cls_logits, cls_pred_w, cls_pred_b, cls_out_ch, H, W);
    retina_pred_conv(reg_feat, reg_deltas, reg_pred_w, reg_pred_b, reg_out_ch, H, W);

#if DEBUG_HEAD_DUMP
    // Dump P3 (80x80) intermediates for comparison with Python
    if (H == 80 && W == 80) {
        auto dump_f32 = [](const char* path, const act_t* data, int n) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
            float mn = data[0], mx = data[0], sm = 0;
            for (int i = 0; i < n && i < 1000000; i++) { sm += data[i]; if(data[i]<mn)mn=data[i]; if(data[i]>mx)mx=data[i]; }
            std::cerr << "  dump " << path << "  n=" << n
                      << "  min=" << mn << "  max=" << mx
                      << "  mean=" << sm/n
                      << "  first5=[" << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << "," << data[4] << "]\n";
        };
        dump_f32("../csim_validation/reg_feat_p3_csim.bin",   reg_feat,   HEAD_FEAT_CH * 80 * 80);
        dump_f32("../csim_validation/cls_feat_p3_csim.bin",   cls_feat,   HEAD_FEAT_CH * 80 * 80);
        dump_f32("../csim_validation/reg_deltas_p3_csim.bin", reg_deltas, ANCHORS_PER_LOC * 4 * 80 * 80);
        dump_f32("../csim_validation/cls_logits_p3_csim.bin", cls_logits, ANCHORS_PER_LOC * NUM_CLASSES * 80 * 80);
        // Print dh stats (channel 3, 7, 11, ... are dh for anchors 0,1,2,...)
        float dh_min=1e9, dh_max=-1e9, dh_sum=0;
        int dh_clamped = 0;
        for (int a = 0; a < ANCHORS_PER_LOC; a++) {
            const act_t* dh_ch = reg_deltas + (a*4+3)*80*80;
            for (int i = 0; i < 80*80; i++) {
                float v = dh_ch[i];
                dh_sum += v; if(v<dh_min)dh_min=v; if(v>dh_max)dh_max=v;
                if(v >= 4.135f) dh_clamped++;
            }
        }
        std::cerr << "  P3 dh: min=" << dh_min << " max=" << dh_max
                  << " mean=" << dh_sum/(ANCHORS_PER_LOC*80*80)
                  << " clamped@4.135: " << dh_clamped << "/" << ANCHORS_PER_LOC*80*80 << "\n";
    }
#endif

    // ---- Inline anchor generation + box decoding + per-class score filtering ----
    // Eliminates anchors[n_anchors*4], delta_flat[n_anchors*4], boxes_flat[n_anchors*4]
    // which would overflow by 57x at P2 (160*160*9 = 230400 anchors vs NMS_PRE=1000).
    // Each class scored independently (multi-label) matching MMDetection behaviour.
    int num_cand = 0;

    SCORE_FILT: for (int h = 0; h < H && num_cand < max_cand; h++) {
        for (int w_i = 0; w_i < W && num_cand < max_cand; w_i++) {
            float cx = (w_i + 0.5f) * stride;
            float cy = (h   + 0.5f) * stride;

            for (int ri = 0; ri < ANCHOR_RATIOS_N && num_cand < max_cand; ri++) {
                for (int si = 0; si < ANCHOR_SCALES_N && num_cand < max_cand; si++) {
                    int a = ri * ANCHOR_SCALES_N + si;

                    float anc_scale = ANCHOR_SCALES_V[si] * stride;
                    float ratio     = ANCHOR_RATIOS_V[ri];
                    float aw = anc_scale / sqrtf(ratio);
                    float ah = anc_scale * sqrtf(ratio);

                    for (int c = 0; c < NUM_CLASSES && num_cand < max_cand; c++) {
                        #pragma HLS PIPELINE II=1
                        int cls_ch = a * NUM_CLASSES + c;
                        score_t s = sigmoid((score_t)cls_logits[cls_ch * H * W + h * W + w_i]);

                        if ((float)s >= SCORE_THR) {
                            float dx = (float)reg_deltas[(a*4+0) * H * W + h * W + w_i];
                            float dy = (float)reg_deltas[(a*4+1) * H * W + h * W + w_i];
                            float dw = (float)reg_deltas[(a*4+2) * H * W + h * W + w_i];
                            float dh = (float)reg_deltas[(a*4+3) * H * W + h * W + w_i];

                            if (dw < -4.135f) dw = -4.135f;
                            if (dh < -4.135f) dh = -4.135f;
                            if (dw >  4.135f) dw =  4.135f;
                            if (dh >  4.135f) dh =  4.135f;

                            float pred_cx = dx * aw + cx;
                            float pred_cy = dy * ah + cy;
                            float pred_w  = expf(dw) * aw;
                            float pred_h  = expf(dh) * ah;

                            float x1 = fmaxf(pred_cx - pred_w * 0.5f, 0.0f);
                            float y1 = fmaxf(pred_cy - pred_h * 0.5f, 0.0f);
                            float x2 = fminf(pred_cx + pred_w * 0.5f, (float)INPUT_W);
                            float y2 = fminf(pred_cy + pred_h * 0.5f, (float)INPUT_H);
                            cand_boxes[num_cand*4+0] = (bbox_t)x1;
                            cand_boxes[num_cand*4+1] = (bbox_t)y1;
                            cand_boxes[num_cand*4+2] = (bbox_t)x2;
                            cand_boxes[num_cand*4+3] = (bbox_t)y2;
                            cand_scores[num_cand]    = s;
                            cand_cls   [num_cand]    = c;
                            num_cand++;
                        }
                    }
                }
            }
        }
    }

    return num_cand;
}

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

    // ---- Pass 1: collect topk candidates from each level into global pool ----
    PER_LEVEL: for (int lvl = 0; lvl < 5; lvl++) {

        const act_t* lvl_feat;
        int H, W, stride;

        if (lvl == 0) {
            lvl_feat = p2;
            H = P2_H; W = P2_W; stride = FPN_STRIDES[0];
        }
        else if (lvl == 1) {
            lvl_feat = p3;
            H = P3_H; W = P3_W; stride = FPN_STRIDES[1];
        }
        else if (lvl == 2) {
            lvl_feat = p4;
            H = P4_H; W = P4_W; stride = FPN_STRIDES[2];
        }
        else if (lvl == 3) {
            lvl_feat = p5;
            H = P5_H; W = P5_W; stride = FPN_STRIDES[3];
        }
        else {
            lvl_feat = p6;
            H = P6_H; W = P6_W; stride = FPN_STRIDES[4];
        }

        int nc = process_level(
            lvl_feat,
            head_cls_w, head_reg_w,
            cls_pred_w, cls_pred_b,
            reg_pred_w, reg_pred_b,
            H, W, stride,
            cand_boxes, cand_scores, cand_cls,
            CAND_BUF_SIZE
        );

        // Per-level topk (matches MMDetection nms_pre per level)
        sort_scores(cand_scores, cand_order, nc);
        int nc_topk = (nc < NMS_PRE) ? nc : NMS_PRE;

        // Append into global pool (guard against overflow)
        int space  = 5 * NMS_PRE - global_nc;
        int copy_n = (nc_topk < space) ? nc_topk : space;
        COPY_LVL: for (int k = 0; k < copy_n; k++) {
            int ki = cand_order[k];
            global_boxes [(global_nc + k) * 4 + 0] = cand_boxes[ki * 4 + 0];
            global_boxes [(global_nc + k) * 4 + 1] = cand_boxes[ki * 4 + 1];
            global_boxes [(global_nc + k) * 4 + 2] = cand_boxes[ki * 4 + 2];
            global_boxes [(global_nc + k) * 4 + 3] = cand_boxes[ki * 4 + 3];
            global_scores[global_nc + k] = cand_scores[ki];
            global_cls   [global_nc + k] = cand_cls[ki];
        }
        global_nc += copy_n;
    }

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

    // ============================================================
    // BACKUP — old per-level NMS (uncomment block below to revert)
    // ============================================================
    /*
    static bbox_t  cand_boxes [CAND_BUF_SIZE * 4];
    static score_t cand_scores[CAND_BUF_SIZE];
    static int     cand_cls   [CAND_BUF_SIZE];
    static int     sort_order [CAND_BUF_SIZE];
    static int     keep_idx   [MAX_DETS];
    static bbox_t  all_boxes [MAX_DETS * 4];
    static score_t all_scores[MAX_DETS];
    static int     all_cls   [MAX_DETS];

    int total_dets = 0;
    PER_LEVEL: for (int lvl = 0; lvl < 5 && total_dets < MAX_DETS; lvl++) {
        const act_t* lvl_feat;
        int H, W, stride;
        if      (lvl == 0) { lvl_feat = p2; H = P2_H; W = P2_W; stride = FPN_STRIDES[0]; }
        else if (lvl == 1) { lvl_feat = p3; H = P3_H; W = P3_W; stride = FPN_STRIDES[1]; }
        else if (lvl == 2) { lvl_feat = p4; H = P4_H; W = P4_W; stride = FPN_STRIDES[2]; }
        else if (lvl == 3) { lvl_feat = p5; H = P5_H; W = P5_W; stride = FPN_STRIDES[3]; }
        else               { lvl_feat = p6; H = P6_H; W = P6_W; stride = FPN_STRIDES[4]; }
        int nc = process_level(lvl_feat, head_cls_w, head_reg_w,
                               cls_pred_w, cls_pred_b, reg_pred_w, reg_pred_b,
                               H, W, stride, cand_boxes, cand_scores, cand_cls, CAND_BUF_SIZE);
        sort_scores(cand_scores, sort_order, nc);
        int nc_topk = (nc < NMS_PRE) ? nc : NMS_PRE;
        int nk = nms(cand_boxes, cand_scores, sort_order, keep_idx, cand_cls, nc_topk, NMS_IOU_THR);
        for (int k = 0; k < nk && total_dets < MAX_DETS; k++) {
            int ki = keep_idx[k];
            detections[total_dets].x1       = cand_boxes[ki*4+0];
            detections[total_dets].y1       = cand_boxes[ki*4+1];
            detections[total_dets].x2       = cand_boxes[ki*4+2];
            detections[total_dets].y2       = cand_boxes[ki*4+3];
            detections[total_dets].score    = cand_scores[ki];
            detections[total_dets].class_id = cand_cls[ki];
            total_dets++;
        }
    }
    for (int i = 1; i < total_dets; i++) {
        Detection tmp = detections[i]; int j = i - 1;
        while (j >= 0 && detections[j].score < tmp.score) { detections[j+1] = detections[j]; j--; }
        detections[j+1] = tmp;
    }
    *num_dets = total_dets;
    */

}

#endif // RETINA_HEAD_H
