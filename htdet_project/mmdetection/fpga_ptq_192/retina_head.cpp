/*
 * retina_head.cpp  (PTQ INT8 variant)
 * RetinaNet head — non-inline function definitions (Vitis HLS).
 * Add this file to the Vitis HLS project alongside retina_head.h.
 *
 * W8A32 PTQ changes:
 *   - retina_conv_once: int8 weights + float w_dequant_scale + float bias
 *   - process_level: split cls/reg conv and pred pointers
 *   - ARRAY_PARTITION cyclic factor=8 → 16
 *   - #pragma HLS RESOURCE → #pragma HLS bind_storage
 */

#include "retina_head.h"

// ============================================================
// Single conv3x3 + per-channel dequant scale + per-channel bias + ReLU.
// lw: int8 conv weights [HEAD_FEAT_CH * HEAD_FEAT_CH * 9]
// bs: float per-channel w_dequant_scale [HEAD_FEAT_CH]
// bb: float per-channel bias [HEAD_FEAT_CH]
// output = ReLU(conv(input, lw) * bs[oc] + bb[oc])
// ============================================================
void retina_conv_once(
    const act_t*    src,
    act_t*          dst,
    const weight_t* lw,
    const meta_t*   bs,
    const meta_t*   bb,
    int H, int W
) {
    CO_OC: for (int oc = 0; oc < HEAD_FEAT_CH; oc++) {
        CO_OH: for (int oh = 0; oh < H; oh++) {
            CO_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                CO_ICT: for (int icb = 0; icb < HEAD_FEAT_CH; icb += HEAD_IC_TILE) {
                    CO_IC: for (int ic = icb; ic < icb + HEAD_IC_TILE; ic++) {
                        #pragma HLS UNROLL
                        CO_KH: for (int kh = 0; kh < 3; kh++) {
                            #pragma HLS UNROLL
                            CO_KW: for (int kw = 0; kw < 3; kw++) {
                                #pragma HLS UNROLL
                                int ih = oh + kh - 1;
                                int iw = ow + kw - 1;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    acc += (acc_t)src[ic*H*W + ih*W + iw]
                                         * (acc_t)(float)lw[(oc*HEAD_FEAT_CH+ic)*9 + kh*3+kw];
                                }
                            }
                        }
                    }
                }
                act_t y = (act_t)(acc * (acc_t)bs[oc] + (acc_t)bb[oc]);
                dst[oc*H*W + oh*W + ow] = (y > (act_t)0) ? y : (act_t)0;
            }
        }
    }
}

// ============================================================
// NMS (greedy, multiclass, sorted descending by score)
// ============================================================
int nms(
    const bbox_t*  boxes,
    const score_t* scores,
    int*           order,
    int*           keep,
    const int*     class_ids,
    int N,
    float iou_thr
) {
    #pragma HLS INLINE

    static bool suppressed[5 * NMS_PRE];
    #pragma HLS bind_storage variable=suppressed type=RAM_T2P impl=BRAM

    for (int i = 0; i < N; i++) suppressed[i] = false;

    int num_keep = 0;
    NMS_I: for (int i = 0; i < N && num_keep < MAX_DETS; i++) {
        int ii = order[i];
        if (suppressed[ii]) continue;
        keep[num_keep++] = ii;
        NMS_J: for (int j = i + 1; j < N; j++) {
            int jj = order[j];
            if (suppressed[jj]) continue;
            if (class_ids[ii] != class_ids[jj]) continue;
            float iou = iou_f(boxes + ii*4, boxes + jj*4);
            if (iou > iou_thr) suppressed[jj] = true;
        }
    }
    return num_keep;
}

// ============================================================
// PROCESS ONE FPN LEVEL
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
) {

    static act_t cls_feat[HEAD_FEAT_CH * P2_H * P2_W];
    static act_t reg_feat[HEAD_FEAT_CH * P2_H * P2_W];
    #pragma HLS bind_storage variable=cls_feat type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=reg_feat type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=cls_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=reg_feat cyclic factor=16 dim=1

    retina_stacked_convs(feat, cls_feat, cls_conv_int8, cls_conv_meta, H, W);
    retina_stacked_convs(feat, reg_feat, reg_conv_int8, reg_conv_meta, H, W);

    const int cls_out_ch = ANCHORS_PER_LOC * NUM_CLASSES;
    const int reg_out_ch = ANCHORS_PER_LOC * 4;

    static act_t cls_logits[ANCHORS_PER_LOC * NUM_CLASSES * P2_H * P2_W];
    static act_t reg_deltas[ANCHORS_PER_LOC * 4           * P2_H * P2_W];
    #pragma HLS bind_storage variable=cls_logits type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=reg_deltas type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=cls_logits cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=reg_deltas cyclic factor=4 dim=1

    retina_pred_conv(cls_feat, cls_logits, cls_pred_w, cls_pred_scale, cls_pred_b, cls_out_ch, H, W);
    retina_pred_conv(reg_feat, reg_deltas, reg_pred_w, reg_pred_scale, reg_pred_b, reg_out_ch, H, W);

#if DEBUG_HEAD_DUMP
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
        dump_f32("../csim_validation/reg_feat_p3_csim_ptq.bin",   reg_feat,   HEAD_FEAT_CH * 80 * 80);
        dump_f32("../csim_validation/cls_feat_p3_csim_ptq.bin",   cls_feat,   HEAD_FEAT_CH * 80 * 80);
        dump_f32("../csim_validation/reg_deltas_p3_csim_ptq.bin", reg_deltas, ANCHORS_PER_LOC * 4 * 80 * 80);
        dump_f32("../csim_validation/cls_logits_p3_csim_ptq.bin", cls_logits, ANCHORS_PER_LOC * NUM_CLASSES * 80 * 80);
    }
#endif

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
