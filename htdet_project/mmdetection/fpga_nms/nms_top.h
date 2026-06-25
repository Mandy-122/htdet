/*
 * nms_top.h
 * Isolated score-filter + NMS module for Vitis HLS C-synthesis.
 *
 * Takes pre-computed cls_logits + reg_deltas for ONE FPN level
 * (output of fpga_retina_head), and produces the final Detection list.
 *
 * ── Pipeline ───────────────────────────────────────────────────────────────
 *
 *  cls_logits [CLS_OUT_CH × TEST_H × TEST_W]   (CHW, pre-sigmoid)
 *  reg_deltas [REG_OUT_CH × TEST_H × TEST_W]   (CHW, pre-exp)
 *       │
 *       ▼
 *  [1] score_filter  — for each (h,w,anchor,class):
 *                        s = sigmoid(logit)
 *                        if s >= SCORE_THR (0.20):
 *                          decode box from anchor + delta
 *                          append to cand_boxes / cand_scores / cand_cls
 *
 *       ▼
 *  [2] sort_scores   — insertion sort, descending by score
 *                      keep topK = min(num_cand, NMS_PRE=1000)
 *
 *       ▼
 *  [3] nms_greedy    — greedy multiclass NMS (IoU threshold = 0.5)
 *                      suppresses lower-score boxes with same class and IoU > thr
 *
 *       ▼
 *  detections [MAX_DETS]   num_dets
 *
 * ── HLS synthesis notes ────────────────────────────────────────────────────
 *
 *  score_filter: the innermost class loop is PIPELINE II=1.
 *    Data-dependent loop bounds (num_cand < CAND_BUF_SIZE guard) prevent
 *    full trip-count knowledge; HLS uses TRIPCOUNT pragma for latency estimate.
 *    Box coords are converted float→coord_t (ap_ufixed<18,11>) here [OPT-8].
 *
 *  sort_scores: insertion sort with a bounded inner for-loop (replaces while).
 *    Outer loop = O(N), inner loop worst-case O(N) → O(N²) total.
 *    For N=NMS_PRE=300: ~45K iterations.  Synthesis reports ~4K cycles (II=4).
 *    SORT_J is not the latency bottleneck; NMS_greedy dominates.
 *
 *  nms_greedy: doubly-nested, data-dependent 'continue' (suppressed[] check).
 *    Inter-iteration dependency on suppressed[]: outer reads suppressed[ii],
 *    inner writes suppressed[jj] — HLS cannot pipeline across outer iterations.
 *    NMS_J is pipelined; achieved II reported as ~10-12 with fixed-point [OPT-8].
 *    For N=NMS_PRE=300: up to N*(N-1)/2 = 44850 inner iters × II cycles.
 *    Estimated latency with fixed-point: ~540K cycles per FPN level.
 *
 * ── Element counts ─────────────────────────────────────────────────────────
 *
 *  Input logits:  CLS_LOGITS_ELEMS = CLS_OUT_CH × TEST_H × TEST_W
 *  Input deltas:  REG_DELTAS_ELEMS = REG_OUT_CH × TEST_H × TEST_W
 *  For P6 (10×10): 36×10×10 = 3 600 each
 */

#ifndef NMS_TOP_H
#define NMS_TOP_H

#include "fpga_types.h"

#define CLS_LOGITS_ELEMS   (CLS_OUT_CH * TEST_H * TEST_W)
#define REG_DELTAS_ELEMS   (REG_OUT_CH * TEST_H * TEST_W)

void nms_top(
    const act_t  cls_logits [CLS_LOGITS_ELEMS],   // [36, H, W] CHW, pre-sigmoid
    const act_t  reg_deltas [REG_DELTAS_ELEMS],   // [36, H, W] CHW, raw deltas
    Detection    detections [MAX_DETS],
    int*         num_dets
);

#endif // NMS_TOP_H
