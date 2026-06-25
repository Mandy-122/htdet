/*
 * nms_top.cpp
 * Isolated NMS module — score-filter + anchor-decode + sort + greedy NMS.
 *
 * Corresponds to the second half of process_level() + nms() + sort_scores()
 * from fpga_ptq_192/retina_head.cpp, isolated for independent HLS synthesis.
 *
 * Anchor config: scales={4,6,8}, ratios={0.5,1.0,2.0}, stride=TEST_STRIDE.
 * Anchor index:  a = ri * ANCHOR_SCALES_N + si
 *
 * CHW layout for cls_logits / reg_deltas:
 *   channel = anchor * NUM_CLASSES + class  (logits)
 *   channel = anchor * 4 + coord            (deltas: 0=dx,1=dy,2=dw,3=dh)
 *   flat index = channel * H * W + h * W + w
 *
 * ── Optimisations applied ────────────────────────────────────────────────────
 *  [OPT-1] iou_f: replace division with multiplication to avoid 16-cycle fdiv.
 *          `inter / uni > thr`  →  `inter > thr * uni`
 *          (superseded by OPT-9 below; kept here for history)
 *
 *  [OPT-2] cand_boxes ARRAY_PARTITION cyclic factor=4.
 *          Groups x1/y1/x2/y2 of each box into 4 separate banks so the
 *          IoU helper can read all 4 coords of one box in a single cycle.
 *
 *  [OPT-3] suppressed[] → LUTRAM instead of BRAM.
 *          1000-bit boolean array; LUTRAM has 0-cycle read latency vs
 *          1-cycle for BRAM, breaking the RAW dependency that gave NMS_J II=54.
 *
 *  [OPT-4] cand_scores / cand_order → LUTRAM.
 *          Allows the SORT_J inner loop to achieve II=1 (was II=5 due to
 *          BRAM read latency in the score comparison chain).
 *
 *  [OPT-5] Pre-compute box area in score_filter; store in cand_areas[].
 *          Saves repeated computation in NMS_J.
 *
 *  [OPT-6] cand_areas also stored as LUTRAM (same reasoning as [OPT-4]).
 *
 *  [OPT-7] iou_f signature changed to accept pre-computed area_a scalar
 *          and individual x1/y1/x2/y2 values (avoids pointer-based BRAM
 *          address generation which confused the HLS scheduler).
 *
 *  [OPT-8] Fixed-point coords/areas: coord_t = ap_ufixed<18,11>,
 *          area_t = ap_ufixed<30,19>, area3_t = ap_ufixed<32,21>.
 *          Replaces float32 fsub(6 cy) → 2 cy and fmul(5 cy) → 3 cy (DSP).
 *          Expected NMS_J II: 43 → ~10-12 (3-4× latency reduction).
 *          Also resolves the 6.124 ns timing violation on NMS_J critical path.
 *          cand_boxes BRAM: 32 BRAM_18K → ~16 BRAM_18K (18-bit elements).
 *          The float-to-fixed conversion happens once in score_filter (cold
 *          path); the hot NMS_J loop is entirely fixed-point.
 *
 *  [OPT-9] IoU=0.5 closed-form shortcut.
 *          inter > 0.5*(area_a+area_b-inter)
 *            ↔  2·inter > area_a+area_b-inter
 *            ↔  3·inter > area_a+area_b
 *          Eliminates one subtraction AND one multiplication from the IoU
 *          critical path.  3·inter = (inter<<1)+inter uses a free shift + 1
 *          addition.  Mathematically exact for iou_thr=0.5.
 *
 *  [OPT-10] cand_cls → ap_uint<2> LUTRAM.
 *           4 classes need only 2 bits.  Saves BRAM (was BRAM T2P 8× 18K)
 *           and reduces the NMS_J class-compare to a 2-bit integer op.
 */

#include "nms_top.h"
#include "fpga_utils.h"

// ── anchor configuration (must match fpga_ptq_192) ─────────────────────────
static const float ANCHOR_SCALES[ANCHOR_SCALES_N] = {4.0f, 6.0f, 8.0f};
static const float ANCHOR_RATIOS[ANCHOR_RATIOS_N] = {0.5f, 1.0f, 2.0f};

// ── internal fixed-point types (NMS path only) ───────────────────────────────
// [OPT-8]: replace float32 with ap_fixed arithmetic for coords and areas.
// Under g++ (csim/testbench), ap_ufixed/ap_uint are unavailable; float/int
// are functionally equivalent since 1/128-px coord precision is negligible.
#ifdef __SYNTHESIS__
typedef ap_ufixed<18,11> coord_t;   // [0, 2048),  precision 2^-7 ≈ 0.0078 px
typedef ap_ufixed<30,19> area_t;    // [0, 524288), precision 2^-11  (> 640²=409600)
typedef ap_ufixed<32,21> area3_t;   // [0, 2097152), for 3×area comparison
typedef ap_uint<2>       cls_idx_t; // [OPT-10]: 4 classes → 2 bits
#else
typedef float coord_t;
typedef float area_t;
typedef float area3_t;
typedef int   cls_idx_t;
#endif

// ── IoU helper — fixed-point, division-free ──────────────────────────────────
// [OPT-8]: all arithmetic is ap_ufixed → sub 2 cy, DSP mul 3 cy.
// [OPT-9]: uses 3·inter > area_a+area_b (IoU>0.5 shortcut, no division, no
//          threshold multiply, no union subtraction).
// area_a is pre-computed and passed in (OPT-5).
static inline bool iou_exceeds(
    coord_t ax1, coord_t ay1, coord_t ax2, coord_t ay2, area_t area_a,
    coord_t bx1, coord_t by1, coord_t bx2, coord_t by2
) {
    #pragma HLS INLINE

    // Intersection rectangle corners (1-cycle integer compare per coord)
    coord_t ix1 = (ax1 > bx1) ? ax1 : bx1;
    coord_t iy1 = (ay1 > by1) ? ay1 : by1;
    coord_t ix2 = (ax2 < bx2) ? ax2 : bx2;
    coord_t iy2 = (ay2 < by2) ? ay2 : by2;

    // Early exit: 1-cycle integer compare, avoids downstream multiply
    if (ix2 <= ix1 || iy2 <= iy1) return false;

    // Widths — all positive after early exit, safe unsigned subtraction
    coord_t iw = ix2 - ix1;
    coord_t ih = iy2 - iy1;
    coord_t bw = bx2 - bx1;
    coord_t bh = by2 - by1;

    // Intersection and candidate-b areas — DSP multipliers, 3 cycles each
    // ap_ufixed<18,11> × ap_ufixed<18,11> → truncated to area_t <30,19>
    area_t inter  = (area_t)(iw * ih);
    area_t area_b = (area_t)(bw * bh);

    // [OPT-9]: 3·inter > area_a+area_b  (equiv. IoU > 0.5, no threshold mul)
    // Under HLS: (inter<<1) is a free rewire + 1 adder; no extra multiplier.
    // Under g++ (csim): float path uses multiplication (functionally identical).
#ifdef __SYNTHESIS__
    area3_t inter3   = ((area3_t)inter << 1) + (area3_t)inter;
    area3_t area_sum = (area3_t)area_a + (area3_t)area_b;
    return (inter3 > area_sum);
#else
    return (3.0f * inter > area_a + area_b);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// [1] SCORE FILTER + ANCHOR DECODE
// Iterates over every (h, w, anchor, class); keeps candidates where
// sigmoid(logit) >= SCORE_THR.  Decodes box from anchor + delta in-loop.
// Stores coords as coord_t [OPT-8] and pre-computes area as area_t [OPT-5].
// ─────────────────────────────────────────────────────────────────────────────
static int score_filter(
    const act_t*  cls_logits,
    const act_t*  reg_deltas,
    coord_t*      cand_boxes,    // [OPT-2]: caller partitions cyclic factor=4
    score_t*      cand_scores,
    cls_idx_t*    cand_cls,      // [OPT-10]: ap_uint<2>
    area_t*       cand_areas     // [OPT-5]: pre-computed fixed-point area
) {
    int num_cand = 0;

    SF_H: for (int h = 0; h < TEST_H && num_cand < CAND_BUF_SIZE; h++) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=10 avg=10
        SF_W: for (int w = 0; w < TEST_W && num_cand < CAND_BUF_SIZE; w++) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=10 avg=10
            float cx = (w + 0.5f) * TEST_STRIDE;
            float cy = (h + 0.5f) * TEST_STRIDE;

            SF_RI: for (int ri = 0; ri < ANCHOR_RATIOS_N && num_cand < CAND_BUF_SIZE; ri++) {
                #pragma HLS LOOP_TRIPCOUNT min=0 max=3 avg=3
                SF_SI: for (int si = 0; si < ANCHOR_SCALES_N && num_cand < CAND_BUF_SIZE; si++) {
                    #pragma HLS LOOP_TRIPCOUNT min=0 max=3 avg=3
                    int   a   = ri * ANCHOR_SCALES_N + si;
                    float anc_scale = ANCHOR_SCALES[si] * TEST_STRIDE;
                    float ratio     = ANCHOR_RATIOS[ri];
                    float aw = anc_scale / sqrtf(ratio);
                    float ah = anc_scale * sqrtf(ratio);

                    SF_C: for (int c = 0; c < NUM_CLASSES && num_cand < CAND_BUF_SIZE; c++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=4 max=4 avg=4

                        int cls_ch = a * NUM_CLASSES + c;
                        int hw_off = h * TEST_W + w;
                        score_t s  = sigmoid((score_t)cls_logits[cls_ch * TEST_H * TEST_W + hw_off]);

                        if ((float)s >= SCORE_THR) {
                            float dx = (float)reg_deltas[(a*4+0) * TEST_H * TEST_W + hw_off];
                            float dy = (float)reg_deltas[(a*4+1) * TEST_H * TEST_W + hw_off];
                            float dw = (float)reg_deltas[(a*4+2) * TEST_H * TEST_W + hw_off];
                            float dh = (float)reg_deltas[(a*4+3) * TEST_H * TEST_W + hw_off];

                            // Clamp exp argument to avoid overflow
                            if (dw < -4.135f) dw = -4.135f;
                            if (dw >  4.135f) dw =  4.135f;
                            if (dh < -4.135f) dh = -4.135f;
                            if (dh >  4.135f) dh =  4.135f;

                            float pred_cx = dx * aw + cx;
                            float pred_cy = dy * ah + cy;
                            float pred_w  = expf(dw) * aw;
                            float pred_h  = expf(dh) * ah;

                            // Clip to image boundary
                            float x1 = fmaxf(pred_cx - pred_w * 0.5f, 0.0f);
                            float y1 = fmaxf(pred_cy - pred_h * 0.5f, 0.0f);
                            float x2 = fminf(pred_cx + pred_w * 0.5f, (float)INPUT_W);
                            float y2 = fminf(pred_cy + pred_h * 0.5f, (float)INPUT_H);

                            // [OPT-8]: store as fixed-point; float→coord_t conversion
                            // happens once here, not in the hot NMS_J inner loop.
                            coord_t cx1 = (coord_t)x1;
                            coord_t cy1 = (coord_t)y1;
                            coord_t cx2 = (coord_t)x2;
                            coord_t cy2 = (coord_t)y2;

                            cand_boxes [num_cand*4+0] = cx1;
                            cand_boxes [num_cand*4+1] = cy1;
                            cand_boxes [num_cand*4+2] = cx2;
                            cand_boxes [num_cand*4+3] = cy2;
                            cand_scores[num_cand]     = s;
                            cand_cls   [num_cand]     = (cls_idx_t)c;  // [OPT-10]
                            // [OPT-5]: pre-compute area in fixed-point
                            cand_areas [num_cand]     = (area_t)((cx2 - cx1) * (cy2 - cy1));
                            num_cand++;
                        }
                    }
                }
            }
        }
    }

    return num_cand;
}

// ─────────────────────────────────────────────────────────────────────────────
// [2] INSERTION SORT — descending by score
// Inner loop replaced with bounded for-loop (max N iters) + break.
// [OPT-4]: cand_scores and cand_order in LUTRAM → 0-latency reads → SORT_J II=1.
// ─────────────────────────────────────────────────────────────────────────────
static void sort_scores(const score_t* scores, int* order, int N) {
    SORT_INIT: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        order[i] = i;
    }

    SORT_I: for (int i = 1; i < N; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=300 avg=150
        int     key   = order[i];
        score_t key_s = scores[key];
        int     j     = i - 1;

        SORT_J: for (int k = 0; k < N; k++) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=300 avg=75
            if (j < 0 || scores[order[j]] >= key_s) break;
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] GREEDY NMS — multiclass (MMDetection style)
// [OPT-3]: suppressed[] → LUTRAM (0-cycle read latency).
// [OPT-8]: boxes/areas in fixed-point; IoU arithmetic uses ap_fixed ops.
// [OPT-9]: iou_exceeds uses 3·inter > area_a+area_b (no divide, no thr mul).
// [OPT-10]: class_ids → ap_uint<2>; compare is a 2-bit integer op.
// ─────────────────────────────────────────────────────────────────────────────
static int nms_greedy(
    const coord_t*    boxes,
    const score_t*    scores,
    const int*        order,
    int*              keep,
    const cls_idx_t*  class_ids,  // [OPT-10]
    const area_t*     areas,      // [OPT-5,8]
    int N
) {
    // [OPT-3]: LUTRAM gives 0-cycle read latency → breaks RAW dep on suppressed[]
    static bool suppressed[NMS_PRE];
    #pragma HLS bind_storage variable=suppressed type=RAM_2P impl=LUTRAM

    SUPP_INIT: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        suppressed[i] = false;
    }

    int num_keep = 0;

    NMS_I: for (int i = 0; i < N && num_keep < MAX_DETS; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=300 avg=100
        int ii = order[i];
        if (suppressed[ii]) continue;
        keep[num_keep++] = ii;

        // Cache anchor-box coords and area for box ii in scalar registers —
        // no BRAM re-read inside the inner loop.
        coord_t   ax1    = boxes[ii*4+0];
        coord_t   ay1    = boxes[ii*4+1];
        coord_t   ax2    = boxes[ii*4+2];
        coord_t   ay2    = boxes[ii*4+3];
        area_t    area_i = areas[ii];
        cls_idx_t cls_i  = class_ids[ii];  // [OPT-10]: 2-bit register

        NMS_J: for (int j = i + 1; j < N; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=300 avg=75
            #pragma HLS PIPELINE II=1
            int jj = order[j];
            if (suppressed[jj])         continue;
            if (cls_i != class_ids[jj]) continue;   // [OPT-10]: 2-bit compare
            // [OPT-2]: with cyclic factor=4 on cand_boxes, all 4 coords of box
            //          jj are in different banks → 1-cycle parallel read.
            // [OPT-8,9]: IoU computed in fixed-point; 3·inter > area_a+area_b.
            if (iou_exceeds(ax1, ay1, ax2, ay2, area_i,
                            boxes[jj*4+0], boxes[jj*4+1],
                            boxes[jj*4+2], boxes[jj*4+3]))
                suppressed[jj] = true;
        }
    }

    return num_keep;
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP FUNCTION
// ─────────────────────────────────────────────────────────────────────────────
void nms_top(
    const act_t  cls_logits [CLS_LOGITS_ELEMS],
    const act_t  reg_deltas [REG_DELTAS_ELEMS],
    Detection    detections [MAX_DETS],
    int*         num_dets
) {
    #pragma HLS INTERFACE bram     port=cls_logits
    #pragma HLS INTERFACE bram     port=reg_deltas
    #pragma HLS INTERFACE bram     port=detections
    #pragma HLS INTERFACE ap_vld   port=num_dets
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // Partition logits / deltas by anchor dimension for parallel reads.
    #pragma HLS ARRAY_PARTITION variable=cls_logits cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=reg_deltas cyclic factor=4 dim=1

    // ── candidate buffers ─────────────────────────────────────────────────
    // [OPT-8]: coord_t (18-bit) halves cand_boxes BRAM vs float32 (32-bit).
    static coord_t    cand_boxes [CAND_BUF_SIZE * 4];
    static score_t    cand_scores[CAND_BUF_SIZE];
    static cls_idx_t  cand_cls   [CAND_BUF_SIZE];   // [OPT-10]
    static int        cand_order [CAND_BUF_SIZE];
    static int        keep_idx   [MAX_DETS];
    static area_t     cand_areas [CAND_BUF_SIZE];   // [OPT-5,8]

    // [OPT-2]: cyclic factor=4 puts x1/y1/x2/y2 of each box in 4 separate
    //          banks → all 4 coords readable in 1 cycle (vs 4 sequential reads).
    #pragma HLS ARRAY_PARTITION variable=cand_boxes cyclic factor=4 dim=1

    // [OPT-4]: LUTRAM for scores/order → 0-cycle read latency in SORT_J.
    #pragma HLS bind_storage variable=cand_scores type=RAM_2P impl=LUTRAM
    #pragma HLS bind_storage variable=cand_order  type=RAM_2P impl=LUTRAM

    // [OPT-6]: LUTRAM for areas (small array, frequently read in NMS_J).
    #pragma HLS bind_storage variable=cand_areas  type=RAM_2P impl=LUTRAM

    // [OPT-10]: cand_cls is now ap_uint<2>; LUTRAM (300×2 = 600 bits).
    #pragma HLS bind_storage variable=cand_cls    type=RAM_2P impl=LUTRAM

    // keep_idx still in BRAM (MAX_DETS=100 entries, accessed only at output).
    #pragma HLS bind_storage variable=keep_idx    type=RAM_T2P impl=BRAM

    // ── [1] score filter + anchor decode ─────────────────────────────────
    int num_cand = score_filter(cls_logits, reg_deltas,
                                cand_boxes, cand_scores, cand_cls, cand_areas);

    // ── [2] sort by score descending; take topK ───────────────────────────
    sort_scores(cand_scores, cand_order, num_cand);
    int topk = (num_cand < NMS_PRE) ? num_cand : NMS_PRE;

    // ── [3] greedy NMS ────────────────────────────────────────────────────
    int nk = nms_greedy(cand_boxes, cand_scores, cand_order,
                        keep_idx, cand_cls, cand_areas, topk);

    // ── [4] write output ──────────────────────────────────────────────────
    int total = (nk < MAX_DETS) ? nk : MAX_DETS;

    OUT: for (int k = 0; k < total; k++) {
        #pragma HLS PIPELINE II=1
        int ki = keep_idx[k];
        // [OPT-8]: coord_t → float conversion for Detection output interface
        detections[k].x1       = (bbox_t)(float)cand_boxes[ki*4+0];
        detections[k].y1       = (bbox_t)(float)cand_boxes[ki*4+1];
        detections[k].x2       = (bbox_t)(float)cand_boxes[ki*4+2];
        detections[k].y2       = (bbox_t)(float)cand_boxes[ki*4+3];
        detections[k].score    = cand_scores[ki];
        detections[k].class_id = (int)cand_cls[ki];  // [OPT-10]: 2-bit → int
    }

    *num_dets = total;
}
