/*
 * testbench_nms.cpp
 * C-simulation testbench for nms_top — isolated score-filter + NMS module.
 *
 * ── Three tests ────────────────────────────────────────────────────────────
 *
 * Test 1 — Zero score:
 *   All cls_logits = -10.0 → sigmoid(-10) ≈ 0  << SCORE_THR=0.20.
 *   Expected: num_dets = 0.
 *
 * Test 2 — Single detection:
 *   One logit set high (2.0 → sigmoid ≈ 0.88) at a known anchor position.
 *   All reg_deltas = 0 (no displacement from anchor centre).
 *   Expected: num_dets = 1, score ≈ 0.88, box matches analytic anchor.
 *
 *   Anchor used: a=4  (ri=1→ratio=1.0, si=1→scale=6)  at (h=5, w=5)
 *   With stride=64:
 *     cx = (5+0.5)*64 = 352,  cy = 352
 *     aw = 6*64 / sqrt(1.0) = 384,  ah = 384
 *     pred box = [160, 160, 544, 544]
 *
 * Test 3 — NMS suppression:
 *   Two overlapping boxes, same class.  Higher-score box survives; lower is
 *   suppressed when IoU > NMS_IOU_THR = 0.5.
 *
 *   Box A: anchor=4 (ratio=1.0, scale=6) → box=[160,160,544,544], score≈0.88
 *   Box B: anchor=5 (ratio=1.0, scale=8) → box=[ 96, 96,608,608], score≈0.82
 *   Both at (h=5, w=5), class=0, delta=0.
 *
 *   IoU(A,B):
 *     intersection: [160,160,544,544] (A fully inside B)
 *       area = 384*384 = 147 456
 *     area_A = 384² = 147 456,  area_B = 512² = 262 144
 *     union  = 147 456 + 262 144 - 147 456 = 262 144
 *     IoU    = 147 456 / 262 144 ≈ 0.5625  >  0.5 ✓ → B suppressed
 *   Expected: num_dets = 1  (only A survives)
 *
 * ── Standalone compile ─────────────────────────────────────────────────────
 *   g++ -std=c++14 -I. -Ihls_stubs -DTEST_H=10 -DTEST_W=10 -DTEST_STRIDE=64 \
 *       testbench_nms.cpp nms_top.cpp -lm -o tb_nms
 *   ./tb_nms
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include "nms_top.h"

// ── helpers ────────────────────────────────────────────────────────────────
static void all_below_thr(act_t* logits, int n) {
    for (int i = 0; i < n; i++) logits[i] = -10.0f;
}

// Set logit for (anchor a, class c, position h,w) to value v
static void set_logit(act_t* logits, int a, int c, int h, int w, float v) {
    int cls_ch = a * NUM_CLASSES + c;
    logits[cls_ch * TEST_H * TEST_W + h * TEST_W + w] = (act_t)v;
}

// Set all four delta channels for (anchor a, position h,w) to 0
static void zero_delta(act_t* deltas, int a, int h, int w) {
    int hw = h * TEST_W + w;
    for (int coord = 0; coord < 4; coord++)
        deltas[(a*4 + coord) * TEST_H * TEST_W + hw] = 0.0f;
}

static const char* cls_names[] = {"holothurian", "echinus", "scallop", "starfish"};

static void print_det(int idx, const Detection& d) {
    std::cout << "  det[" << idx << "]  "
              << cls_names[d.class_id]
              << "  score=" << d.score
              << "  box=[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]\n";
}

// ── main ───────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=================================================\n";
    std::cout << "nms_top  C-simulation testbench\n";
    std::cout << "  TEST_H=" << TEST_H << "  TEST_W=" << TEST_W
              << "  TEST_STRIDE=" << TEST_STRIDE << "\n";
    std::cout << "  CAND_BUF_SIZE=" << CAND_BUF_SIZE
              << "  NMS_PRE=" << NMS_PRE
              << "  SCORE_THR=" << SCORE_THR << "\n";
    std::cout << "=================================================\n\n";

    static act_t     cls_logits[CLS_LOGITS_ELEMS];
    static act_t     reg_deltas[REG_DELTAS_ELEMS];
    static Detection detections[MAX_DETS];
    int num_dets = 0;

    bool all_pass = true;

    // ══════════════════════════════════════════════════════════════════════
    // TEST 1: All logits far below threshold → 0 detections
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 1] All logits = -10.0 (sigmoid ≈ 0 << " << SCORE_THR << ")\n";
    all_below_thr(cls_logits, CLS_LOGITS_ELEMS);
    memset(reg_deltas, 0, sizeof(reg_deltas));
    memset(detections, 0, sizeof(detections));
    num_dets = 0;

    nms_top(cls_logits, reg_deltas, detections, &num_dets);

    std::cout << "  num_dets = " << num_dets << "  (expected 0)\n";
    bool t1 = (num_dets == 0);
    std::cout << "  " << (t1 ? "[PASS]" : "[FAIL]") << "\n\n";
    all_pass &= t1;

    // ══════════════════════════════════════════════════════════════════════
    // TEST 2: Single detection — known anchor + zero delta
    // anchor=4 (ratio=1.0, scale=6) at (h=5,w=5), stride=64, class=0
    // Expected box: [160, 160, 544, 544],  score = sigmoid(2.0) ≈ 0.8808
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 2] Single detection — anchor=4, (h=5,w=5), logit=2.0\n";
    all_below_thr(cls_logits, CLS_LOGITS_ELEMS);
    memset(reg_deltas, 0, sizeof(reg_deltas));
    memset(detections, 0, sizeof(detections));
    num_dets = 0;

    set_logit(cls_logits, /*a=*/4, /*c=*/0, /*h=*/5, /*w=*/5, 2.0f);
    zero_delta(reg_deltas, /*a=*/4, /*h=*/5, /*w=*/5);

    nms_top(cls_logits, reg_deltas, detections, &num_dets);

    std::cout << "  num_dets = " << num_dets << "  (expected 1)\n";
    if (num_dets > 0) print_det(0, detections[0]);

    // Analytic expected values for anchor=4, delta=0, stride=64
    // ratio=1.0, scale=6 → aw=ah=384; cx=cy=352
    // box = [352-192, 352-192, 352+192, 352+192] = [160, 160, 544, 544]
    const float expected_score = 1.0f / (1.0f + expf(-2.0f));   // exact sigmoid(2.0)
    const float expected_box[4] = {160.0f, 160.0f, 544.0f, 544.0f};
    bool t2 = (num_dets == 1);
    if (num_dets >= 1) {
        t2 &= (fabsf(detections[0].score - expected_score) < 1e-3f);
        t2 &= (fabsf(detections[0].x1 - expected_box[0]) < 1.0f);
        t2 &= (fabsf(detections[0].y1 - expected_box[1]) < 1.0f);
        t2 &= (fabsf(detections[0].x2 - expected_box[2]) < 1.0f);
        t2 &= (fabsf(detections[0].y2 - expected_box[3]) < 1.0f);
        t2 &= (detections[0].class_id == 0);
        std::cout << "  expected  score=" << expected_score
                  << "  box=[" << expected_box[0] << "," << expected_box[1]
                  << "," << expected_box[2] << "," << expected_box[3] << "]\n";
    }
    std::cout << "  " << (t2 ? "[PASS]" : "[FAIL]") << "\n\n";
    all_pass &= t2;

    // ══════════════════════════════════════════════════════════════════════
    // TEST 3: NMS suppression — two overlapping boxes, same class
    //   Box A: anchor=4 (ratio=1.0, scale=6) at (h=5,w=5), logit=2.0, score≈0.88
    //     box = [160, 160, 544, 544]
    //   Box B: anchor=5 (ratio=1.0, scale=8) at (h=5,w=5), logit=1.5, score≈0.82
    //     box = [96, 96, 608, 608]  (clamped: max(96,0)=96, min(608,640)=608)
    //   IoU(A,B) = 147456/262144 ≈ 0.5625 > NMS_IOU_THR=0.5 → B suppressed
    //   Expected: num_dets=1, only A survives
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 3] NMS suppression — two overlapping boxes, same class\n";
    all_below_thr(cls_logits, CLS_LOGITS_ELEMS);
    memset(reg_deltas, 0, sizeof(reg_deltas));
    memset(detections, 0, sizeof(detections));
    num_dets = 0;

    // Box A: anchor=4, class=0, higher score
    set_logit(cls_logits, 4, 0, 5, 5, 2.0f);
    zero_delta(reg_deltas, 4, 5, 5);

    // Box B: anchor=5, class=0, lower score → should be suppressed
    set_logit(cls_logits, 5, 0, 5, 5, 1.5f);
    zero_delta(reg_deltas, 5, 5, 5);

    nms_top(cls_logits, reg_deltas, detections, &num_dets);

    std::cout << "  num_dets = " << num_dets << "  (expected 1 — Box B suppressed)\n";
    for (int k = 0; k < num_dets; k++) print_det(k, detections[k]);

    float iou_ab = 0.5625f;   // precomputed (see header comment)
    std::cout << "  analytic IoU(A,B) = " << iou_ab
              << "  (> NMS_IOU_THR=" << NMS_IOU_THR << " → B suppressed)\n";
    bool t3 = (num_dets == 1);
    if (num_dets >= 1)
        t3 &= (detections[0].class_id == 0);
    std::cout << "  " << (t3 ? "[PASS]" : "[FAIL]") << "\n\n";
    all_pass &= t3;

    // ══════════════════════════════════════════════════════════════════════
    // TEST 4: Multi-class — two boxes, different classes → NMS does NOT suppress
    //   Box A: anchor=4, class=0, logit=2.0
    //   Box B: anchor=5, class=1, logit=1.5
    //   Same spatial IoU ≈ 0.5625, but different class → both survive NMS
    //   Expected: num_dets=2
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "[Test 4] Multi-class — same spatial overlap, different classes (no suppress)\n";
    all_below_thr(cls_logits, CLS_LOGITS_ELEMS);
    memset(reg_deltas, 0, sizeof(reg_deltas));
    memset(detections, 0, sizeof(detections));
    num_dets = 0;

    set_logit(cls_logits, 4, 0, 5, 5, 2.0f);   // class 0
    zero_delta(reg_deltas, 4, 5, 5);

    set_logit(cls_logits, 5, 1, 5, 5, 1.5f);   // class 1 (different class)
    zero_delta(reg_deltas, 5, 5, 5);

    nms_top(cls_logits, reg_deltas, detections, &num_dets);

    std::cout << "  num_dets = " << num_dets << "  (expected 2 — different classes)\n";
    for (int k = 0; k < num_dets; k++) print_det(k, detections[k]);
    bool t4 = (num_dets == 2);
    std::cout << "  " << (t4 ? "[PASS]" : "[FAIL]") << "\n\n";
    all_pass &= t4;

    // ══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=================================================\n";
    std::cout << "nms_top  CSIM " << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    std::cout << "=================================================\n";
    return all_pass ? 0 : 1;
}
