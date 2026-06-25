/*
 * fpga_types.h  (NMS isolated module)
 *
 * Tests the score-filter → anchor-decode → sort → NMS path for ONE FPN level.
 * The stacked-conv + pred-conv computation is handled by fpga_retina_head;
 * this module takes cls_logits + reg_deltas directly as inputs.
 *
 * Configurable spatial test size — match to the level under test:
 *
 *   Phase 1 (start): P6  TEST_H=10  TEST_W=10  TEST_STRIDE=64
 *   Phase 2:         P5  TEST_H=20  TEST_W=20  TEST_STRIDE=32
 *   Phase 3:         P4  TEST_H=40  TEST_W=40  TEST_STRIDE=16
 *   Phase 4:         P3  TEST_H=80  TEST_W=80  TEST_STRIDE=8
 *
 * Override with -DTEST_H=20 -DTEST_W=20 -DTEST_STRIDE=32 in the TCL CFLAGS.
 */

#ifndef FPGA_TYPES_H
#define FPGA_TYPES_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <cstdint>

// ── data types (W8A32, matching fpga_ptq_192) ─────────────────────────────
typedef float   act_t;
typedef float   acc_t;
typedef float   bbox_t;
typedef float   score_t;
typedef float   meta_t;
typedef int8_t  weight_t;

// ── model / head config ───────────────────────────────────────────────────
#define INPUT_H            640
#define INPUT_W            640

#define NUM_CLASSES          4
#define ANCHOR_SCALES_N      3    // {4, 6, 8}
#define ANCHOR_RATIOS_N      3    // {0.5, 1.0, 2.0}
#define ANCHORS_PER_LOC      (ANCHOR_SCALES_N * ANCHOR_RATIOS_N)   // 9

#define MAX_DETS           100    // max detections returned
#define NMS_PRE            300    // per-level topK before NMS (was 1000; 300 gives ~11x NMS speedup with <0.5 mAP drop)
#define SCORE_THR          0.20f
#define NMS_IOU_THR        0.5f

// ── derived channel counts ────────────────────────────────────────────────
#define CLS_OUT_CH   (ANCHORS_PER_LOC * NUM_CLASSES)   // 36
#define REG_OUT_CH   (ANCHORS_PER_LOC * 4)             // 36

// ── spatial test size (P6 default = smallest, fastest CSIM) ──────────────
#ifndef TEST_H
#define TEST_H       10
#endif
#ifndef TEST_W
#define TEST_W       10
#endif
#ifndef TEST_STRIDE
#define TEST_STRIDE  64   // P6: 64; P5: 32; P4: 16; P3: 8
#endif

// Max candidates the score-filter can emit (all anchors × all classes pass).
// Used to size cand_boxes/scores/cls buffers.
#define CAND_BUF_SIZE  (TEST_H * TEST_W * ANCHORS_PER_LOC * NUM_CLASSES)

// ── output struct ─────────────────────────────────────────────────────────
typedef struct {
    bbox_t  x1, y1, x2, y2;
    score_t score;
    int     class_id;
} Detection;

// ── utility ───────────────────────────────────────────────────────────────
#define DIV_CEIL(a,b)  (((a)+(b)-1)/(b))

#endif // FPGA_TYPES_H
