#!/usr/bin/env python3
"""
gen_nms_ref.py
Python-side reference for nms_top CSIM validation.

Runs the same four synthetic test cases as testbench_nms.cpp and prints
the expected detections for each test.  Use this to cross-check CSIM output.

Usage:
    cd /path/to/mmdetection
    python3 fpga_nms/gen_nms_ref.py
"""

import numpy as np

# ── model config (must match fpga_types.h) ───────────────────────────────────
TEST_H       = 10
TEST_W       = 10
TEST_STRIDE  = 64
NUM_CLASSES  = 4
ANCHOR_SCALES = np.array([4.0, 6.0, 8.0], dtype=np.float32)
ANCHOR_RATIOS = np.array([0.5, 1.0, 2.0], dtype=np.float32)
ANCHORS_PER_LOC = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)   # 9
CLS_OUT_CH   = ANCHORS_PER_LOC * NUM_CLASSES    # 36
REG_OUT_CH   = ANCHORS_PER_LOC * 4              # 36
SCORE_THR    = 0.20
NMS_IOU_THR  = 0.5
NMS_PRE      = 1000
MAX_DETS     = 100

CLS_LOGITS_ELEMS = CLS_OUT_CH * TEST_H * TEST_W
REG_DELTAS_ELEMS = REG_OUT_CH * TEST_H * TEST_W


# ── helpers ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def set_logit(logits, a, c, h, w, v):
    cls_ch = a * NUM_CLASSES + c
    logits[cls_ch, h, w] = v


def zero_delta(deltas, a, h, w):
    for coord in range(4):
        deltas[a*4 + coord, h, w] = 0.0


def score_filter(cls_logits_chw, reg_deltas_chw):
    """Score filter + anchor decode → candidate list."""
    boxes, scores, cls_ids = [], [], []
    H, W = TEST_H, TEST_W
    stride = TEST_STRIDE

    for h in range(H):
        for w in range(W):
            cx = (w + 0.5) * stride
            cy = (h + 0.5) * stride
            for ri, ratio in enumerate(ANCHOR_RATIOS):
                for si, scale in enumerate(ANCHOR_SCALES):
                    a   = ri * len(ANCHOR_SCALES) + si
                    aw  = scale * stride / np.sqrt(ratio)
                    ah  = scale * stride * np.sqrt(ratio)
                    for c in range(NUM_CLASSES):
                        cls_ch = a * NUM_CLASSES + c
                        s = float(sigmoid(cls_logits_chw[cls_ch, h, w]))
                        if s >= SCORE_THR:
                            dx = float(reg_deltas_chw[a*4+0, h, w])
                            dy = float(reg_deltas_chw[a*4+1, h, w])
                            dw = float(reg_deltas_chw[a*4+2, h, w])
                            dh = float(reg_deltas_chw[a*4+3, h, w])
                            dw = max(-4.135, min(4.135, dw))
                            dh = max(-4.135, min(4.135, dh))
                            pred_cx = dx * aw + cx
                            pred_cy = dy * ah + cy
                            pred_w  = np.exp(dw) * aw
                            pred_h  = np.exp(dh) * ah
                            x1 = max(pred_cx - pred_w * 0.5, 0.0)
                            y1 = max(pred_cy - pred_h * 0.5, 0.0)
                            x2 = min(pred_cx + pred_w * 0.5, float(TEST_W * TEST_STRIDE))
                            y2 = min(pred_cy + pred_h * 0.5, float(TEST_H * TEST_STRIDE))
                            boxes.append([x1, y1, x2, y2])
                            scores.append(s)
                            cls_ids.append(c)

    return (np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32),
            np.array(scores, dtype=np.float32),
            np.array(cls_ids, dtype=np.int32))


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = x2 - x1; ih = y2 - y1
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih
    return inter / (float((a[2]-a[0])*(a[3]-a[1])) + float((b[2]-b[0])*(b[3]-b[1])) - inter)


def nms_greedy(boxes, scores, cls_ids, iou_thr):
    """Greedy multiclass NMS — matching C++ implementation."""
    order = np.argsort(-scores)[:NMS_PRE]
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for i in range(len(order)):
        if suppressed[i]: continue
        keep.append(order[i])
        if len(keep) >= MAX_DETS: break
        for j in range(i+1, len(order)):
            if suppressed[j]: continue
            if cls_ids[order[i]] != cls_ids[order[j]]: continue
            if iou(boxes[order[i]], boxes[order[j]]) > iou_thr:
                suppressed[j] = True
    return keep


def run_nms(cls_logits_flat, reg_deltas_flat):
    logits_chw = cls_logits_flat.reshape(CLS_OUT_CH, TEST_H, TEST_W)
    deltas_chw = reg_deltas_flat.reshape(REG_OUT_CH, TEST_H, TEST_W)
    boxes, scores, cls_ids = score_filter(logits_chw, deltas_chw)
    if len(scores) == 0:
        return [], [], [], []
    keep = nms_greedy(boxes, scores, cls_ids, NMS_IOU_THR)
    return ([boxes[k] for k in keep],
            [scores[k] for k in keep],
            [cls_ids[k] for k in keep],
            keep)


CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]

def print_dets(boxes, scores, cls_ids):
    if not boxes:
        print("  (no detections)")
        return
    for i, (b, s, c) in enumerate(zip(boxes, scores, cls_ids)):
        print(f"  det[{i}]  {CLASS_NAMES[c]}  score={s:.5f}"
              f"  box=[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]")


# ── test cases (must match testbench_nms.cpp) ─────────────────────────────────
def main():
    print("=================================================")
    print("gen_nms_ref.py  — Python reference for nms_top CSIM")
    print(f"  H={TEST_H}  W={TEST_W}  stride={TEST_STRIDE}"
          f"  SCORE_THR={SCORE_THR}  NMS_IOU_THR={NMS_IOU_THR}")
    print("=================================================\n")

    # ── Test 1: all logits = -10 ──────────────────────────────────────────
    print("[Test 1] All logits = -10.0")
    logits = np.full(CLS_LOGITS_ELEMS, -10.0, dtype=np.float32)
    deltas = np.zeros(REG_DELTAS_ELEMS, dtype=np.float32)
    boxes, scores, cls_ids, _ = run_nms(logits, deltas)
    print(f"  num_dets = {len(boxes)}  (expected 0)")
    print_dets(boxes, scores, cls_ids)
    print()

    # ── Test 2: single detection ──────────────────────────────────────────
    print("[Test 2] Single detection — anchor=4, (h=5,w=5), logit=2.0, class=0")
    logits = np.full(CLS_LOGITS_ELEMS, -10.0, dtype=np.float32)
    deltas = np.zeros(REG_DELTAS_ELEMS, dtype=np.float32)
    logits_chw = logits.reshape(CLS_OUT_CH, TEST_H, TEST_W)
    deltas_chw = deltas.reshape(REG_OUT_CH, TEST_H, TEST_W)
    set_logit(logits_chw, 4, 0, 5, 5, 2.0)
    zero_delta(deltas_chw, 4, 5, 5)
    boxes, scores, cls_ids, _ = run_nms(logits_chw.ravel(), deltas_chw.ravel())
    print(f"  num_dets = {len(boxes)}  (expected 1)")
    print_dets(boxes, scores, cls_ids)
    # analytic check
    exp_score = float(sigmoid(2.0))
    print(f"  expected: score={exp_score:.5f}  box=[160.0,160.0,544.0,544.0]")
    print()

    # ── Test 3: NMS suppression ────────────────────────────────────────────
    print("[Test 3] NMS suppression — anchor=4 (score≈0.88) vs anchor=5 (score≈0.82), class=0")
    logits = np.full(CLS_LOGITS_ELEMS, -10.0, dtype=np.float32)
    deltas = np.zeros(REG_DELTAS_ELEMS, dtype=np.float32)
    logits_chw = logits.reshape(CLS_OUT_CH, TEST_H, TEST_W)
    deltas_chw = deltas.reshape(REG_OUT_CH, TEST_H, TEST_W)
    set_logit(logits_chw, 4, 0, 5, 5, 2.0)
    zero_delta(deltas_chw, 4, 5, 5)
    set_logit(logits_chw, 5, 0, 5, 5, 1.5)
    zero_delta(deltas_chw, 5, 5, 5)
    boxes, scores, cls_ids, keep = run_nms(logits_chw.ravel(), deltas_chw.ravel())
    print(f"  num_dets = {len(boxes)}  (expected 1 — anchor=5 suppressed)")
    print_dets(boxes, scores, cls_ids)
    # verify IoU
    # anchor 4: box=[160,160,544,544]; anchor 5: box=[96,96,608,608]
    a4_box = np.array([160.0, 160.0, 544.0, 544.0])
    a5_box = np.array([ 96.0,  96.0, 608.0, 608.0])
    print(f"  analytic IoU(anchor4, anchor5) = {iou(a4_box, a5_box):.4f}"
          f"  (> {NMS_IOU_THR} → suppressed)")
    print()

    # ── Test 4: multi-class, no suppression ───────────────────────────────
    print("[Test 4] Multi-class — anchor=4 class=0 vs anchor=5 class=1 (no suppress)")
    logits = np.full(CLS_LOGITS_ELEMS, -10.0, dtype=np.float32)
    deltas = np.zeros(REG_DELTAS_ELEMS, dtype=np.float32)
    logits_chw = logits.reshape(CLS_OUT_CH, TEST_H, TEST_W)
    deltas_chw = deltas.reshape(REG_OUT_CH, TEST_H, TEST_W)
    set_logit(logits_chw, 4, 0, 5, 5, 2.0)   # class 0
    zero_delta(deltas_chw, 4, 5, 5)
    set_logit(logits_chw, 5, 1, 5, 5, 1.5)   # class 1
    zero_delta(deltas_chw, 5, 5, 5)
    boxes, scores, cls_ids, _ = run_nms(logits_chw.ravel(), deltas_chw.ravel())
    print(f"  num_dets = {len(boxes)}  (expected 2 — different classes, no suppress)")
    print_dets(boxes, scores, cls_ids)
    print()

    print("=================================================")
    print("Compare above with testbench_nms.cpp CSIM output.")
    print("=================================================")


if __name__ == "__main__":
    main()
