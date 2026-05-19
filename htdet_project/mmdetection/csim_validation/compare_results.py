"""
compare_results.py
------------------
Compare Python detections (python_detections.txt)
with C simulation detections (csim_detections.txt).

Both files use the same format written by detect_python.py / testbench.cpp:
  <num_dets>
  <class_id> <score> <x1> <y1> <x2> <y2>
  ...

Run after C simulation has produced csim_detections.txt:
  python compare_results.py [--py python_detections.txt] [--csim csim_detections.txt]
"""

import argparse
import os

CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']


def load_detections(path):
    dets = []
    if not os.path.exists(path):
        print(f"[warn] File not found: {path}")
        return dets
    with open(path) as f:
        n = int(f.readline().strip())
        for _ in range(n):
            parts = f.readline().split()
            cls_id = int(parts[0])
            score  = float(parts[1])
            x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            dets.append((cls_id, score, x1, y1, x2, y2))
    return dets


def box_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw  = max(0, ix2 - ix1);  ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def print_dets(label, dets):
    print(f"\n{'─'*60}")
    print(f"  {label}  ({len(dets)} detection(s))")
    print(f"{'─'*60}")
    if not dets:
        print("  (none)")
        return
    print(f"  {'#':<4}  {'Class':<14}  {'Score':>7}  {'x1':>7}  {'y1':>7}  {'x2':>7}  {'y2':>7}")
    for i, (cls_id, score, x1, y1, x2, y2) in enumerate(dets):
        print(f"  {i:<4}  {CLASS_NAMES[cls_id]:<14}  {score:7.4f}  "
              f"{x1:7.1f}  {y1:7.1f}  {x2:7.1f}  {y2:7.1f}")


def compare(py_dets, csim_dets, iou_thr=0.5, score_tol=0.1):
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")

    if not py_dets and not csim_dets:
        print("  Both outputs have zero detections.")
        return

    matched_py   = [False] * len(py_dets)
    matched_csim = [False] * len(csim_dets)

    matches = []
    for i, pd in enumerate(py_dets):
        best_iou = 0;  best_j = -1
        for j, cd in enumerate(csim_dets):
            if matched_csim[j]:
                continue
            if pd[0] != cd[0]:    # class must match
                continue
            iou = box_iou(pd[2:], cd[2:])
            if iou > best_iou:
                best_iou = iou;  best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_py[i]       = True
            matched_csim[best_j] = True
            matches.append((i, best_j, best_iou))

    print(f"\n  Python detections : {len(py_dets)}")
    print(f"  C-sim  detections : {len(csim_dets)}")
    print(f"  Matched pairs     : {len(matches)}  (IoU >= {iou_thr})")

    if matches:
        print(f"\n  {'Py#':<5} {'CSim#':<6} {'Class':<14} "
              f"{'IoU':>6}  {'Py score':>9}  {'CSim score':>10}  {'ΔScore':>8}  "
              f"{'Py box (x1,y1,x2,y2)':<28}  {'CSim box'}")
        print(f"  {'─'*5} {'─'*6} {'─'*14} {'─'*6}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*28}  {'─'*28}")
        for i, j, iou in matches:
            pd = py_dets[i];  cd = csim_dets[j]
            ds = abs(pd[1] - cd[1])
            py_box   = f"[{pd[2]:.1f},{pd[3]:.1f},{pd[4]:.1f},{pd[5]:.1f}]"
            csim_box = f"[{cd[2]:.1f},{cd[3]:.1f},{cd[4]:.1f},{cd[5]:.1f}]"
            flag = "  ← large score diff" if ds > score_tol else ""
            print(f"  {i:<5} {j:<6} {CLASS_NAMES[pd[0]]:<14} "
                  f"{iou:6.3f}  {pd[1]:9.4f}  {cd[1]:10.4f}  {ds:8.4f}  "
                  f"{py_box:<28}  {csim_box}{flag}")

    unmatched_py = [i for i, m in enumerate(matched_py) if not m]
    if unmatched_py:
        print(f"\n  Python-only (no C-sim match):")
        for i in unmatched_py:
            pd = py_dets[i]
            print(f"    #{i}  {CLASS_NAMES[pd[0]]}  score={pd[1]:.4f}  "
                  f"box=[{pd[2]:.1f},{pd[3]:.1f},{pd[4]:.1f},{pd[5]:.1f}]")

    unmatched_csim = [j for j, m in enumerate(matched_csim) if not m]
    if unmatched_csim:
        print(f"\n  C-sim-only (no Python match):")
        for j in unmatched_csim:
            cd = csim_dets[j]
            print(f"    #{j}  {CLASS_NAMES[cd[0]]}  score={cd[1]:.4f}  "
                  f"box=[{cd[2]:.1f},{cd[3]:.1f},{cd[4]:.1f},{cd[5]:.1f}]")

    # Overall verdict
    print()
    if len(matches) == len(py_dets) == len(csim_dets):
        print("  RESULT: ALL detections matched. C simulation is consistent with Python.")
    elif len(matches) > 0:
        pct = 100 * len(matches) / max(len(py_dets), len(csim_dets))
        print(f"  RESULT: Partial match ({pct:.0f}%). "
              "Typical cause: fixed-point quantisation shifts borderline detections.")
    else:
        print("  RESULT: No matched detections. "
              "Check that both runs used the same image and INPUT_H/INPUT_W=640.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--py',   default=os.path.join(script_dir, 'python_detections.txt'))
    parser.add_argument('--csim', default=os.path.join(script_dir, 'csim_detections.txt'))
    parser.add_argument('--iou',  type=float, default=0.5)
    args = parser.parse_args()

    py_dets   = load_detections(args.py)
    csim_dets = load_detections(args.csim)

    print_dets("PYTHON detections", py_dets)
    print_dets("C-SIM  detections", csim_dets)
    compare(py_dets, csim_dets, iou_thr=args.iou)
