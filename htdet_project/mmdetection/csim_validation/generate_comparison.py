#!/usr/bin/env python3
"""
Generate per-image 4-mode comparison tables and a cross-image summary.
Modes: Python (FP32) | FP32 C-sim | W8A32 C-sim | W8A8 C-sim
"""
import os, json, math

BASE = os.path.dirname(os.path.abspath(__file__))
CLASSES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}

IMAGES = ["YDXJ0001_10003", "GOPR0293_10229", "CHN083846_0291"]


def load_dets(path):
    """Load detections from file: first line = count, then class score x1 y1 x2 y2."""
    if not os.path.exists(path):
        return None
    dets = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 6:
            dets.append({
                "class": int(parts[0]),
                "score": float(parts[1]),
                "box": [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            })
    return dets


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


def match_to_anchor(anchor_dets, query_dets, iou_thr=0.5):
    """For each anchor det find best-IoU match in query_dets. Returns list of (iou, det|None)."""
    used = set()
    results = []
    for a in anchor_dets:
        best_iou, best_idx = 0.0, -1
        for i, q in enumerate(query_dets):
            if i in used:
                continue
            v = iou(a["box"], q["box"])
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= iou_thr:
            used.add(best_idx)
            results.append((best_iou, query_dets[best_idx]))
        else:
            results.append((best_iou, None))
    # unmatched query dets
    unmatched = [query_dets[i] for i in range(len(query_dets)) if i not in used]
    return results, unmatched


def fmt_box(box):
    return "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*box)


def write_per_image_table(image_name, py_dets, fp32_dets, w8a32_dets, w8a8_dets, out_path):
    fp32_matches,  fp32_unmatched  = match_to_anchor(py_dets, fp32_dets  or [])
    w8a32_matches, w8a32_unmatched = match_to_anchor(py_dets, w8a32_dets or [])
    w8a8_matches,  w8a8_unmatched  = match_to_anchor(py_dets, w8a8_dets  or [])

    fp32_ok  = sum(1 for _,d in fp32_matches  if d)
    w8a32_ok = sum(1 for _,d in w8a32_matches if d)
    w8a8_ok  = sum(1 for _,d in w8a8_matches  if d)

    n = len(py_dets)
    fp32_str  = fp32_dets  and str(len(fp32_dets))  or "N/A"
    w8a32_str = w8a32_dets and str(len(w8a32_dets)) or "N/A"
    w8a8_str  = w8a8_dets  and str(len(w8a8_dets))  or "N/A"

    W = 200
    sep = "=" * W
    dash = "-" * W

    lines = []
    lines.append(sep)
    lines.append("  Image    : {}.jpg".format(image_name))
    lines.append("  Python   : {} detections (FP32 reference, score_thr=0.20)".format(n))
    lines.append("  FP32csim : {} detections  matched={} ({:.1f}%)".format(
        fp32_str, fp32_ok, 100*fp32_ok/n if n else 0))
    lines.append("  W8A32csim: {} detections  matched={} ({:.1f}%)".format(
        w8a32_str, w8a32_ok, 100*w8a32_ok/n if n else 0))
    lines.append("  W8A8csim : {} detections  matched={} ({:.1f}%)".format(
        w8a8_str, w8a8_ok, 100*w8a8_ok/n if n else 0))
    lines.append(sep)

    hdr = ("  {:>3}  {:<14}  {:>5}  "
           "{:>8}  {:>8} {:>7}  "
           "{:>8}  {:>8} {:>7}  "
           "{:>8}  {:>8} {:>7}  "
           "Py box                   FP32 box                 W8A32 box                W8A8 box").format(
        "#", "Class", "PyScr",
        "FP32Scr", "FP32IoU", "FP32Δ",
        "W32Scr", "W32IoU", "W32Δ",
        "W8Scr", "W8IoU", "W8Δ"
    )
    lines.append(hdr)
    lines.append(dash)

    for i, py in enumerate(py_dets):
        cls = CLASSES.get(py["class"], str(py["class"]))
        py_s = py["score"]

        fp32_iou_v, fp32_d  = fp32_matches[i]
        w32_iou_v,  w32_d   = w8a32_matches[i]
        w8_iou_v,   w8_d    = w8a8_matches[i]

        fp32_s  = fp32_d["score"]  if fp32_d  else float("nan")
        w32_s   = w32_d["score"]   if w32_d   else float("nan")
        w8_s    = w8_d["score"]    if w8_d    else float("nan")

        fp32_box  = fmt_box(fp32_d["box"])  if fp32_d  else "N/A"
        w32_box   = fmt_box(w32_d["box"])   if w32_d   else "N/A"
        w8_box    = fmt_box(w8_d["box"])    if w8_d    else "N/A"

        def ds(a, b):
            if math.isnan(a) or math.isnan(b):
                return "  N/A"
            return "{:+.4f}".format(b - a)

        fp32_flag  = " <- large Δ" if abs(fp32_s - py_s) > 0.1  and fp32_d  else ""
        w32_flag   = " <- large Δ" if abs(w32_s  - py_s) > 0.1  and w32_d   else ""
        w8_flag    = " <- large Δ" if abs(w8_s   - py_s) > 0.15 and w8_d    else ""

        row = ("  {:>3}  {:<14}  {:>5.4f}  "
               "{:>8.4f}  {:>7.3f} {}  "
               "{:>8.4f}  {:>7.3f} {}  "
               "{:>8.4f}  {:>7.3f} {}  "
               "{:<25}{:<25}{:<25}{}").format(
            i, cls, py_s,
            fp32_s  if not math.isnan(fp32_s)  else -1, fp32_iou_v,  ds(py_s, fp32_s),
            w32_s   if not math.isnan(w32_s)   else -1, w32_iou_v,   ds(py_s, w32_s),
            w8_s    if not math.isnan(w8_s)    else -1, w8_iou_v,    ds(py_s, w8_s),
            fmt_box(py["box"]), fp32_box, w32_box, w8_box
        )
        lines.append(row + fp32_flag + w32_flag + w8_flag)

    # unmatched
    all_unmatched = []
    for mode, ul in [("FP32", fp32_unmatched), ("W8A32", w8a32_unmatched), ("W8A8", w8a8_unmatched)]:
        for d in ul:
            all_unmatched.append((mode, d))
    if all_unmatched:
        lines.append(dash)
        lines.append("  UNMATCHED detections (no Python counterpart, IoU < 0.5):")
        for mode, d in all_unmatched:
            lines.append("    [{}] {} score={:.4f} box={}".format(
                mode, CLASSES.get(d["class"], str(d["class"])), d["score"], fmt_box(d["box"])))

    lines.append(sep)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Written:", out_path)


def write_summary(results, out_path):
    W = 100
    sep = "=" * W
    dash = "-" * W
    lines = []
    lines.append(sep)
    lines.append("  HTDet C-Simulation: 4-Mode Comparison Summary")
    lines.append("  Python (FP32 ref) | FP32 C-sim | W8A32 C-sim (INT8 wts) | W8A8 C-sim (INT8 wts+acts)")
    lines.append(sep)
    lines.append("  {:<22}  {:>7}  {:>12}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}".format(
        "Image", "PySrc", "FP32dets", "FP32match%", "W8A32dets", "W8A32match%", "W8A8dets", "W8A8match%"))
    lines.append(dash)
    for r in results:
        lines.append("  {:<22}  {:>7}  {:>12}  {:>13}  {:>13}  {:>13}  {:>13}  {:>13}".format(
            r["image"],
            r["py_n"],
            r.get("fp32_n", "N/A"),
            "{:.1f}%".format(r["fp32_pct"]) if "fp32_pct" in r else "N/A",
            r.get("w8a32_n", "N/A"),
            "{:.1f}%".format(r["w8a32_pct"]) if "w8a32_pct" in r else "N/A",
            r.get("w8a8_n", "N/A"),
            "{:.1f}%".format(r["w8a8_pct"]) if "w8a8_pct" in r else "N/A",
        ))
    lines.append(sep)
    lines.append("")
    lines.append("  Score degradation (avg |ΔScore| for matched pairs):")
    lines.append("  {:<22}  {:>12}  {:>13}  {:>13}".format(
        "Image", "FP32 avg|Δ|", "W8A32 avg|Δ|", "W8A8 avg|Δ|"))
    lines.append(dash)
    for r in results:
        lines.append("  {:<22}  {:>12}  {:>13}  {:>13}".format(
            r["image"],
            "{:.4f}".format(r["fp32_avg_delta"])  if "fp32_avg_delta"  in r else "N/A",
            "{:.4f}".format(r["w8a32_avg_delta"]) if "w8a32_avg_delta" in r else "N/A",
            "{:.4f}".format(r["w8a8_avg_delta"])  if "w8a8_avg_delta"  in r else "N/A",
        ))
    lines.append(sep)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Written:", out_path)


def process_image(image_name):
    py_path    = os.path.join(BASE, "multi_image", image_name, "python_detections.txt")
    fp32_path  = os.path.join(BASE, "multi_image", image_name, "csim_detections.txt")
    w8a32_path = os.path.join(BASE, "w8a32",       image_name, "csim_detections.txt")
    w8a8_path  = os.path.join(BASE, "w8a8",        image_name, "csim_detections.txt")
    out_dir    = os.path.join(BASE, "w8a8",        image_name)

    py_dets    = load_dets(py_path)
    fp32_dets  = load_dets(fp32_path)
    w8a32_dets = load_dets(w8a32_path)
    w8a8_dets  = load_dets(w8a8_path)

    if py_dets is None:
        print("  SKIP {} — no python_detections.txt".format(image_name))
        return None

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_table_4mode.txt")
    write_per_image_table(image_name, py_dets, fp32_dets, w8a32_dets, w8a8_dets, out_path)

    def avg_delta(matches, py_dets):
        deltas = [abs(d["score"] - py_dets[i]["score"]) for i, (_, d) in enumerate(matches) if d]
        return sum(deltas) / len(deltas) if deltas else float("nan")

    fp32_m,  _ = match_to_anchor(py_dets, fp32_dets  or [])
    w32_m,   _ = match_to_anchor(py_dets, w8a32_dets or [])
    w8_m,    _ = match_to_anchor(py_dets, w8a8_dets  or [])

    n = len(py_dets)
    r = {"image": image_name + ".jpg", "py_n": n}
    if fp32_dets  is not None:
        r["fp32_n"]         = len(fp32_dets)
        r["fp32_pct"]       = 100 * sum(1 for _,d in fp32_m  if d) / n
        r["fp32_avg_delta"] = avg_delta(fp32_m,  py_dets)
    if w8a32_dets is not None:
        r["w8a32_n"]         = len(w8a32_dets)
        r["w8a32_pct"]       = 100 * sum(1 for _,d in w32_m  if d) / n
        r["w8a32_avg_delta"] = avg_delta(w32_m,  py_dets)
    if w8a8_dets  is not None:
        r["w8a8_n"]          = len(w8a8_dets)
        r["w8a8_pct"]        = 100 * sum(1 for _,d in w8_m   if d) / n
        r["w8a8_avg_delta"]  = avg_delta(w8_m,   py_dets)
    return r


if __name__ == "__main__":
    print("Generating 4-mode comparison tables...")
    results = []
    for img in IMAGES:
        print("\n[{}]".format(img))
        r = process_image(img)
        if r:
            results.append(r)

    summary_path = os.path.join(BASE, "w8a8", "comparison_all_modes.txt")
    os.makedirs(os.path.join(BASE, "w8a8"), exist_ok=True)
    write_summary(results, summary_path)
    print("\nDone.")
