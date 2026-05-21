"""
generate_full_comparison.py
Produces a detailed 4-mode comparison: Python ref vs FP32 vs W8A32 vs W8A8.
Includes scores, bounding box coordinates, IoU, and per-axis deltas.
Run from: csim_validation/w8a8/
"""

import os, math

BASE  = os.path.join(os.path.dirname(__file__), '..')
IMAGES = ['YDXJ0001_10003', 'CHN083846_0291', 'GOPR0293_10229']
CLASS_NAMES = {0:'holothurian', 1:'echinus', 2:'scallop', 3:'starfish'}

def parse(path):
    dets = []
    try:
        with open(path) as f:
            n = int(f.readline())
            for _ in range(n):
                p = f.readline().split()
                dets.append((int(p[0]), float(p[1]),
                             float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    except FileNotFoundError:
        pass
    return dets

def iou(a, b):
    ix1=max(a[2],b[2]); iy1=max(a[3],b[3])
    ix2=min(a[4],b[4]); iy2=min(a[5],b[5])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    aa=(a[4]-a[2])*(a[5]-a[3]); ab=(b[4]-b[2])*(b[5]-b[3])
    uni=aa+ab-inter
    return inter/uni if uni>0 else 0

def best_match(ref, dets, used):
    bi, bv = -1, 0
    for j,d in enumerate(dets):
        if used[j]: continue
        v = iou(ref, d)
        if v > bv:
            bv, bi = v, j
    return bi, bv

def box_str(d):
    return f"[{d[2]:6.1f},{d[3]:6.1f},{d[4]:6.1f},{d[5]:6.1f}]"

def center_wh(d):
    cx = (d[2]+d[4])/2; cy = (d[3]+d[5])/2
    w  = d[4]-d[2];     h  = d[5]-d[3]
    return cx,cy,w,h

lines = []
W = 100

def hr(ch='='):   lines.append(ch*W)
def sec(t):       hr(); lines.append(t); hr()
def blank():      lines.append('')

sec("HTDet 4-Mode Detection Comparison Report")
lines.append("Modes: Python (PyTorch ref)  |  FP32 C-sim  |  W8A32 C-sim (weight-only INT8)  |  W8A8 C-sim (weight+act INT8)")
lines.append("Matching: IoU-anchored to Python reference detections (threshold 0.50)")
blank()

summary_rows = []

for img in IMAGES:
    py   = parse(f'{BASE}/multi_image/{img}/python_detections.txt')
    fp32 = parse(f'{BASE}/multi_image/{img}/csim_detections.txt')
    w32  = parse(f'{BASE}/w8a32/{img}/csim_detections.txt')
    w8   = parse(f'{BASE}/w8a8/{img}/csim_detections.txt')

    sec(f"IMAGE: {img}")
    lines.append(f"  Python ref : {len(py):3d} detections")
    lines.append(f"  FP32 C-sim : {len(fp32):3d} detections")
    lines.append(f"  W8A32 C-sim: {len(w32):3d} detections")
    lines.append(f"  W8A8 C-sim : {len(w8):3d} detections")
    blank()

    # ── Per-detection table ──────────────────────────────────────────────
    lines.append("── Per-Detection Match (IoU-matched to Python reference) ──")
    blank()
    hdr = (f"{'#':>2}  {'Class':<13} {'Mode':<10} "
           f"{'Score':>7}  {'ΔScore':>8}  "
           f"{'Box [x1,y1,x2,y2]':^31}  "
           f"{'cx':>6} {'cy':>6} {'w':>6} {'h':>6}  "
           f"{'IoU':>6}  {'Δcx':>6} {'Δcy':>6} {'Δw':>6} {'Δh':>6}")
    lines.append(hdr)
    lines.append('-'*W)

    used_fp32=[False]*len(fp32); used_w32=[False]*len(w32); used_w8=[False]*len(w8)
    matched_w8=[False]*len(w8)

    fp32_matched=w32_matched=w8_matched=0
    w8_score_errs=[]; w8_iou_vals=[]

    for i,ref in enumerate(py):
        rcx,rcy,rw,rh = center_wh(ref)
        cls = CLASS_NAMES[ref[0]]

        def row(name, dets, used):
            ji, vi = best_match(ref, dets, used)
            if ji < 0 or vi < 0.5:
                return (f"  {'':>2}  {cls:<13} {name:<10} "
                        f"{'MISS':>7}  {'':>8}  "
                        f"{'—':^31}  "
                        f"{'':>6} {'':>6} {'':>6} {'':>6}  "
                        f"{'—':>6}  {'':>6} {'':>6} {'':>6} {'':>6}"), -1, -1
            used[ji]=True
            d = dets[ji]
            ds = f"{d[1]:7.4f}"
            delta_s = f"{d[1]-ref[1]:+8.4f}"
            dcx,dcy,dw,dh = center_wh(d)
            return (f"  {i+1:>2}  {cls:<13} {name:<10} "
                    f"{ds}  {delta_s}  "
                    f"{box_str(d)}  "
                    f"{dcx:6.1f} {dcy:6.1f} {dw:6.1f} {dh:6.1f}  "
                    f"{vi:6.3f}  "
                    f"{dcx-rcx:+6.1f} {dcy-rcy:+6.1f} {dw-rw:+6.1f} {dh-rh:+6.1f}"), ji, vi

        # Python ref row (baseline)
        lines.append(f"  {i+1:>2}  {cls:<13} {'Python':<10} "
                     f"{ref[1]:7.4f}  {'baseline':>8}  "
                     f"{box_str(ref)}  "
                     f"{rcx:6.1f} {rcy:6.1f} {rw:6.1f} {rh:6.1f}  "
                     f"{'1.000':>6}  "
                     f"{'—':>6} {'—':>6} {'—':>6} {'—':>6}")

        r, _, _ = row('FP32', fp32, used_fp32)
        lines.append(r)
        r, _, _ = row('W8A32', w32, used_w32)
        lines.append(r)
        r, ji, vi = row('W8A8', w8, used_w8)
        lines.append(r)
        if ji>=0 and vi>=0.5:
            matched_w8[ji]=True; w8_matched+=1
            w8_score_errs.append(abs(w8[ji][1]-ref[1]))
            w8_iou_vals.append(vi)
        lines.append('')

    # Extra (unmatched) W8A8 detections
    extras = [w8[j] for j in range(len(w8)) if not matched_w8[j]]
    if extras:
        lines.append(f"  W8A8 false positives ({len(extras)}, no Python ref match at IoU>0.5):")
        for d in extras:
            lines.append(f"    {CLASS_NAMES[d[0]]:<13} score={d[1]:.4f}  {box_str(d)}")
        blank()

    # Per-image summary stats
    n_ref = len(py)
    w8_recall = w8_matched / n_ref if n_ref else 0
    w8_prec   = w8_matched / len(w8) if w8 else 0
    avg_iou   = sum(w8_iou_vals)/len(w8_iou_vals) if w8_iou_vals else 0
    avg_score_err = sum(w8_score_errs)/len(w8_score_errs) if w8_score_errs else 0

    lines.append(f"  ── Image Summary ──")
    lines.append(f"  W8A8 recall    : {w8_matched}/{n_ref} = {w8_recall:.1%}")
    lines.append(f"  W8A8 precision : {w8_matched}/{len(w8)} = {w8_prec:.1%}  ({len(extras)} FP)")
    lines.append(f"  Avg IoU (W8A8 matched): {avg_iou:.3f}")
    lines.append(f"  Avg |ΔScore|  (W8A8 vs Python): {avg_score_err:.4f}")
    blank()

    summary_rows.append((img, n_ref, len(fp32), len(w32), len(w8),
                         w8_matched, len(extras), w8_recall, w8_prec,
                         avg_iou, avg_score_err))

# ── Cross-image summary ───────────────────────────────────────────────────
sec("CROSS-IMAGE SUMMARY")
lines.append(f"{'Image':<22} {'Ref':>4} {'FP32':>5} {'W32':>5} {'W8A8':>5}  "
             f"{'Recall':>7} {'Prec':>7} {'FP':>4}  {'AvgIoU':>7} {'AvgΔS':>8}")
lines.append('-'*W)
for img,nr,nf,n32,n8,nm,nfp,rec,prec,aiou,ads in summary_rows:
    lines.append(f"  {img:<20} {nr:>4} {nf:>5} {n32:>5} {n8:>5}  "
                 f"{rec:>7.1%} {prec:>7.1%} {nfp:>4}  {aiou:>7.3f} {ads:>8.4f}")
blank()

all_iou   = [r[9] for r in summary_rows]
all_score = [r[10] for r in summary_rows]
all_rec   = [r[7] for r in summary_rows]
all_prec  = [r[8] for r in summary_rows]
lines.append(f"  {'MEAN':<20} {'':>4} {'':>5} {'':>5} {'':>5}  "
             f"{sum(all_rec)/len(all_rec):>7.1%} {sum(all_prec)/len(all_prec):>7.1%} {'':>4}  "
             f"{sum(all_iou)/len(all_iou):>7.3f} {sum(all_score)/len(all_score):>8.4f}")
blank()

lines.append("KEY:")
lines.append("  ΔScore = mode_score − python_score  (+ means mode is more confident than Python)")
lines.append("  AvgΔS  = mean absolute score error vs Python reference")
lines.append("  Δcx/Δcy/Δw/Δh = box center and size delta vs Python reference (pixels)")
lines.append("  IoU    = overlap between mode box and Python reference box")
lines.append("  FP     = false positives (W8A8 dets with no Python ref match at IoU>0.5)")

out = '\n'.join(lines)
outpath = os.path.join(os.path.dirname(__file__), 'comparison_all_modes.txt')
with open(outpath, 'w') as f:
    f.write(out)
print(out)
print(f"\n[Saved to {outpath}]")
