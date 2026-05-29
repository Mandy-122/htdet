"""
Generate CSIM comparison figures and qualitative detection images for MTP2.
Run from MTP2_Mani_Deep_G/:
    python3 generate_csim_figures.py
"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

os.makedirs("images", exist_ok=True)

ROOT   = ".."
URPC   = "/workspace/ckarfa/htdet/data/urpc/val2018/images"
MULTI  = f"{ROOT}/csim_validation/multi_image"
W8A32  = f"{ROOT}/csim_validation/w8a32"

CLASS_NAMES = {0: 'holothurian', 1: 'echinus', 2: 'scallop', 3: 'starfish'}
CLASS_COLORS = {
    'holothurian': '#e74c3c',
    'echinus':     '#3498db',
    'scallop':     '#2ecc71',
    'starfish':    '#f39c12',
}

# ─── parse detection file ───────────────────────────────────────────────────
def parse_dets(path):
    dets = []
    try:
        with open(path) as f:
            n = int(f.readline().strip())
            for _ in range(n):
                parts = f.readline().split()
                cls, score = int(parts[0]), float(parts[1])
                x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                dets.append({'cls': cls, 'score': score, 'box': [x1, y1, x2, y2]})
    except Exception as e:
        print(f"  Warning: could not parse {path}: {e}")
    return dets

# ─── draw detections on an axis ──────────────────────────────────────────────
def draw_dets(ax, img, dets, title, score_thr=0.25):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    for d in dets:
        if d['score'] < score_thr:
            continue
        x1, y1, x2, y2 = d['box']
        cname = CLASS_NAMES.get(d['cls'], str(d['cls']))
        color = CLASS_COLORS.get(cname, 'white')
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2-x1, y2-y1,
            boxstyle="square,pad=0", linewidth=1.8,
            edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(y1-3, 2), f"{cname[0].upper()} {d['score']:.2f}",
                color='white', fontsize=6.5,
                bbox=dict(facecolor=color, alpha=0.75, pad=1, edgecolor='none'))

# ─── Fig A: side-by-side Python vs CSIM float32 for 3 images ────────────────
def fig_csim_float32():
    test_imgs = ['GOPR0293_10229', 'CHN083846_0291', 'YDXJ0001_10003']
    fig, axes = plt.subplots(len(test_imgs), 2, figsize=(12, 5.5*len(test_imgs)))

    for row, name in enumerate(test_imgs):
        img_path = f"{URPC}/{name}.jpg"
        py_path  = f"{MULTI}/{name}/python_detections.txt"
        cs_path  = f"{MULTI}/{name}/csim_detections.txt"

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            img = np.zeros((192, 192, 3), dtype=np.uint8)

        py_dets = parse_dets(py_path)
        cs_dets = parse_dets(cs_path)

        ax_py = axes[row, 0] if len(test_imgs) > 1 else axes[0]
        ax_cs = axes[row, 1] if len(test_imgs) > 1 else axes[1]

        n_py = sum(1 for d in py_dets if d['score'] >= 0.25)
        n_cs = sum(1 for d in cs_dets if d['score'] >= 0.25)

        draw_dets(ax_py, img, py_dets, f"Python (float32) — {name}\n{n_py} dets (score≥0.25)")
        draw_dets(ax_cs, img, cs_dets, f"C-Sim (float32) — {name}\n{n_cs} dets (score≥0.25)")

    # legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle('Python Float32 vs C-Simulation: Bounding Box Comparison\n(192×192 HTDet Model)',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig('images/fig_csim_float32_comparison.png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    print("Saved fig_csim_float32_comparison.png")

# ─── Fig B: side-by-side Python vs CSIM W8A32 ────────────────────────────────
def fig_csim_w8a32():
    test_imgs = ['GOPR0293_10229', 'CHN083846_0291', 'YDXJ0001_10003']
    fig, axes = plt.subplots(len(test_imgs), 2, figsize=(12, 5.5*len(test_imgs)))

    for row, name in enumerate(test_imgs):
        img_path = f"{URPC}/{name}.jpg"
        py_path  = f"{MULTI}/{name}/python_detections.txt"    # float32 reference
        cs_path  = f"{W8A32}/{name}/csim_detections.txt"

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            img = np.zeros((192, 192, 3), dtype=np.uint8)

        py_dets = parse_dets(py_path)
        cs_dets = parse_dets(cs_path)

        ax_py = axes[row, 0]
        ax_cs = axes[row, 1]

        n_py = sum(1 for d in py_dets if d['score'] >= 0.25)
        n_cs = sum(1 for d in cs_dets if d['score'] >= 0.25)

        draw_dets(ax_py, img, py_dets, f"Python (float32 ref) — {name}\n{n_py} dets (score≥0.25)")
        draw_dets(ax_cs, img, cs_dets, f"C-Sim (W8A32 PTQ) — {name}\n{n_cs} dets (score≥0.25)")

    patches = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle('Python Float32 vs C-Simulation W8A32 (INT8 Weights): Bounding Box Comparison',
                 fontsize=12, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig('images/fig_csim_w8a32_comparison.png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    print("Saved fig_csim_w8a32_comparison.png")

# ─── Fig C: 5-image CSIM summary bar chart ───────────────────────────────────
def fig_csim_summary():
    summary = {
        'CHN083846\n_0291':  {'py': 37, 'cs_f32': 37, 'match_f32': 100.0, 'cs_w8':  37, 'match_w8': 100.0},
        'G0024172\n_0636':   {'py': 75, 'cs_f32': 79, 'match_f32':  89.9, 'cs_w8':  None, 'match_w8': None},
        'GOPR0293\n_10229':  {'py': 19, 'cs_f32': 19, 'match_f32': 100.0, 'cs_w8':  19, 'match_w8':  94.7},
        'YDXJ0001\n_10003':  {'py':  7, 'cs_f32':  7, 'match_f32': 100.0, 'cs_w8':   7, 'match_w8': 100.0},
        'YN010001\n_1312':   {'py': 14, 'cs_f32': 15, 'match_f32':  93.3, 'cs_w8':  None, 'match_w8': None},
    }

    names = list(summary.keys())
    py_counts   = [summary[n]['py']       for n in names]
    f32_counts  = [summary[n]['cs_f32']   for n in names]
    f32_match   = [summary[n]['match_f32'] for n in names]
    w8_match    = [summary[n]['match_w8'] if summary[n]['match_w8'] else 0 for n in names]

    x = np.arange(len(names))
    w = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: detection count bars
    b1 = ax1.bar(x - w, py_counts,  w, label='Python (float32)', color='steelblue',  alpha=0.85)
    b2 = ax1.bar(x,     f32_counts, w, label='CSIM (float32)',   color='crimson',    alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel('Number of Detections (score ≥ 0.20)')
    ax1.set_title('Python vs CSIM Detection Counts\n(Float32, 5 Test Images)')
    ax1.legend()
    ax1.set_ylim(0, 95)
    for b in [b1, b2]:
        for r in b:
            ax1.text(r.get_x()+r.get_width()/2, r.get_height()+0.8,
                     str(int(r.get_height())), ha='center', fontsize=8)

    # Right: match percentage
    valid_f32 = [(n, f32_match[i]) for i, n in enumerate(names)]
    valid_w8  = [(n, w8_match[i])  for i, n in enumerate(names) if summary[n]['match_w8'] is not None]

    f32_x = [x[names.index(n)] - 0.15 for n, _ in valid_f32]
    f32_y = [v for _, v in valid_f32]
    w8_x  = [x[names.index(n)] + 0.15 for n, _ in valid_w8]
    w8_y  = [v for _, v in valid_w8]

    ax2.bar(f32_x, f32_y, 0.28, label='Float32 CSIM', color='steelblue', alpha=0.85)
    ax2.bar(w8_x,  w8_y,  0.28, label='W8A32 CSIM',   color='darkorange', alpha=0.85)
    ax2.axhline(100, color='green', linestyle='--', linewidth=1.2, alpha=0.6, label='100% match')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel('Detection Match Rate (%)')
    ax2.set_title('Python vs CSIM Match Rate\n(IoU ≥ 0.5, Float32 and W8A32)')
    ax2.set_ylim(70, 105)
    ax2.legend(fontsize=9)
    for xi, yi in zip(f32_x, f32_y):
        ax2.text(xi, yi+0.5, f'{yi:.0f}%', ha='center', fontsize=8)
    for xi, yi in zip(w8_x, w8_y):
        ax2.text(xi, yi+0.5, f'{yi:.0f}%', ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig('images/fig_csim_summary.png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    print("Saved fig_csim_summary.png")

# ─── Fig D: qualitative detection images (preds_vis/good) ────────────────────
def fig_qualitative_detections():
    good_dir = f"{ROOT}/preds_vis/good"
    all_imgs = sorted(os.listdir(good_dir))[:6]   # pick first 6

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    class_labels = {
        'holothurian': '#e74c3c',
        'echinus':     '#3498db',
        'scallop':     '#2ecc71',
        'starfish':    '#f39c12',
    }

    for ax, fname in zip(axes, all_imgs):
        try:
            img = Image.open(os.path.join(good_dir, fname)).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.imshow(np.zeros((192,192,3), dtype=np.uint8))
        ax.axis('off')
        ax.set_title(fname.replace('_', '\n'), fontsize=8)

    for ax in axes[len(all_imgs):]:
        ax.axis('off')

    patches = [mpatches.Patch(color=v, label=k) for k, v in class_labels.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Qualitative Detection Results — HTDet on URPC\n(MobileViT-S + FA-FPN, score ≥ 0.20)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig('images/fig_qualitative_detections.png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    print("Saved fig_qualitative_detections.png")

# ─── Fig E: single-image 4-way comparison (Float32 Py | Float32 CSIM | W8A32 CSIM) ──
def fig_single_image_triple():
    name     = 'GOPR0293_10229'
    img_path = f"{URPC}/{name}.jpg"
    py_path  = f"{MULTI}/{name}/python_detections.txt"
    f32_path = f"{MULTI}/{name}/csim_detections.txt"
    w8_path  = f"{W8A32}/{name}/csim_detections.txt"

    try:
        img = np.array(Image.open(img_path).convert('RGB'))
    except Exception:
        img = np.zeros((192, 192, 3), dtype=np.uint8)

    py_dets  = parse_dets(py_path)
    f32_dets = parse_dets(f32_path)
    w8_dets  = parse_dets(w8_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    draw_dets(axes[0], img, py_dets,  f"Python (float32)\n{sum(d['score']>=0.2 for d in py_dets)} dets")
    draw_dets(axes[1], img, f32_dets, f"C-Sim float32\n{sum(d['score']>=0.2 for d in f32_dets)} dets")
    draw_dets(axes[2], img, w8_dets,  f"C-Sim W8A32 (PTQ)\n{sum(d['score']>=0.2 for d in w8_dets)} dets")

    patches = [mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f'Detection Comparison: Python Float32 | CSIM Float32 | CSIM W8A32\n({name})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig('images/fig_single_image_triple.png', bbox_inches='tight', dpi=140)
    plt.close(fig)
    print("Saved fig_single_image_triple.png")

# ─── Fig F: score delta scatter — Python vs CSIM per-detection ───────────────
def fig_score_delta():
    """Scatter plot: Python score vs CSIM score for all matched detections."""
    py_scores, cs_scores, colors = [], [], []
    all_classes = []
    data_root = MULTI

    for name in ['GOPR0293_10229', 'CHN083846_0291', 'YDXJ0001_10003', 'YN010001_1312']:
        py_dets = parse_dets(f"{data_root}/{name}/python_detections.txt")
        cs_dets = parse_dets(f"{data_root}/{name}/csim_detections.txt")
        # simple pairing by rank order (same detections matched)
        n = min(len(py_dets), len(cs_dets))
        for i in range(n):
            py_scores.append(py_dets[i]['score'])
            cs_scores.append(cs_dets[i]['score'])
            cname = CLASS_NAMES.get(py_dets[i]['cls'], 'echinus')
            all_classes.append(cname)
            colors.append(CLASS_COLORS.get(cname, 'gray'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Scatter: Python score vs CSIM score
    for cname, color in CLASS_COLORS.items():
        idx = [i for i, c in enumerate(all_classes) if c == cname]
        ax1.scatter([py_scores[i] for i in idx],
                    [cs_scores[i] for i in idx],
                    c=color, alpha=0.65, s=30, label=cname, zorder=3)
    lim = [0.15, 1.02]
    ax1.plot(lim, lim, 'k--', linewidth=1.2, alpha=0.5, label='y = x (perfect match)')
    ax1.set_xlabel('Python Float32 Score')
    ax1.set_ylabel('C-Sim Float32 Score')
    ax1.set_title('Detection Score Correlation\n(Python Float32 vs C-Sim Float32)')
    ax1.set_xlim(*lim); ax1.set_ylim(*lim)
    ax1.legend(fontsize=8)

    # Histogram of |score delta|
    deltas = [abs(py_scores[i] - cs_scores[i]) for i in range(len(py_scores))]
    ax2.hist(deltas, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(np.mean(deltas), color='crimson', linestyle='--', linewidth=1.5,
                label=f'Mean |Δ| = {np.mean(deltas)*1000:.2f}×10⁻³')
    ax2.set_xlabel('|Python Score − C-Sim Score|')
    ax2.set_ylabel('Count')
    ax2.set_title('Score Difference Distribution\n(All Matched Detections, Float32)')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig('images/fig_score_delta.png', bbox_inches='tight', dpi=130)
    plt.close(fig)
    print("Saved fig_score_delta.png")

# ─── Run all ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig_csim_float32()
    fig_csim_w8a32()
    fig_csim_summary()
    fig_qualitative_detections()
    fig_single_image_triple()
    fig_score_delta()
    print("\nAll CSIM figures saved to images/")
