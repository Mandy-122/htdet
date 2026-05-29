"""Generate score delta distribution figure for Chapter 5."""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'images'), exist_ok=True)

BASE = '/workspace/ckarfa/htdet/htdet_project/mmdetection/csim_validation/multi_image'
deltas, images = [], []
for d in sorted(os.listdir(BASE)):
    fpath = os.path.join(BASE, d, 'comparison_table.txt')
    if not os.path.exists(fpath): continue
    with open(fpath) as f:
        for line in f:
            m = re.search(r'\d+\s+\w+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                deltas.append(abs(float(m.group(4))))
                images.append(d)

arr = np.array(deltas)

# ── Figure: score delta distribution ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Left: histogram of ALL deltas
bins = np.array([0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.35])
counts, _ = np.histogram(arr, bins=bins)
centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
labels  = ['<0.5m', '0.5–1m', '1–2m', '2–5m', '5–10m', '10–50m', '50–100m', '>100m']
colors  = ['#4cb87a' if c < 0.005 else '#e0a234' if c < 0.05 else '#e84141'
           for c in centers]
ax1.bar(range(len(counts)), counts, color=colors, alpha=0.85, edgecolor='white')
ax1.set_xticks(range(len(counts)))
ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
ax1.set_ylabel('Number of detections')
ax1.set_xlabel('|ΔScore| range')
ax1.set_title('Score Delta Distribution\n(148 matched detections, 5 images)')
# Legend
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(color='#4cb87a', label=f'Float rounding (<5m): {(arr<0.005).sum()}'),
    Patch(color='#e0a234', label=f'Moderate (5–50m): {((arr>=0.005)&(arr<0.05)).sum()}'),
    Patch(color='#e84141', label=f'Large (>50m): {(arr>=0.05).sum()}'),
], fontsize=8)

# Right: cumulative CDF
sorted_d = np.sort(arr)
cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
ax2.plot(sorted_d * 1000, cdf * 100, color='steelblue', lw=2)
ax2.axvline(1.0,  color='#4cb87a', linestyle='--', lw=1.2,
            label=f'{(arr<0.001).sum()}/{len(arr)} detections < 1m ({(arr<0.001).mean()*100:.0f}%)')
ax2.axvline(5.0,  color='#e0a234', linestyle='--', lw=1.2,
            label=f'{(arr<0.005).sum()}/{len(arr)} detections < 5m ({(arr<0.005).mean()*100:.0f}%)')
ax2.set_xlabel('|ΔScore| (×10⁻³)')
ax2.set_ylabel('Cumulative %')
ax2.set_title('CDF of Score Differences\n(m = ×10⁻³)')
ax2.set_xlim(0, 15)
ax2.legend(fontsize=8)

fig.suptitle('Python vs C-Simulation: Score Delta Analysis (Float32, 5 Validation Images)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = os.path.join(SCRIPT_DIR, 'images', 'fig_score_delta.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")
print(f"Stats: mean={arr.mean()*1000:.3f}e-3  max={arr.max():.4f}  "
      f"<1e-3:{(arr<0.001).sum()}/148  <5e-3:{(arr<0.005).sum()}/148")
