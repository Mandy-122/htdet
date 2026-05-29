"""Generate fig_compression_study.png for Chapter 4 compression section."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("images", exist_ok=True)

STYLE = {
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.35,
    'axes.spines.top': False, 'axes.spines.right': False,
}
plt.rcParams.update(STYLE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: mAP50 vs GFLOPs scatter ────────────────────────────────────────────
configs = {
    'Baseline\n(256ch)':            (198.9, 76.34, 12.4, '#2878b5', 130, 'o'),
    'Weight Pruned\n30%':           (198.9, 76.18, 12.4, '#e84141', 100, 'D'),
    'Channel-224\n(22% savings)':   (155.7, 73.37, 10.7, '#e0a234', 100, 's'),
    'Channel-192\n(41% savings)':   (118.1, 72.87,  9.2, '#4cb87a', 100, 's'),
    'Channel-128\n(70% savings)':   ( 59.9, 63.94,  6.9, '#e74c3c', 100, 's'),
}

for label, (gflops, m50, params, color, size, marker) in configs.items():
    ax1.scatter(gflops, m50, s=size, color=color, marker=marker,
                zorder=5, edgecolors='white', linewidths=0.8)
    offset = (4, 3) if 'Baseline' not in label and 'Weight' not in label else (4, -8)
    if 'FA-FPN' in label:
        offset = (4, 3)
    if 'Weight' in label:
        offset = (4, -10)
    ax1.annotate(label, (gflops, m50),
                 textcoords='offset points', xytext=offset,
                 fontsize=8, color=color)

ax1.set_xlabel('GFLOPs')
ax1.set_ylabel('mAP$_{50}$ (%)')
ax1.set_title('Accuracy vs. Computational Cost')
ax1.set_xlim(30, 230)
ax1.set_ylim(58, 82)
ax1.axvline(198.9, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# ── Right: bar chart — mAP50 and GFLOPs reduction ────────────────────────────
labels  = ['Baseline', 'Weight\nPruned 30%', 'Channel\n224', 'Channel\n192', 'Channel\n128']
map50   = [76.34, 76.18, 73.37, 72.87, 63.94]
gflops_red = [0, 0, 21.7, 40.6, 69.9]
colors  = ['#2878b5', '#e84141', '#e0a234', '#4cb87a', '#e74c3c']

x = np.arange(len(labels))
w = 0.38

ax2_twin = ax2.twinx()
b1 = ax2.bar(x - w/2, map50,     w, color=colors, alpha=0.85, label='mAP$_{50}$ (%)')
b2 = ax2_twin.bar(x + w/2, gflops_red, w, color=colors, alpha=0.40, label='GFLOPs reduction (%)')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('mAP$_{50}$ (%)')
ax2.set_ylim(50, 88)
ax2_twin.set_ylabel('GFLOPs Reduction (%)')
ax2_twin.set_ylim(0, 100)
ax2.set_title('mAP$_{50}$ and GFLOPs Reduction per Variant')

for rect, val in zip(b1, map50):
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for rect, val in zip(b2, gflops_red):
    if val > 0:
        ax2_twin.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.5,
                      f'{val:.0f}%', ha='center', va='bottom', fontsize=7.5, color='gray')

lines = [mpatches.Patch(color='steelblue', alpha=0.85, label='mAP$_{50}$ (%)'),
         mpatches.Patch(color='steelblue', alpha=0.40, label='GFLOPs Reduction (%)')]
ax2.legend(handles=lines, loc='lower left', fontsize=9)

fig.suptitle('HTDet Compression Study: Weight Pruning, Channel Reduction, and Quantization',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('images/fig_compression_study.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved fig_compression_study.png")
