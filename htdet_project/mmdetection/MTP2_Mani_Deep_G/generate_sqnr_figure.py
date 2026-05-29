"""Generate fig_sqnr_layers.png — per-layer SQNR bar chart for W8A32 PTQ."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Per-layer SQNR from ptq_w8a8_report.txt (inf → capped at 52 for display)
# Format: (short_label, sqnr, module)
layers = [
    # Backbone
    ('stem',          44.4, 'backbone'),
    ('s0.exp',        43.6, 'backbone'), ('s0.dw',  48.3, 'backbone'), ('s0.proj', 43.4, 'backbone'),
    ('s1.0.exp',      44.7, 'backbone'), ('s1.0.dw',47.5, 'backbone'), ('s1.0.proj',42.8,'backbone'),
    ('s1.1.exp',      44.1, 'backbone'), ('s1.1.dw',46.7, 'backbone'), ('s1.1.proj',42.8,'backbone'),
    ('s1.2.exp',      44.3, 'backbone'), ('s1.2.dw',47.6, 'backbone'), ('s1.2.proj',43.1,'backbone'),
    ('s2.0.exp',      44.0, 'backbone'), ('s2.0.dw',49.2, 'backbone'), ('s2.0.proj',42.7,'backbone'),
    ('s2.kxk',        36.8, 'backbone'), ('s2.1x1', 43.8, 'backbone'),
    ('s2.proj',       43.2, 'backbone'), ('s2.fuse', 36.8,'backbone'),
    ('s3.0.exp',      41.0, 'backbone'), ('s3.0.dw',49.0, 'backbone'), ('s3.0.proj',42.6,'backbone'),
    ('s3.kxk',        38.4, 'backbone'), ('s3.1x1', 43.8, 'backbone'),
    ('s3.proj',       43.3, 'backbone'), ('s3.fuse', 36.9,'backbone'),
    ('s4.0.exp',      42.7, 'backbone'), ('s4.0.dw',48.3, 'backbone'), ('s4.0.proj',42.6,'backbone'),
    ('s4.kxk',        39.0, 'backbone'), ('s4.1x1', 43.1, 'backbone'),
    ('s4.proj',       43.0, 'backbone'), ('s4.fuse', 38.0,'backbone'),
    ('final',         42.1, 'backbone'),
    # FPN
    ('lat0',  48.2, 'fpn'), ('lat1', 48.1, 'fpn'), ('lat2', 47.9, 'fpn'), ('lat3', 47.2, 'fpn'),
    ('fpn0',  47.2, 'fpn'), ('fpn1', 47.2, 'fpn'), ('fpn2', 47.3, 'fpn'), ('fpn3', 52.0, 'fpn'),
    # Head
    ('cls0',  52.0,'head'), ('cls1', 52.0,'head'), ('cls2', 52.0,'head'), ('cls3', 52.0,'head'),
    ('reg0',  52.0,'head'), ('reg1', 52.0,'head'), ('reg2', 52.0,'head'), ('reg3', 52.0,'head'),
    ('cls_p', 40.6,'head'), ('reg_p',40.9,'head'),
]

labels  = [l[0] for l in layers]
sqnrs   = [l[1] for l in layers]
modules = [l[2] for l in layers]

color_map = {'backbone': '#4c9be8', 'fpn': '#e0a234', 'head': '#4cb87a'}
colors = [color_map[m] for m in modules]

INF_CAP = 52.0
display_sqnrs = [min(v, INF_CAP) for v in sqnrs]

fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(len(labels))
bars = ax.bar(x, display_sqnrs, color=colors, alpha=0.85, width=0.75)

ax.axhline(35, color='darkorange', linestyle=':', linewidth=1.2, label='Near-lossless threshold (35 dB)')
ax.axhline(np.mean([v for v in sqnrs if v < INF_CAP]), color='black', linestyle='--',
           linewidth=1.2, label=f'Mean SQNR = {np.mean([v for v in sqnrs if v < INF_CAP]):.1f} dB')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=7.5)
ax.set_ylabel('SQNR (dB)')
ax.set_ylim(30, 56)
ax.set_title('W8A32 PTQ — Per-Layer SQNR (192-Channel Model, 53 Conv2d Layers)', fontweight='bold')

from matplotlib.patches import Patch
legend_patches = [
    Patch(color='#4c9be8', alpha=0.85, label='Backbone'),
    Patch(color='#e0a234', alpha=0.85, label='FPN Neck'),
    Patch(color='#4cb87a', alpha=0.85, label='RetinaNet Head'),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color='darkorange', linestyle=':', lw=1.2, label='Near-lossless (35 dB)'),
    plt.Line2D([0], [0], color='black',      linestyle='--', lw=1.2, label=f'Mean = {np.mean([v for v in sqnrs if v < INF_CAP]):.1f} dB'),
], fontsize=8, loc='lower right')

# Mark inf bars
for i, (bar, v) in enumerate(zip(bars, sqnrs)):
    if v >= INF_CAP:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                '∞', ha='center', fontsize=7, color='#4cb87a')

fig.tight_layout()
fig.savefig('images/fig_sqnr_layers.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved fig_sqnr_layers.png")
