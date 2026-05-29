"""Generate fig_perclass_ap.png — per-class AP across channel reduction variants."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Per-class AP (mAP, IoU 0.50:0.95) from --eval-options classwise=True
classes = ['Holothurian', 'Echinus', 'Scallop', 'Starfish']
configs = {
    'Baseline\n(256ch)':  [0.330, 0.489, 0.330, 0.501],
    'Channel\n224':       [0.298, 0.466, 0.244, 0.478],
    'Channel\n192':       [0.285, 0.469, 0.262, 0.478],
    'Channel\n128':       [0.224, 0.453, 0.122, 0.462],
}
colors = ['#e06c75', '#61afef', '#98c379', '#e5c07b']

x = np.arange(len(configs))
n_cls = len(classes)
total_w = 0.75
bar_w = total_w / n_cls
offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_cls)

fig, ax = plt.subplots(figsize=(9, 5))
for i, (cls, color) in enumerate(zip(classes, colors)):
    vals = [v[i] for v in configs.values()]
    bars = ax.bar(x + offsets[i], vals, bar_w, label=cls, color=color, alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#333')

ax.set_xticks(x)
ax.set_xticklabels(list(configs.keys()), fontsize=10)
ax.set_ylabel('AP (IoU 0.50:0.95)')
ax.set_ylim(0, 0.62)
ax.set_title('Per-Class AP: Effect of Structured Channel Reduction', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)

fig.tight_layout()
fig.savefig('images/fig_perclass_ap.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved fig_perclass_ap.png")
