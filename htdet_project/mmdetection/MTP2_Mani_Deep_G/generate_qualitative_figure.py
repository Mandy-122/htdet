"""Generate fig_qualitative_detections.png — 2x3 grid of clean detection results."""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, "images"), exist_ok=True)

SELECTED = [
    'GOPR0293_31039.jpg',
    'GOPR0293_30246.jpg',
    'GOPR0293_24228.jpg',
    'GOPR0293_36471.jpg',
    'GOPR0293_29736.jpg',
    'GOPR0293_11179.jpg',
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'clean_detections')

CAPTIONS = [
    '(a) 4-class scene: holothurian, echinus,\nscallop, starfish',
    '(b) All 4 classes with varied scales',
    '(c) Rocky terrain: multi-class detection',
    '(d) High-confidence detections (score≥0.50)',
    '(e) Sparse scene: 5 clean detections',
    '(f) Multi-class: 6 well-separated boxes',
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, fname, caption in zip(axes, SELECTED, CAPTIONS):
    path = os.path.join(SRC_DIR, fname)
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(caption, fontsize=8.5, pad=4)
    ax.axis('off')

# Legend
legend_items = [
    mpatches.Patch(color=(220/255, 0, 0),       label='Holothurian'),
    mpatches.Patch(color=(0, 0, 220/255),        label='Echinus'),
    mpatches.Patch(color=(0, 180/255, 0),        label='Scallop'),
    mpatches.Patch(color=(255/255, 165/255, 0),  label='Starfish'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('HTDet Qualitative Detection Results — URPC Validation Set (score $\\geq$ 0.50)',
             fontsize=12, fontweight='bold', y=0.995)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(os.path.join(SCRIPT_DIR, 'images', 'fig_qualitative_detections.png'), bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved fig_qualitative_detections.png")
