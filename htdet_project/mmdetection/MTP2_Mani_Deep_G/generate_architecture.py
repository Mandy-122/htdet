"""
Generate HTDet architecture diagram — top-to-bottom, names only.
python3 generate_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(figsize=(5.5, 7.5))
ax.set_xlim(0, 5.5)
ax.axis('off')

# ── colour palette ────────────────────────────────────────────────────────────
C_IO   = '#7f8c8d'
C_BACK = '#2878b5'
C_NECK = '#e84141'
C_HEAD = '#2ecc71'
C_OUT  = '#f39c12'
BW     = 3.4   # box width
BH     = 0.90  # box height
CX     = 5.5 / 2  # centre x

def block(ax, cy, label, color, sublabel=''):
    x = CX - BW/2
    y = cy - BH/2
    rect = FancyBboxPatch((x, y), BW, BH,
                          boxstyle="round,pad=0.06,rounding_size=0.15",
                          linewidth=1.6, edgecolor='#2c3e50',
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    ty = cy + (0.14 if sublabel else 0)
    ax.text(CX, ty, label, ha='center', va='center',
            fontsize=17, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(CX, cy - 0.22, sublabel, ha='center', va='center',
                fontsize=9, color='#ecf0f1', zorder=4)

def arrow_label(ax, y_top, y_bot, label):
    ymid = (y_top + y_bot) / 2
    ax.annotate('', xy=(CX, y_bot + 0.02), xytext=(CX, y_top - 0.02),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.8))
    ax.text(CX + 1.0, ymid, label, ha='left', va='center',
            fontsize=8, color='#555555',
            bbox=dict(facecolor='#f2f3f4', edgecolor='#aab7b8',
                      boxstyle='round,pad=0.2', linewidth=0.8))

# ── blocks — tighter vertical spacing ─────────────────────────────────────────
GAP  = 0.42   # gap between box edge and next box edge
STEP = BH + GAP
TOP  = 6.60   # top block centre (leaves room for title above)
positions = {
    'input':  TOP,
    'back':   TOP - STEP,
    'neck':   TOP - 2*STEP,
    'head':   TOP - 3*STEP,
    'output': TOP - 4*STEP,
}
BOT = positions['output'] - BH/2  # bottom of last box
ax.set_ylim(BOT - 0.15, TOP + BH/2 + 0.70)  # tight crop with small margins

block(ax, positions['input'],  'Input Image',    C_IO,   '640 × 640 × 3')
block(ax, positions['back'],   'MobileViT-S',    C_BACK, 'Backbone')
block(ax, positions['neck'],   'FPN',             C_NECK, 'Neck  (5 levels, 256 ch)')
block(ax, positions['head'],   'RetinaNet Head',  C_HEAD, 'Detection Head')
block(ax, positions['output'], 'Detections',      C_OUT,  'class · score · bbox')

# ── arrows + feature labels ───────────────────────────────────────────────────
arrow_label(ax,
            positions['input'] - BH/2,
            positions['back']  + BH/2,
            '640×640×3')

arrow_label(ax,
            positions['back']  - BH/2,
            positions['neck']  + BH/2,
            'C₁ 64 ch\nC₂ 96 ch\nC₃ 128 ch\nC₄ 640 ch')

arrow_label(ax,
            positions['neck']  - BH/2,
            positions['head']  + BH/2,
            'P₂–P₆\n256 ch each')

arrow_label(ax,
            positions['head']  - BH/2,
            positions['output']+ BH/2,
            'cls + reg logits\n→ NMS')

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(CX, TOP + BH/2 + 0.42, 'HTDet Architecture',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#2c3e50')

fig.tight_layout()
fig.savefig('images/htdet_overview.png', bbox_inches='tight', dpi=160)
plt.close(fig)
print("Saved images/htdet_overview.png")
