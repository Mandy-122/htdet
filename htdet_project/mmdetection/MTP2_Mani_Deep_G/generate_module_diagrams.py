"""
Generate 3 compact module diagrams:
  - mobilevit_backbone.png
  - fpn_neck.png
  - retina_head.png
Run from MTP2_Mani_Deep_G/: python3 generate_module_diagrams.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("images", exist_ok=True)

# ── shared drawing helpers ────────────────────────────────────────────────────
def make_fig(n_rows, col_width=3.2):
    """Return (fig, ax) sized to fit n_rows blocks with tight margins."""
    BH, GAP, PAD = 0.52, 0.30, 0.30
    height = n_rows * (BH + GAP) + PAD * 2 + 0.45  # +0.45 for title
    fig, ax = plt.subplots(figsize=(col_width, height))
    ax.set_xlim(0, col_width)
    ax.set_ylim(0, height)
    ax.axis('off')
    return fig, ax, col_width / 2, BH, GAP, PAD, height

def block(ax, cx, cy, bw, bh, label, color, text_size=9.5):
    rect = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                          boxstyle="round,pad=0.04,rounding_size=0.10",
                          linewidth=1.3, edgecolor='#2c3e50',
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=text_size, fontweight='bold', color='white', zorder=4)

def arrow(ax, cx, y_top, y_bot):
    ax.annotate('', xy=(cx, y_bot + 0.02), xytext=(cx, y_top - 0.02),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.4))

def side_label(ax, cx, bw, ymid, text):
    ax.text(cx + bw/2 + 0.08, ymid, text,
            ha='left', va='center', fontsize=7.2, color='#444444',
            bbox=dict(facecolor='#f4f4f4', edgecolor='#cccccc',
                      boxstyle='round,pad=0.15', linewidth=0.7))

def title(ax, cx, top_y, text):
    ax.text(cx, top_y - 0.18, text, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#2c3e50')

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = '#2878b5'
RED    = '#e84141'
GREEN  = '#2ecc71'
ORANGE = '#f39c12'
GREY   = '#7f8c8d'
TEAL   = '#16a085'
PURPLE = '#8e44ad'
DARK   = '#2c3e50'

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MobileViT-S Backbone
# ══════════════════════════════════════════════════════════════════════════════
def fig_backbone():
    layers = [
        # (label,               color,   side annotation)
        ('Input  640×640×3',   GREY,    ''),
        ('Stem Conv 3×3',      DARK,    '320×320×16'),
        ('Stage 0  MBConv×1',  BLUE,    '160×160×32'),
        ('Stage 1  MBConv×1',  BLUE,    '80×80×64  → C₁'),
        ('Stage 2  MBConv×3',  BLUE,    '40×40×96  → C₂'),
        ('Stage 3  MBConv+ViT', TEAL,   '20×20×128 → C₃'),
        ('Stage 4  MBConv+ViT', TEAL,   '20×20×160'),
        ('Final Conv 1×1',     PURPLE,  '20×20×640 → C₄'),
    ]

    N    = len(layers)
    BH   = 0.52
    GAP  = 0.30
    PAD  = 0.25
    CW   = 4.8          # wider figure
    BW   = 2.55         # box width (leaves ~1.85 for side label)
    CX   = BW / 2 + 0.15  # box centre x (slightly right of left edge)
    height = N * (BH + GAP) - GAP + PAD * 2 + 0.55

    fig, ax = plt.subplots(figsize=(CW, height))
    ax.set_xlim(0, CW)
    ax.set_ylim(0, height)
    ax.axis('off')

    top = height - PAD - 0.45
    title(ax, CW / 2, height - 0.02, 'MobileViT-S  Backbone')

    for i, (lbl, col, ann) in enumerate(layers):
        cy = top - i * (BH + GAP)
        block(ax, CX, cy, BW, BH, lbl, col, text_size=9)
        if ann:
            ax.text(CX + BW/2 + 0.12, cy, ann,
                    ha='left', va='center', fontsize=8, color='#333333',
                    bbox=dict(facecolor='#f4f4f4', edgecolor='#bbbbbb',
                              boxstyle='round,pad=0.18', linewidth=0.7))
        if i < N - 1:
            arrow(ax, CX, cy - BH/2, cy - BH/2 - GAP)

    ax.set_ylim(top - (N-1)*(BH+GAP) - BH/2 - 0.15, height)
    fig.tight_layout()
    fig.savefig('images/mobilevit_backbone.png', bbox_inches='tight', dpi=160)
    plt.close(fig)
    print("Saved mobilevit_backbone.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FPN Neck
# ══════════════════════════════════════════════════════════════════════════════
def fig_fpn():
    layers = [
        # (label,                      color,   side annotation)
        ('C₁  C₂  C₃  C₄  (inputs)',  GREY,    '64 / 96 / 128 / 640 ch'),
        ('Lateral Conv 1×1  ×4',       ORANGE,  '→ 256 ch each'),
        ('Top-down Upsample + Add',    ORANGE,  'semantic fusion'),
        ('FPN Conv 3×3  ×4',           RED,     '→ P₂ P₃ P₄ P₅'),
        ('Extra Conv 3×3  /2',         RED,     '→ P₆  (from C₄)'),
        ('P₂ P₃ P₄ P₅ P₆  (outputs)', GREY,    '5 × 256 ch'),
    ]

    N    = len(layers)
    BH   = 0.52
    GAP  = 0.30
    PAD  = 0.25
    CW   = 4.8
    BW   = 2.55
    CX   = BW / 2 + 0.15
    height = N * (BH + GAP) - GAP + PAD * 2 + 0.55

    fig, ax = plt.subplots(figsize=(CW, height))
    ax.set_xlim(0, CW)
    ax.set_ylim(0, height)
    ax.axis('off')

    top = height - PAD - 0.45
    title(ax, CW / 2, height - 0.02, 'FPN  Neck')

    for i, (lbl, col, ann) in enumerate(layers):
        cy = top - i * (BH + GAP)
        block(ax, CX, cy, BW, BH, lbl, col, text_size=9)
        if ann:
            ax.text(CX + BW/2 + 0.12, cy, ann,
                    ha='left', va='center', fontsize=8, color='#333333',
                    bbox=dict(facecolor='#f4f4f4', edgecolor='#bbbbbb',
                              boxstyle='round,pad=0.18', linewidth=0.7))
        if i < N - 1:
            arrow(ax, CX, cy - BH/2, cy - BH/2 - GAP)

    ax.set_ylim(top - (N-1)*(BH+GAP) - BH/2 - 0.15, height)
    fig.tight_layout()
    fig.savefig('images/fpn_neck.png', bbox_inches='tight', dpi=160)
    plt.close(fig)
    print("Saved fpn_neck.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RetinaNet Head
# ══════════════════════════════════════════════════════════════════════════════
def fig_head():
    BH    = 0.56
    GAP   = 0.35
    PAD   = 0.25
    CW    = 5.2
    COL_W = 1.85
    CX_L  = CW * 0.27
    CX_R  = CW * 0.73

    # 3 rows per column: conv block, predictor, side label
    # + shared input + fork labels + shared output
    N_ROWS = 2   # conv stack + predictor
    height = (N_ROWS + 2) * (BH + GAP) + PAD * 2 + 0.90

    fig, ax = plt.subplots(figsize=(CW, height))
    ax.set_xlim(0, CW)
    ax.set_ylim(0, height)
    ax.axis('off')

    top = height - PAD - 0.45
    title(ax, CW/2, height - 0.02, 'RetinaNet  Head')

    # ── shared input ──────────────────────────────────────────────────────────
    cy_in = top
    block(ax, CW/2, cy_in, CW - 0.20, BH,
          'P₂ – P₆  (shared input, 256 ch)', GREY, text_size=8.5)

    # ── subnet labels (directly below input box, before arrows) ──────────────
    cy_lbl = cy_in - BH/2 - 0.32
    ax.text(CX_L, cy_lbl, 'Classification Subnet',
            ha='center', va='center', fontsize=9,
            color='#1a8a4a', fontweight='bold')
    ax.text(CX_R, cy_lbl, 'Regression Subnet',
            ha='center', va='center', fontsize=9,
            color='#1a4a8a', fontweight='bold')

    # ── fork arrows (from input box to below labels) ──────────────────────────
    fork_y  = cy_in - BH/2 - 0.06
    arrow_y = cy_lbl - 0.22
    for cx_col in [CX_L, CX_R]:
        ax.annotate('', xy=(cx_col, arrow_y),
                    xytext=(CW/2, fork_y),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2))

    # ── conv stack block (single block for 4× conv) ───────────────────────────
    cy_conv = cy_lbl - 0.52 - BH/2
    block(ax, CX_L, cy_conv, COL_W, BH, '4× Conv 3×3 + ReLU', GREEN,   text_size=8.5)
    block(ax, CX_R, cy_conv, COL_W, BH, '4× Conv 3×3 + ReLU', BLUE,    text_size=8.5)

    arrow(ax, CX_L, cy_conv - BH/2, cy_conv - BH/2 - GAP)
    arrow(ax, CX_R, cy_conv - BH/2, cy_conv - BH/2 - GAP)

    # ── predictor block ───────────────────────────────────────────────────────
    cy_pred = cy_conv - BH - GAP
    block(ax, CX_L, cy_pred, COL_W, BH, 'Cls Predictor\nConv 3×3', '#1a8a4a', text_size=8.2)
    block(ax, CX_R, cy_pred, COL_W, BH, 'Reg Predictor\nConv 3×3', '#1a4a8a', text_size=8.2)

    ax.text(CX_L + COL_W/2 + 0.10, cy_pred, '36 scores',
            ha='left', va='center', fontsize=7.5, color='#333333',
            bbox=dict(facecolor='#f4f4f4', edgecolor='#bbbbbb',
                      boxstyle='round,pad=0.18', linewidth=0.7))
    ax.text(CX_R + COL_W/2 + 0.10, cy_pred, '36 offsets',
            ha='left', va='center', fontsize=7.5, color='#333333',
            bbox=dict(facecolor='#f4f4f4', edgecolor='#bbbbbb',
                      boxstyle='round,pad=0.18', linewidth=0.7))

    # ── merge arrows → output ─────────────────────────────────────────────────
    cy_out = cy_pred - BH/2 - GAP * 2.0
    for cx_col in [CX_L, CX_R]:
        ax.annotate('', xy=(CW/2, cy_out + BH/2 + 0.06),
                    xytext=(cx_col, cy_pred - BH/2),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2))

    block(ax, CW/2, cy_out, CW - 0.20, BH, 'NMS  →  Detections', ORANGE, text_size=9)
    ax.text(CW/2 + (CW - 0.20)/2 + 0.10, cy_out, 'class · score · bbox',
            ha='left', va='center', fontsize=7.5, color='#333333',
            bbox=dict(facecolor='#f4f4f4', edgecolor='#bbbbbb',
                      boxstyle='round,pad=0.18', linewidth=0.7))

    ax.set_ylim(cy_out - BH/2 - 0.15, height)
    fig.tight_layout()
    fig.savefig('images/retina_head.png', bbox_inches='tight', dpi=160)
    plt.close(fig)
    print("Saved retina_head.png")


if __name__ == '__main__':
    fig_backbone()
    fig_fpn()
    fig_head()
    print("\nAll module diagrams saved to images/")
