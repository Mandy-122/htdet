"""Generate all figures for Chapter 4 (Additional Experiments)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Data ─────────────────────────────────────────────────────────────────────

BASELINE_MAP50 = 0.7634  # MobileViT-S + FPN 256ch (ch3 reference)
BASELINE_MAP   = 0.4124

SELECTED_MAP50 = 0.7287  # MobileViT-S + FPN 192ch (selected for FPGA)
SELECTED_MAP   = 0.4072  # ch3 table value at 256→192 channel reduction

models = [
    ('Baseline\n(S + FPN-256)',     0.4124, 0.7634, '#888888'),
    ('FA-FPN\n(S + FPN-256)',       0.4269, 0.7770, '#4c9be8'),
    ('UCA-FPN\n(S + FPN-256)',      0.4293, 0.7788, '#4c9be8'),
    ('White Balance\n(S + FPN-256)',0.4118, 0.7638, '#e8a74c'),
    ('MobileViT-XS\n(XS + FPN-192)',0.3321, 0.6865, '#e84c4c'),
    ('XS + KD\n(XS + FPN-192)',     0.3686, 0.7202, '#e87c4c'),
    ('EfficientNet-B3\n(B3 + FPN-256)', 0.3869, 0.7272, '#e84c4c'),
    ('Selected\n(S + FPN-192)',      0.4072, 0.7287, '#2caa56'),
]

# ── Figure 1: Model Comparison Bar Chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

names    = [m[0] for m in models]
maps     = [m[1] for m in models]
map50s   = [m[2] for m in models]
colors   = [m[3] for m in models]
y_pos    = np.arange(len(models))

# Left: mAP (COCO 50:95)
ax = axes[0]
bars = ax.barh(y_pos, maps, color=colors, alpha=0.85, edgecolor='white', height=0.6)
ax.axvline(BASELINE_MAP, color='#888888', linestyle='--', lw=1.2, alpha=0.7, label=f'Baseline ({BASELINE_MAP:.4f})')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('mAP (COCO 50:95)')
ax.set_title('COCO mAP (50:95)\nAll Explored Models', fontweight='bold')
ax.set_xlim(0.30, 0.46)
for bar, v in zip(bars, maps):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
            va='center', fontsize=7.5)
ax.legend(fontsize=8)

# Right: mAP₅₀
ax = axes[1]
bars = ax.barh(y_pos, map50s, color=colors, alpha=0.85, edgecolor='white', height=0.6)
ax.axvline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.2, alpha=0.7, label=f'Baseline ({BASELINE_MAP50:.4f})')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel('mAP$_{50}$')
ax.set_title('COCO mAP$_{50}$\nAll Explored Models', fontweight='bold')
ax.set_xlim(0.62, 0.82)
for bar, v in zip(bars, map50s):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
            va='center', fontsize=7.5)
ax.legend(fontsize=8)

# Legend patches
legend_patches = [
    mpatches.Patch(color='#888888', label='Baseline (reference)'),
    mpatches.Patch(color='#4c9be8', label='Attention neck variants'),
    mpatches.Patch(color='#e8a74c', label='Augmentation study'),
    mpatches.Patch(color='#e84c4c', label='Alternative backbone (below baseline)'),
    mpatches.Patch(color='#e87c4c', label='XS + Knowledge Distillation'),
    mpatches.Patch(color='#2caa56', label='Selected for FPGA'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.07), frameon=True)

fig.suptitle('HTDet: All Explored Model Variants vs Baseline', fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(IMG_DIR, 'fig_ch4_model_comparison.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")


# ── Figure 2: Neck Variant Training Curves ────────────────────────────────────
# FA-FPN and UCA-FPN were fine-tuned from the pretrained baseline checkpoint.
fafpn_curve = [
    0.7274,0.7601,0.7681,0.7663,0.7653,0.7719,0.7706,0.7736,0.7715,0.7680,
    0.7710,0.7777,0.7721,0.7745,0.7756,0.7708,0.7721,0.7742,0.7743,0.7730,
    0.7768,0.7698,0.7772,0.7699,0.7736,0.7720,0.7763,0.7753,0.7765,0.7718,
    0.7770,0.7683,0.7702,0.7697,0.7713,0.7693,0.7662,0.7712,0.7701,0.7717,
    0.7756,0.7688,0.7698,0.7736,0.7743,0.7763,0.7740,
]
ucafpn_curve = [
    0.7130,0.7516,0.7606,0.7610,0.7630,0.7672,0.7671,0.7688,0.7737,0.7722,
    0.7675,0.7693,0.7671,0.7702,0.7724,0.7766,0.7691,0.7701,0.7755,0.7730,
    0.7718,0.7733,0.7717,0.7716,0.7743,0.7751,0.7733,0.7672,0.7758,0.7676,
    0.7659,0.7676,0.7788,0.7696,0.7626,0.7703,0.7661,0.7687,0.7665,0.7716,
    0.7667,0.7726,0.7694,0.7666,0.7687,0.7706,0.7698,
]

fig, ax = plt.subplots(figsize=(9, 4.5))
epochs_47 = np.arange(1, 48)
ax.plot(epochs_47, fafpn_curve,  color='#4c9be8', lw=1.8, label='FA-FPN (best: 0.7770 @ ep31)')
ax.plot(epochs_47, ucafpn_curve, color='#9b4ce8', lw=1.8, label='UCA-FPN (best: 0.7788 @ ep33)')
ax.axhline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.5,
           label=f'Baseline (S+FPN-256): {BASELINE_MAP50}')
ax.axhline(SELECTED_MAP50, color='#2caa56', linestyle=':', lw=1.5,
           label=f'Selected (S+FPN-192): {SELECTED_MAP50}')

# Mark best points
best_fa  = np.argmax(fafpn_curve)
best_uca = np.argmax(ucafpn_curve)
ax.scatter(best_fa+1,  fafpn_curve[best_fa],  color='#4c9be8', s=60, zorder=5)
ax.scatter(best_uca+1, ucafpn_curve[best_uca], color='#9b4ce8', s=60, zorder=5)

ax.set_xlabel('Epoch')
ax.set_ylabel('mAP$_{50}$')
ax.set_title('Attention Neck Variants: FA-FPN and UCA-FPN vs Baseline\n'
             '(fine-tuned from pretrained MobileViT-S + FPN-256 checkpoint)', fontweight='bold')
ax.set_ylim(0.70, 0.80)
ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(IMG_DIR, 'fig_ch4_neck_curves.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")


# ── Figure 3: Backbone Variant Curves ─────────────────────────────────────────
xs_curve = [
    0.0076,0.1016,0.2156,0.3153,0.3585,0.3969,0.4242,0.4442,0.4322,0.4625,
    0.4914,0.5240,0.5205,0.5410,0.5521,0.5454,0.5842,0.5847,0.5858,0.5856,
    0.6126,0.6019,0.6129,0.5885,0.6216,0.6116,0.6263,0.6243,0.6391,0.6459,
    0.6270,0.6278,0.6539,0.6376,0.6498,0.6442,0.6508,0.6672,0.6550,0.6663,
    0.6739,0.6626,0.6714,0.6762,0.6789,0.6725,0.6753,0.6770,0.6774,0.6733,
    0.6775,0.6827,0.6717,0.6772,0.6733,0.6761,0.6739,0.6795,0.6865,0.6762,
]
xs_kd_curve = [
    0.6764,0.7006,0.7097,0.6953,0.7046,0.7051,0.6852,0.7019,0.7054,0.7109,
    0.7015,0.7004,0.7020,0.7090,0.7073,0.6992,0.7088,0.7199,0.7221,0.7011,
    0.7100,0.7053,0.7147,0.6998,0.6966,0.7145,0.7218,0.6968,0.7022,0.6963,
    0.6959,0.6962,0.7073,0.7154,0.7152,0.7050,0.7105,0.7017,0.7016,0.6990,
    0.7165,0.7202,0.7062,0.7033,0.7074,0.7121,0.7050,0.7118,0.7062,0.7072,
    0.7040,0.7049,0.7103,0.7038,0.7022,0.7101,0.7144,0.7069,0.7053,0.7181,
]
effb3_curve = [
    0.3927,0.5832,0.6329,0.6874,0.6906,0.7094,0.7227,0.7349,0.7252,0.7318,
    0.7356,0.7317,0.7134,0.7284,0.7290,0.7248,0.7280,0.7357,0.7310,0.7273,
    0.7319,0.7259,0.7277,0.7197,0.7258,0.7244,0.7291,0.7191,0.7226,0.7183,
    0.7273,0.7286,0.7289,0.7279,0.7265,0.7279,0.7286,0.7300,0.7268,0.7228,
    0.7292,0.7232,0.7274,0.7272,0.7268,0.7265,0.7205,0.7205,0.7221,0.7270,
    0.7218,0.7241,0.7236,0.7214,0.7236,0.7220,0.7203,0.7226,0.7191,0.7203,
]
epochs_60 = np.arange(1, 61)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: backbone comparison (XS, XS+KD)
ax = axes[0]
ax.plot(epochs_60, xs_curve,    color='#e84c4c', lw=1.8, label=f'MobileViT-XS + FPN-192 (best: 0.6865)')
ax.plot(epochs_60, xs_kd_curve, color='#e87c4c', lw=1.8, label=f'MobileViT-XS + KD (best: 0.7202)')
ax.axhline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.5,
           label=f'Baseline (S+FPN-256): {BASELINE_MAP50}')
ax.axhline(SELECTED_MAP50, color='#2caa56', linestyle=':', lw=1.5,
           label=f'Selected (S+FPN-192): {SELECTED_MAP50}')

kd_delta = max(xs_kd_curve) - max(xs_curve)
ax.annotate(f'+{kd_delta:.4f} from KD',
            xy=(np.argmax(xs_kd_curve)+1, max(xs_kd_curve)),
            xytext=(45, 0.71),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
            fontsize=8, color='#e87c4c')

ax.set_xlabel('Epoch')
ax.set_ylabel('mAP$_{50}$')
ax.set_title('MobileViT-XS: Standalone vs Knowledge Distillation\n(teacher: MobileViT-S 76.34%)', fontweight='bold')
ax.set_ylim(0.0, 0.80)
ax.legend(fontsize=8.5)

# Right: EfficientNet-B3 vs Baseline
ax = axes[1]
ax.plot(epochs_60, effb3_curve, color='#b84ce8', lw=1.8, label=f'EfficientNet-B3 + FPN-256 (best: 0.7272)')
ax.axhline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.5,
           label=f'Baseline MobileViT-S (0.7634)')
ax.axhline(SELECTED_MAP50, color='#2caa56', linestyle=':', lw=1.5,
           label=f'Selected (S+FPN-192): {SELECTED_MAP50}')
ax.set_xlabel('Epoch')
ax.set_ylabel('mAP$_{50}$')
ax.set_title('EfficientNet-B3 + FPN-256\nvs MobileViT-S Baseline', fontweight='bold')
ax.set_ylim(0.34, 0.80)
ax.legend(fontsize=8.5)

fig.suptitle('Alternative Backbone Experiments on URPC', fontsize=12, fontweight='bold')
fig.tight_layout()
out = os.path.join(IMG_DIR, 'fig_ch4_backbone_curves.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")


# ── Figure 4: White Balance Effect ─────────────────────────────────────────
wbal_curve = [
    0.3117,0.4674,0.5453,0.5649,0.6407,0.6722,0.6834,0.6932,0.6754,0.7122,
    0.7026,0.7397,0.7285,0.7233,0.7105,0.7322,0.7360,0.7363,0.7419,0.7413,
    0.7558,0.7532,0.7311,0.7391,0.7491,0.7434,0.7544,0.7441,0.7433,0.7528,
    0.7386,0.7510,0.7532,0.7481,0.7462,0.7489,0.7432,0.7484,0.7285,0.7477,
    0.7649,0.7636,0.7642,0.7638,0.7669,0.7638,0.7650,0.7602,0.7617,0.7593,
    0.7607,0.7631,0.7611,0.7604,0.7606,0.7597,0.7593,0.7601,0.7606,0.7606,
]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(epochs_60, wbal_curve,  color='#e8a74c', lw=1.8, label=f'White Balance + URPC Norm (best: 0.7638)')
ax.axhline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.5,
           label=f'Baseline (standard ImageNet norm): {BASELINE_MAP50}')
ax.axhline(SELECTED_MAP50, color='#2caa56', linestyle=':', lw=1.5,
           label=f'Selected S+FPN-192: {SELECTED_MAP50}')

ax.fill_between(epochs_60, BASELINE_MAP50 - 0.001, BASELINE_MAP50 + 0.001,
                color='#888888', alpha=0.15, label='±0.001 around baseline')

ax.set_xlabel('Epoch')
ax.set_ylabel('mAP$_{50}$')
ax.set_title('White Balance Preprocessing: URPC-Specific Normalisation\n'
             '(URPC channel means: R=82.1, G=143.5, B=125.5 vs ImageNet: R=123.7, G=116.3, B=103.5)',
             fontweight='bold')
ax.set_ylim(0.28, 0.80)
ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(IMG_DIR, 'fig_ch4_whitbal_curve.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")


# ── Figure 5: FPGA Suitability Scatter ───────────────────────────────────────
# Axes: mAP50 (accuracy) vs FPGA implementation cost (qualitative 1-10 scale)
# Implementation cost: 1=easy (standard convs only), 10=very hard (FFT/dynamic pools)

models_scatter = [
    # name, mAP50, fpga_cost, color, marker
    ('Baseline\n(S+FPN-256)',     0.7634, 3, '#888888',  'o'),
    ('FA-FPN',                    0.7770, 9, '#4c9be8',  's'),
    ('UCA-FPN',                   0.7788, 7, '#9b4ce8',  's'),
    ('White Balance',             0.7638, 2, '#e8a74c',  '^'),
    ('MobileViT-XS',              0.6865, 2, '#e84c4c',  'D'),
    ('XS + KD',                   0.7202, 2, '#e87c4c',  'D'),
    ('EfficientNet-B3',           0.7272, 5, '#b84ce8',  'p'),
    ('Selected\n(S+FPN-192)',     0.7287, 3, '#2caa56',  '*'),
]

fig, ax = plt.subplots(figsize=(9, 5.5))

for name, map50, cost, color, marker in models_scatter:
    size = 200 if marker == '*' else 100
    ax.scatter(cost, map50, color=color, marker=marker, s=size, zorder=4,
               edgecolors='white', linewidths=0.8)
    offset_x = 0.15
    offset_y = 0.003
    ax.annotate(name, (cost + offset_x, map50 + offset_y), fontsize=7.5,
                va='bottom', ha='left')

ax.axhline(BASELINE_MAP50, color='#888888', linestyle='--', lw=1.0, alpha=0.5)
ax.set_xlabel('FPGA Implementation Complexity\n(1 = standard convolutions only, 10 = FFT/dynamic operations)', fontsize=9)
ax.set_ylabel('mAP$_{50}$')
ax.set_title('Accuracy vs FPGA Complexity Trade-off\n(Selected model balances accuracy with implementability)',
             fontweight='bold')
ax.set_xlim(0.5, 11)
ax.set_ylim(0.65, 0.80)

# Annotation regions
ax.axvline(4, color='gray', linestyle=':', alpha=0.4, lw=1.0)
ax.text(2.0, 0.793, 'FPGA-Friendly\nZone', fontsize=8, color='green', alpha=0.6, ha='center')
ax.text(7.0, 0.793, 'High HLS Complexity', fontsize=8, color='red', alpha=0.6, ha='center')

# Legend
legend_items = [
    mpatches.Patch(color='#888888', label='Baseline'),
    mpatches.Patch(color='#4c9be8', label='FA-FPN / UCA-FPN (attention neck)'),
    mpatches.Patch(color='#e8a74c', label='White Balance augmentation'),
    mpatches.Patch(color='#e84c4c', label='MobileViT-XS / XS+KD'),
    mpatches.Patch(color='#b84ce8', label='EfficientNet-B3'),
    mpatches.Patch(color='#2caa56', label='Selected (S+FPN-192, W8A8)'),
]
ax.legend(handles=legend_items, fontsize=8, loc='lower right')
fig.tight_layout()
out = os.path.join(IMG_DIR, 'fig_ch4_fpga_tradeoff.png')
fig.savefig(out, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out}")

print("\nAll Chapter 4 figures done.")
