"""
Generate all figures for the MTP2 thesis report.
Run from MTP2_Mani_Deep_G/ directory:
    python3 generate_figures.py
All figures are saved to ./images/
"""

import re
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("images", exist_ok=True)

STYLE = {
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
plt.rcParams.update(STYLE)

WORK = "../work_dirs"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse epoch + mAP from a log file
# ─────────────────────────────────────────────────────────────────────────────
def parse_map_curve(log_path):
    epochs, mAP, mAP50 = [], [], []
    try:
        with open(log_path) as f:
            for line in f:
                if "Epoch(val)" not in line or "bbox_mAP:" not in line:
                    continue
                ep = re.search(r'Epoch\(val\) \[(\d+)\]', line)
                m  = re.search(r'bbox_mAP: ([\d.]+)', line)
                m50= re.search(r'bbox_mAP_50: ([\d.]+)', line)
                if ep and m and m50:
                    epochs.append(int(ep.group(1)))
                    mAP.append(float(m.group(1)))
                    mAP50.append(float(m50.group(1)))
    except FileNotFoundError:
        pass
    return epochs, mAP, mAP50


def best_log(run):
    """Return the most informative log file for a run directory."""
    d = f"{WORK}/{run}"
    stdout = f"{d}/train_stdout.log"
    if os.path.exists(stdout):
        return stdout
    logs = sorted(f for f in os.listdir(d) if f.endswith('.log') and not f.endswith('.json'))
    return f"{d}/{logs[-1]}" if logs else None


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Baseline training curve (mAP50 vs epoch)
# ─────────────────────────────────────────────────────────────────────────────
def fig_baseline_curve():
    log = f"{WORK}/htdet_mobilevit_April21st_2/20260421_170653.log"
    eps, _, m50 = parse_map_curve(log)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps, [v * 100 for v in m50], color='steelblue', linewidth=1.8, label='mAP$_{50}$ (%)')
    ax.axhline(max(m50) * 100, color='steelblue', linestyle='--', linewidth=1,
               alpha=0.6, label=f'Best = {max(m50)*100:.1f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('Baseline HTDet Training Curve\n(MobileViT-S + FPN, 640×640)')
    ax.set_ylim(60, 80)
    ax.legend()
    fig.tight_layout()
    fig.savefig('images/fig_baseline_curve.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_baseline_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Backbone comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_backbone_comparison():
    models = [
        'MobileViT-S\n(pretrained)',
        'MobileViTv2-150\n(scratch)',
        'EfficientNet-B3\n(scratch)',
        'ConvNeXt-Tiny\n(scratch)',
    ]
    mAP50 = [75.5, 76.1, 72.0, 65.2]
    mAP   = [40.8, 41.3, 38.3, 33.5]
    colors = ['steelblue', 'steelblue', 'darkorange', 'forestgreen']
    # MobileViT variants use same color, CNN variants use different

    x = np.arange(len(models))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, mAP50, w, label='mAP$_{50}$ (%)', color=['#2878b5', '#2878b5', '#e0a234', '#4cb87a'], alpha=0.85)
    b2 = ax.bar(x + w/2, mAP,   w, label='mAP (%)',        color=['#2878b5', '#2878b5', '#e0a234', '#4cb87a'], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel('AP (%)')
    ax.set_title('Backbone Comparison on URPC\n(Standard FPN Neck, 60 Epochs)')
    ax.set_ylim(0, 90)
    ax.legend()
    for rect in b1:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.5,
                f'{rect.get_height():.1f}', ha='center', va='bottom', fontsize=8.5)
    for rect in b2:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.5,
                f'{rect.get_height():.1f}', ha='center', va='bottom', fontsize=8.5)
    fig.tight_layout()
    fig.savefig('images/fig_backbone_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_backbone_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Multi-model training curves comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_curves():
    runs = {
        'MobileViT-S + FPN (baseline)': (
            f"{WORK}/htdet_mobilevit_April21st_2/20260421_170653.log", 'steelblue'),
        'MobileViT-S + FA-FPN (best)': (
            best_log('htdet_s_256_fafpn'), 'crimson'),
        'MobileViTv2-150 + FPN (scratch)': (
            best_log('htdet_mobilevitv2_150_scratch'), 'darkorange'),
        'EfficientNet-B3 + FPN (scratch)': (
            best_log('htdet_effb3_scratch'), 'forestgreen'),
        'ConvNeXt-Tiny + FPN (scratch)': (
            best_log('htdet_convnext_scratch'), 'purple'),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (path, color) in runs.items():
        if path is None:
            continue
        eps, _, m50 = parse_map_curve(path)
        if eps:
            ax.plot(eps, [v * 100 for v in m50], color=color, linewidth=1.6, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('mAP$_{50}$ Training Curves — All Architectures')
    ax.set_ylim(30, 85)
    ax.legend(fontsize=8.5, loc='lower right')
    fig.tight_layout()
    fig.savefig('images/fig_model_curves.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_model_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — FA-FPN vs FPN + KD compression comparison (bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_neck_and_compression():
    labels = [
        'Baseline\nFPN (640)',
        'FA-FPN\n(256)',
        'Pruned\n+FT (640)',
        'Compact\n192 (no KD)',
        'Compact\n192 + KD',
    ]
    m50  = [75.5, 77.6, 75.3, 67.6, 71.8]
    gflops = [198.9, 198.9, 95.0, 59.9, 59.9]

    colors = ['#2878b5', '#e84141', '#e0a234', '#a0a0a0', '#4cb87a']
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x - 0.2, m50, 0.38, color=colors, alpha=0.85, label='mAP$_{50}$ (%)', zorder=3)
    line, = ax2.plot(x, gflops, 'k--o', linewidth=1.6, markersize=6, label='GFLOPs', zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel('mAP$_{50}$ (%)')
    ax1.set_ylim(50, 90)
    ax2.set_ylabel('GFLOPs')
    ax2.set_ylim(0, 280)

    for rect, val in zip(bars, m50):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8.5)

    ax1.set_title('Accuracy vs. Efficiency: Architecture and Compression Variants')
    lines = [bars, line]
    labels2 = ['mAP$_{50}$ (%)', 'GFLOPs']
    ax1.legend([mpatches.Patch(color='grey'), line], labels2, loc='lower left')
    fig.tight_layout()
    fig.savefig('images/fig_accuracy_efficiency.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_accuracy_efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — KD benefit curve: KD vs no-KD (192 model, epoch-wise)
# ─────────────────────────────────────────────────────────────────────────────
def fig_kd_comparison():
    kd_log  = best_log('htdet_xs_192_kd')
    nokd_log = best_log('htdet_xs_192')

    eps_kd,  _, m50_kd  = parse_map_curve(kd_log)
    eps_nkd, _, m50_nkd = parse_map_curve(nokd_log)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(eps_kd,  [v*100 for v in m50_kd],  color='crimson',   linewidth=1.8, label='MobileViT-XS 192 + KD')
    ax.plot(eps_nkd, [v*100 for v in m50_nkd], color='steelblue', linewidth=1.8, linestyle='--', label='MobileViT-XS 192 (no KD)')
    ax.fill_between(eps_kd,
                    [v*100 for v in m50_nkd[:len(eps_kd)]],
                    [v*100 for v in m50_kd],
                    alpha=0.15, color='crimson')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('Effect of Knowledge Distillation\n(192×192 Compact Model)')
    ax.set_ylim(50, 80)
    ax.legend()
    fig.tight_layout()
    fig.savefig('images/fig_kd_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_kd_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Channel pruning fine-tune recovery curve
# ─────────────────────────────────────────────────────────────────────────────
def fig_pruning_recovery():
    log = best_log('htdet_pruned_finetune')
    eps, _, m50 = parse_map_curve(log)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eps, [v*100 for v in m50], color='darkorange', linewidth=1.8, label='Pruned model (fine-tune)')
    ax.axhline(75.5, color='steelblue', linestyle='--', linewidth=1.4, label='Baseline (no pruning): 75.5%')
    ax.axhline(max(m50)*100, color='darkorange', linestyle=':', linewidth=1.2,
               label=f'Best pruned: {max(m50)*100:.1f}%')
    ax.set_xlabel('Fine-Tune Epoch')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('Structured Pruning Recovery\n(30% Channel Pruning → Fine-Tune, 640×640)')
    ax.set_ylim(55, 80)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('images/fig_pruning_recovery.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_pruning_recovery.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — PTQ SQNR distribution across layers (histogram)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ptq_sqnr():
    # Values from the PTQ report (all 53 finite SQNR entries)
    sqnr_values = [
        44.8, 43.8, 47.9, 43.3, 44.6, 48.2, 42.8, 44.1, 46.7, 42.9,
        44.3, 47.6, 43.1, 44.0, 49.2, 42.6, 36.8, 43.8, 43.2, 36.8,
        40.9, 49.0, 42.6, 38.4, 43.8, 43.3, 36.9, 42.7, 48.4, 42.6,
        39.0, 43.1, 43.0, 38.0, 42.1, 48.2, 48.1, 47.8, 47.2, 47.2,
        47.2, 47.3, 40.6, 40.9,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    ax = axes[0]
    ax.hist(sqnr_values, bins=12, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(sqnr_values), color='crimson', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(sqnr_values):.1f} dB')
    ax.axvline(35, color='darkorange', linestyle=':', linewidth=1.4,
               label='Near-lossless threshold (35 dB)')
    ax.set_xlabel('SQNR (dB)')
    ax.set_ylabel('Number of Layers')
    ax.set_title('W8A32 PTQ: SQNR Distribution\n(53 Conv2d Layers)')
    ax.legend(fontsize=9)

    # Layer-wise SQNR bar chart (grouped by component)
    ax = axes[1]
    layer_labels = [
        'Stem', 'Stage0', 'Stage1\n(3L)', 'Stage2\n(3L)', 'Stage3\n(ViT)', 'Stage4\n(ViT)',
        'Final\nConv', 'Neck\nlat (4L)', 'Neck\nfpn (4L)', 'Head\ncls (4L)', 'Head\nreg (4L)',
        'Pred\ncls', 'Pred\nreg',
    ]
    # representative mean SQNR per group
    group_sqnr = [44.8, 43.3, 44.5, 43.2, 38.5, 40.2, 42.1, 47.8, 47.3, float('inf'), float('inf'), 40.6, 40.9]
    group_sqnr_plot = [min(v, 52) for v in group_sqnr]
    inf_idx = [i for i, v in enumerate(group_sqnr) if v == float('inf')]

    colors2 = ['#2878b5'] * 7 + ['#e0a234'] * 4 + ['#4cb87a'] * 2
    bars = ax.bar(range(len(layer_labels)), group_sqnr_plot, color=colors2, alpha=0.85)
    ax.axhline(35, color='darkorange', linestyle=':', linewidth=1.4, label='Near-lossless (35 dB)')
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=7.5, rotation=15)
    ax.set_ylabel('SQNR (dB)')
    ax.set_title('W8A32 SQNR by Layer Group\n(Blue=Backbone, Orange=Neck, Green=Head)')
    ax.set_ylim(0, 58)
    for i in inf_idx:
        ax.text(i, group_sqnr_plot[i] + 0.5, '∞', ha='center', va='bottom', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('images/fig_ptq_sqnr.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_ptq_sqnr.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Detection count comparison Float32 vs W8A32
# ─────────────────────────────────────────────────────────────────────────────
def fig_ptq_detection():
    images  = ['Img 0', 'Img 1', 'Img 2', 'Img 3', 'Img 4']
    float32 = [12, 10, 6, 4, 13]
    w8a32   = [13, 10, 8, 5, 13]

    x = np.arange(len(images))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, float32, w, label='Float32', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + w/2, w8a32,   w, label='W8A32 (PTQ)', color='crimson', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(images)
    ax.set_ylabel('Number of Detections (score ≥ 0.20)')
    ax.set_title('Float32 vs W8A32 Detection Count\n(5 Test Images, score ≥ 0.20)')
    ax.legend()
    ax.set_ylim(0, 18)

    for rect in [*b1, *b2]:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.15,
                str(int(rect.get_height())), ha='center', va='bottom', fontsize=9)

    ax.text(0.98, 0.96, 'Total: 45 → 49 (+8.9%)', transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    fig.tight_layout()
    fig.savefig('images/fig_ptq_detection.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_ptq_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — Compression vs accuracy scatter (Params vs mAP50)
# ─────────────────────────────────────────────────────────────────────────────
def fig_compression_scatter():
    configs = {
        'Baseline\n(FPN 640)':        (12.4, 75.5, 'steelblue'),
        'FA-FPN\n(256)':              (12.9, 77.6, 'crimson'),
        'MobileViTv2-150':            (14.0, 76.1, 'darkorange'),
        'EfficientNet-B3':            (12.0, 72.0, 'forestgreen'),
        'ConvNeXt-Tiny':              (11.0, 65.2, 'purple'),
        'Pruned+FT\n(30%)':           (5.7,  75.3, 'steelblue'),
        'Compact 192\n(no KD)':       (6.9,  67.6, 'gray'),
        'Compact 192\n+ KD':          (6.9,  71.8, 'darkcyan'),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (params, m50, color) in configs.items():
        ax.scatter(params, m50, s=120, color=color, zorder=5)
        ax.annotate(label, (params, m50), textcoords='offset points',
                    xytext=(6, 3), fontsize=8, color=color)

    ax.set_xlabel('Model Parameters (M)')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('Compression vs Accuracy Trade-off\n(All HTDet Configurations)')
    ax.set_xlim(2, 17)
    ax.set_ylim(58, 82)
    fig.tight_layout()
    fig.savefig('images/fig_compression_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_compression_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Train loss curve for the baseline run
# ─────────────────────────────────────────────────────────────────────────────
def fig_train_loss():
    json_log = f"{WORK}/htdet_mobilevit_April21st_2/20260421_170653.log.json"
    iterations, losses = [], []
    total_iter = 0
    try:
        with open(json_log) as f:
            for line in f:
                d = json.loads(line.strip())
                if d.get('mode') == 'train' and 'loss' in d:
                    total_iter += 1
                    if total_iter % 4 == 0:  # subsample every 4 iters
                        iterations.append(total_iter)
                        losses.append(d['loss'])
    except Exception as e:
        print(f"Could not parse json log: {e}")
        return

    # Smooth with a rolling mean
    window = 20
    losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    iters_smooth = iterations[window-1:]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, losses, color='steelblue', alpha=0.3, linewidth=0.8, label='Raw loss')
    ax.plot(iters_smooth, losses_smooth, color='steelblue', linewidth=1.8, label='Smoothed (w=20)')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss Curve\n(MobileViT-S + FPN, 60 Epochs)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('images/fig_train_loss.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_train_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — FA-FPN variants comparison (v1, v2, v3, scratch)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fafpn_variants():
    runs = {
        'FA-FPN (pretrained, best)': (best_log('htdet_s_256_fafpn'), 'crimson', '-'),
        'FA-FPN v2':                 (best_log('htdet_s_256_fafpn_v2'), 'darkorange', '--'),
        'FA-FPN v3':                 (best_log('htdet_s_256_fafpn_v3'), 'purple', '-.'),
        'FA-FPN (scratch)':          (best_log('htdet_s_256_fafpn_scratch'), 'gray', ':'),
        'Baseline FPN (pretrained)': (f"{WORK}/htdet_mobilevit_April21st_2/20260421_170653.log", 'steelblue', '-'),
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, (path, color, ls) in runs.items():
        if path is None:
            continue
        eps, _, m50 = parse_map_curve(path)
        if eps:
            ax.plot(eps, [v*100 for v in m50], color=color, linestyle=ls,
                    linewidth=1.6, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP$_{50}$ (%)')
    ax.set_title('FA-FPN Variants vs Baseline FPN')
    ax.set_ylim(60, 82)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig('images/fig_fafpn_variants.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved fig_fafpn_variants.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig_baseline_curve()
    fig_backbone_comparison()
    fig_model_curves()
    fig_neck_and_compression()
    fig_kd_comparison()
    fig_pruning_recovery()
    fig_ptq_sqnr()
    fig_ptq_detection()
    fig_compression_scatter()
    fig_train_loss()
    fig_fafpn_variants()
    print("\nAll figures saved to images/")
