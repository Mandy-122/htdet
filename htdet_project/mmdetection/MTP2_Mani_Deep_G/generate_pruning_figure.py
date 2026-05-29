"""Generate fig_pruning_recovery.png — mAP50 recovery during pruning fine-tune."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Actual per-epoch mAP50 from htdet_pruned_finetune log (24 epochs)
map50_finetune = [
    0.7527, 0.7548, 0.7545, 0.7541, 0.7563,
    0.7533, 0.7558, 0.7540, 0.7592, 0.7522,
    0.7574, 0.7533, 0.7577, 0.7584, 0.7517,
    0.7545, 0.7534, 0.7509, 0.7504, 0.7503,
    0.7533, 0.7532, 0.7532, 0.7618,
]
epochs = list(range(1, len(map50_finetune) + 1))
baseline_map50 = 0.7634

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(epochs, [v * 100 for v in map50_finetune],
        color='#e84141', linewidth=2, marker='o', markersize=4, label='Pruned model (fine-tuning)')
ax.axhline(baseline_map50 * 100, color='#2878b5', linestyle='--', linewidth=1.5,
           label=f'Baseline mAP$_{{50}}$ = {baseline_map50*100:.1f}%')
ax.axhline(max(map50_finetune) * 100, color='#4cb87a', linestyle=':', linewidth=1.2,
           label=f'Best pruned = {max(map50_finetune)*100:.1f}%')

ax.set_xlabel('Fine-tuning Epoch')
ax.set_ylabel('mAP$_{50}$ (%)')
ax.set_title('Weight Pruning (30%) — mAP$_{50}$ Recovery During Fine-tuning', fontweight='bold')
ax.set_xlim(0.5, len(epochs) + 0.5)
ax.set_ylim(74.5, 77.0)
ax.legend(fontsize=9)

# Annotate the final value
ax.annotate(f'{max(map50_finetune)*100:.2f}%',
            xy=(epochs[map50_finetune.index(max(map50_finetune))], max(map50_finetune) * 100),
            xytext=(15, 6), textcoords='offset points',
            fontsize=9, color='#4cb87a',
            arrowprops=dict(arrowstyle='->', color='#4cb87a', lw=1.2))

fig.tight_layout()
fig.savefig('images/fig_pruning_recovery.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved fig_pruning_recovery.png")
