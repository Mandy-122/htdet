# Finetuning config for channel-pruned model
# After channel pruning, the model needs fine-tuning to recover accuracy

#Channel Pruning with weights zeroed out and stay the same during fine-tuning, but gflops won't reduce
#Success till 12 epochs - can continue till end

_base_ = ['./htdet_gpu.py']

# Load the channel-pruned checkpoint
load_from = './work_dirs/pruned_models/epoch_42_channel_pruned_r0.30.pth'

work_dir = './work_dirs/htdet_pruned_finetune'

# Shorter schedule for fine-tuning
runner = dict(
    type='EpochBasedRunner',
    max_epochs=30  # Reduced from 60
)

# Lower learning rate for fine-tuning
optimizer = dict(
    type='SGD',
    lr=0.0001,  # 10x lower than original
    momentum=0.9,
    weight_decay=0.0001
)

# Shorter warmup for fine-tuning
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,  # Reduced from 500
    warmup_ratio=0.01,  # Lower warmup ratio
    step=[20, 26]  # Reduced from [40, 55]
)

# Evaluate more frequently
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

# Save checkpoints
checkpoint_config = dict(interval=1)

# Log more frequently to monitor recovery
log_config = dict(interval=10)
