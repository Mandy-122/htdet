# Low GFLOPs Model - Fine-tuning from best model
# Loads epoch_42 weights and adapts FPN/Head with reduced channels
#Best model - 0.38 mAP, 70% GFLPS saved, fine-tuning htdet_gpu.py
_base_ = ['./htdet_gpu_low_gflops.py']

# Load the best trained model
# Backbone weights will be loaded, FPN/Head will be initialized randomly
# (since their dimensions changed from 256 -> 128)
load_from = './work_dirs/low_gflops_init/epoch_42_low_gflops_init.pth'

work_dir = './work_dirs/htdet_low_gflops_128'

# Shorter schedule since backbone is pretrained
runner = dict(
    type='EpochBasedRunner',
    max_epochs=60  # Reduced from 60
)

# Lower LR for fine-tuning (backbone is already trained)
optimizer = dict(
    type='SGD',
    lr=0.0005,  # Half of original (backbone pretrained)
    momentum=0.9,
    weight_decay=0.0001
)

# Shorter warmup
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,  # Reduced from 500
    warmup_ratio=0.01,  # Lower starting point
    step=[30, 36]  # Reduced from [40, 55]
)

# Evaluate frequently
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

checkpoint_config = dict(interval=1)
log_config = dict(interval=10)
