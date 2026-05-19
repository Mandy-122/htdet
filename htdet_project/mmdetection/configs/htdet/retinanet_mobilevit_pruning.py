_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True

# -----------------------------
# Model
# -----------------------------
model = dict(
    type='RetinaNet',

    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevit_s',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4)
    ),

    neck=dict(
        type='FPN',
        in_channels=[64, 96, 128, 640],
        out_channels=256,
        num_outs=5
    ),

    bbox_head=dict(
        type='RetinaHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,

        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 6, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),

        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.]
        ),

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),

        loss_bbox=dict(
            type='L1Loss',
            loss_weight=1.0
        )
    )
)

# -----------------------------
# TRAIN / TEST CFG (CORRECT PLACE)
# -----------------------------
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1
    ),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100
)

# -----------------------------
# Load pruned weights
# -----------------------------
#load_from = 'work_dirs/pruned_model_epoch42.pth'
load_from = 'work_dirs/slimmed_model.pth'

# -----------------------------
# Dataloader
# -----------------------------
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# -----------------------------
# Optimizer (fine-tuning)
# -----------------------------
optimizer = dict(
    type='SGD',
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2)
)

# -----------------------------
# LR Schedule
# -----------------------------
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    step=[10, 20]
)

# -----------------------------
# Training
# -----------------------------
runner = dict(
    type='EpochBasedRunner',
    max_epochs=30
)

evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1)
log_config = dict(interval=20)

work_dir = './work_dirs/pruned_mobilevit_retinanet'