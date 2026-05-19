# MobileViT v2 + RetinaNet (Improved Config)
# Uses MobileViT v2 instead of v1 for better accuracy-efficiency tradeoff
# Expected mAP: 0.43-0.45+ (vs current 0.414)

_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True

# =============================================================================
# MODEL - MobileViT v2 instead of MobileViT v1
# =============================================================================
model = dict(
    type='RetinaNet',

    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevitv2_100',  # v2 with 1.0 width - better than mobilevit_s
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4)
    ),

    neck=dict(
        type='FPN',
        in_channels=[96, 128, 192, 768],  # MobileViT v2 feature channels
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
    ),

    train_cfg=dict(
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
    ),

    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# =============================================================================
# DATA PIPELINE - Multi-scale + Augmentations
# =============================================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    # Multi-scale training (640-800) - KEY for small objects
    dict(
        type='Resize',
        img_scale=[(640, 640), (800, 800)],
        multiscale_mode='range',
        keep_ratio=True
    ),

    dict(type='RandomFlip', flip_ratio=0.5),

    # Mosaic augmentation (optional - uncomment for more augmentation)
    # dict(
    #     type='Mosaic',
    #     img_scale=(800, 800),
    #     pad_val=114.0
    # ),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),  # Higher eval resolution
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,

    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

# =============================================================================
# OPTIMIZER
# =============================================================================
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2)
)

# =============================================================================
# LR SCHEDULE - Cosine for better convergence
# =============================================================================
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr=1e-5
)

# =============================================================================
# RUNNER - Early stopping friendly
# =============================================================================
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50
)

# =============================================================================
# EVAL & LOGGING
# =============================================================================
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP'
)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook')]
)

# =============================================================================
# MISC
# =============================================================================
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'

auto_scale_lr = dict(enable=False, base_batch_size=16)

work_dir = './work_dirs/htdet_mobilevitv2_retina'
