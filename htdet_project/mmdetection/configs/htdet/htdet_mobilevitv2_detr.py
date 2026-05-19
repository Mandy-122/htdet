# RT-DETR with MobileViT v2 backbone
# Transformer-based detector - no anchors needed
# Expected mAP: 0.44-0.48+ (DETR often outperforms RetinaNet)

_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True

# =============================================================================
# MODEL - RT-DETR (Real-time DETR)
# =============================================================================
model = dict(
    type='DETR',

    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevitv2_100',
        pretrained=True,
        features_only=True,
        out_indices=(2, 3, 4)  # Use last 3 stages
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],  # MobileViT v2 channels
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32)
    ),

    bbox_head=dict(
        type='DETRHead',
        num_classes=4,
        in_channels=256,

        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            ),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                )
            )
        ),

        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5
        ),

        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0
        ),

        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0
        ),

        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0
        )
    ),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, bbox_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
        )
    ),

    test_cfg=dict(
        max_per_img=100
    )
)

# =============================================================================
# DATA PIPELINE
# =============================================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(
        type='Resize',
        img_scale=[(640, 640), (800, 800)],
        multiscale_mode='range',
        keep_ratio=True
    ),

    dict(type='RandomFlip', flip_ratio=0.5),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),  # DETR doesn't need size divisibility
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
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
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Lower LR for backbone
            'norm': dict(decay_mult=0.0)
        }
    )
)

optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2),
    cumulative_updates=1
)

# =============================================================================
# LR SCHEDULE
# =============================================================================
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[40]
)

# =============================================================================
# RUNNER
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

work_dir = './work_dirs/htdet_mobilevitv2_detr'
