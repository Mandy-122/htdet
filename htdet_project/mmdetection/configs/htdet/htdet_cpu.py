# configs/htdet/htdet_cpu.py

_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]




# -----------------------------
# Model
# -----------------------------
model = dict(
    type='RetinaNet',

    backbone=dict(
        type='HTDetMobileViT',
        width=64
    ),

    neck=dict(
    type='FGFPN',   # <-- HTDet neck
    in_channels=[64, 128, 256, 512, 1024],
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
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
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

    # -----------------------------
    # IMPORTANT: Train/Test Config
    # -----------------------------
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


# -----------------------------
# Dataloader (CPU)
# -----------------------------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,

    train=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_train2018.json',
        img_prefix='data/urpc/train2018/images/',
        classes=('echinus', 'holothurian', 'scallop', 'starfish'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(416, 416), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18
            ),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    ),

    val=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=('echinus', 'holothurian', 'scallop', 'starfish'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor',
                         keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    ),

    test=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=('echinus', 'holothurian', 'scallop', 'starfish'),
        pipeline=None
    )
)



# -----------------------------
# Optimizer
# -----------------------------
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)



optimizer_config = dict(grad_clip=None)


# -----------------------------
# Training
# -----------------------------
runner = dict(
    type='EpochBasedRunner',
    max_epochs=24
)

log_config = dict(interval=20)

evaluation = dict(interval=1, metric='bbox')


# -----------------------------
# Checkpoint
# -----------------------------
checkpoint_config = dict(interval=1)
# -----------------------------
# LR Schedule (HTDet style)
# -----------------------------
# -----------------------------
# LR Schedule (Paper Setting)
# -----------------------------
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.000666,
    step=[16, 22]
)


