# htdet_gpu.py + White Balance correction preprocessing
#
# Problem: URPC images have strong color cast — G channel is +61 above R,
# B channel is +43 above R. This is classic underwater light absorption:
# red wavelengths are absorbed first, leaving green-blue dominant scenes.
# The backbone's ImageNet-pretrained features expect balanced RGB.
#
# Fix: RandomToneCurve (augmentation-style white balance) + channel-wise
# normalization tuned to the actual URPC channel means instead of ImageNet.
#
# URPC channel means (BGR): B=125.5, G=143.5, R=82.1
# URPC channel stds  (BGR): B=31.7,  G=34.5,  R=36.9
#
# Using dataset-specific normalization so the backbone sees a distribution
# closer to what it was pretrained on (balanced channels).

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

# URPC-specific normalization (RGB order): mean and std computed from dataset
urpc_mean = [82.1, 143.5, 125.5]   # R, G, B
urpc_std  = [36.9, 34.5, 31.7]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=[
            # Randomly shift tone curves per channel — simulates white balance
            # variation and forces backbone to be color-cast robust
            dict(type='RandomToneCurve', scale=0.1, p=0.5),
            # Mild channel shuffle to reduce reliance on any single color
            dict(type='ChannelShuffle', p=0.1),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=urpc_mean,
        std=urpc_std,
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=urpc_mean,
                std=urpc_std,
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_train2018.json',
        img_prefix='data/urpc/train2018/images/',
        classes=('holothurian', 'echinus', 'scallop', 'starfish'),
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=('holothurian', 'echinus', 'scallop', 'starfish'),
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=('holothurian', 'echinus', 'scallop', 'starfish'),
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cudnn_benchmark = True

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevit_s',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4)),
    neck=dict(
        type='FPN',
        in_channels=[64, 96, 128, 640],
        out_channels=256,
        num_outs=5),
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
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[40, 55])
runner = dict(type='EpochBasedRunner', max_epochs=60)
work_dir = 'work_dirs/htdet_gpu_whitebal'
auto_resume = False
gpu_ids = [0]
