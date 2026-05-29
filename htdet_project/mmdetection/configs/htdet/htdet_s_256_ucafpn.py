# HTDet mobilevit_s + 256-ch UCA-FPN (Underwater Contextual Attention FPN)
#
# Backbone : mobilevit_s  (same as 44.12% teacher)
# Neck     : UCAFPN, out_channels=256, CBAM after each FPN level
# Head     : RetinaHead, feat_channels=256
# Load from: htdet_mobilevit_finetune_stage2/latest.pth (44.12% mAP)
#            — backbone + FPN + head all load; CBAM modules init randomly
#
# Novelty baseline: compare against htdet_teacher_256 (44.12%) with plain FPN
# Expected: > 44.12% mAP with CBAM attention
#
# Launch:
#   PYTHONPATH=. /workspace/ckarfa/anaconda3/envs/mani/bin/python3.9 \
#     tools/train.py configs/htdet/htdet_s_256_ucafpn.py --gpu-id 0

custom_imports = dict(
    imports=['custom_modules.ua_fpn'],
    allow_failed_imports=False)

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(480, 480), (800, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
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
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
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
load_from = 'work_dirs/htdet_mobilevit_finetune_stage2/latest.pth'
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
        type='UCAFPN',
        in_channels=[64, 96, 128, 640],
        out_channels=256,
        num_outs=5,
        cbam_reduction=8,
        spatial_kernel=7),
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
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
        reg_decoded_bbox=True),
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

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[40, 55])
runner = dict(type='EpochBasedRunner', max_epochs=60)
work_dir = 'work_dirs/htdet_s_256_ucafpn'
auto_resume = False
gpu_ids = [0]
