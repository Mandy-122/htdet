# YOLOX-s — URPC 4-class adaptation
#
# Architecture: anchor-free, CSPDarknet + YOLOXPAFPN + YOLOXHead
# Augmentation: Mosaic + MixUp (YOLOX-style, disabled last 15 epochs)
# Recipe  : 60 epochs (original COCO=300 scaled down), warmup 5 ep, batch=8
#           img_scale=640, no ImageNet normalization (YOLOX uses 0-255 scale)
#
# State-of-the-art comparison baseline for paper.
# YOLOX-s COCO mAP: 40.5% @ 26.8 GFLOPs. Expect URPC ~40-46% range.

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'
classes = ('holothurian', 'echinus', 'scallop', 'starfish')
img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_train2018.json',
        img_prefix='data/urpc/train2018/images/',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=classes,
        pipeline=test_pipeline))

# Disable Mosaic augmentation for last 15 epochs (standard YOLOX practice)
custom_hooks = [
    dict(
        type='NumClassCheckHook'),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=15,
        priority=48)
]

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cudnn_benchmark = True

model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=4,
        in_channels=128,
        feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# YOLOX uses SGD with nesterov and momentum warm-up
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)

# Cosine LR with 5-epoch warm-up (standard YOLOX schedule)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=1,
    num_last_epochs=15,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=60)
work_dir = 'work_dirs/htdet_yolox_s_urpc'
auto_resume = False
gpu_ids = [0]
