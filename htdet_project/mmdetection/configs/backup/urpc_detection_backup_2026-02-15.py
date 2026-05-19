# configs/_base_/datasets/urpc_detection.py

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'

# configs/_base_/datasets/urpc_detection.py

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'

classes = (
    'holothurian',
    'echinus',
    'scallop',
    'starfish'
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,

    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2018.json',
        img_prefix=data_root + 'train2018/images/',
        classes=classes,
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2018.json',
        img_prefix=data_root + 'val2018/images/',
        classes=classes,
        pipeline=test_pipeline
    ),

    test=dict(
        type='CocoDataset',
        ann_file='data/urpc/annotations/instances_val2018.json',
        img_prefix='data/urpc/val2018/images/',
        classes=('echinus','holothurian','scallop','starfish'),
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='bbox')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2018.json',
        img_prefix=data_root + 'train2018/images/',
        classes=classes,
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2018.json',
        img_prefix=data_root + 'val2018/images/',
        classes=classes,
        pipeline=test_pipeline
    ),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2018.json',
        img_prefix=data_root + 'val2018/images/',
        classes=classes,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='bbox')
