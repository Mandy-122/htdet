_base_ = '../detr/detr_r50_8x2_150e_coco.py'

dataset_type = 'CocoDataset'
data_root = 'data/urpc/'

classes = ('holothurian', 'echinus', 'scallop', 'starfish')

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    bbox_head=dict(
        num_classes=4
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(type='Resize',
                     img_scale=[(480,1333),(512,1333),(544,1333),(576,1333),
                                (608,1333),(640,1333),(672,1333),(704,1333),
                                (736,1333),(768,1333),(800,1333)],
                     multiscale_mode='value',
                     keep_ratio=True)
            ]
        ]
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_bboxes','gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333,800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,

    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_train2018.json',
        img_prefix=data_root + 'train2018/images/',
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2018.json',
        img_prefix=data_root + 'val2018/images/',
        pipeline=test_pipeline
    ),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_val2018.json',
        img_prefix=data_root + 'val2018/images/',
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='bbox')