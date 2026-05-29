# HTDet 192-ch KD training: mobilevit_s + FPN(192-ch) student
# distilled from mobilevit_s + FPN(256-ch) teacher (44.12% mAP)
#
# Strategy:
#   - Start from htdet_low_gflops_192/latest.pth (37.32% FP32 baseline)
#   - Teacher: htdet_mobilevit_finetune_stage2/latest.pth (44.12%) — frozen
#   - KD loss: per-level FPN L2 feature mimicking (weight=0.5)
#   - Detection losses: CIoU + FocalLoss (unchanged)
#   - LR: low (fine-tune regime) with CosineAnnealing
#   - Augmentation: same as v2 (multi-scale + PhotoMetricDistortion)
#
# GPU: A100 40GB — batch=4 (leaves room for teacher eval + v2 parallel run)
# Launch (single GPU):
#   PYTHONPATH=. /workspace/ckarfa/anaconda3/envs/mani/bin/python3.9 \
#     tools/train.py configs/htdet/htdet_xs_192_kd.py --gpu-id 0

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
    samples_per_gpu=2,
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
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
cudnn_benchmark = True

model = dict(
    type='HTDetKD',
    teacher_config='configs/htdet/htdet_teacher_256.py',
    teacher_ckpt='work_dirs/htdet_mobilevit_finetune_stage2/latest.pth',
    kd_weight=0.5,
    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevit_s',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4)),
    neck=dict(
        type='FPN',
        in_channels=[64, 96, 128, 640],
        out_channels=192,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=4,
        in_channels=192,
        stacked_convs=4,
        feat_channels=192,
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

# Fine-tune from existing 37.32% student checkpoint
load_from = 'work_dirs/htdet_low_gflops_192/latest.pth'

# Low LR for fine-tuning (linear scale from bs=4 base)
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=60)
work_dir = 'work_dirs/htdet_xs_192_kd'
auto_resume = False
gpu_ids = [0]
