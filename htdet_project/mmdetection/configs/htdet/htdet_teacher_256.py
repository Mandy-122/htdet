# Teacher model config for KD: mobilevit_s + FPN(256-ch) + RetinaNet head
# mAP=44.12% on URPC val (htdet_mobilevit_finetune_stage2/latest.pth)
# This file is ONLY used as teacher_config in htdet_xs_192_kd.py.
# It is NOT a training config (no data/optimizer/runner sections).

model = dict(
    type='RetinaNet',
    backbone=dict(
        type='TIMMBackbone',
        model_name='mobilevit_s',
        pretrained=False,
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
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
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
