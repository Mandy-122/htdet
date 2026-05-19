# # Channel Pruned Model Config
# # Uses HTDetMobileViTPruned backbone with width multiplier for channel reduction
# # Also reduces FPN and head channels for actual GFLOPs reduction
# # 30% channel reduction via width_mult=0.7

# _base_ = [
#     '../_base_/datasets/urpc_detection.py',
#     '../_base_/default_runtime.py'
# ]

# cudnn_benchmark = True


# # -----------------------------
# # Model (Channel Pruned)
# # -----------------------------
# model = dict(
#     type='RetinaNet',

#     # Use pruned backbone with width multiplier
#     backbone=dict(
#         type='HTDetMobileViTPruned',
#         in_channels=3,
#         width=16,
#         width_mult=0.7,  # 30% channel reduction
#         init_cfg=None
#     ),

#     # Backbone out_channels with width_mult=0.7: [11, 22, 44, 88, 176]
#     # Also reduce FPN out_channels for GFLOPs reduction
#     neck=dict(
#         type='FPN',
#         in_channels=[11, 22, 44, 88, 176],
#         out_channels=176,  # Reduced from 256 (30% reduction)
#         num_outs=5
#     ),

#     bbox_head=dict(
#         type='RetinaHead',
#         num_classes=4,
#         in_channels=176,  # Match FPN out_channels
#         stacked_convs=4,
#         feat_channels=176,  # Reduced from 256

#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[4, 6, 8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]
#         ),

#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[0., 0., 0., 0.],
#             target_stds=[1., 1., 1., 1.]
#         ),

#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0
#         ),

#         loss_bbox=dict(
#             type='L1Loss',
#             loss_weight=1.0
#         )
#     ),

#     train_cfg=dict(
#         assigner=dict(
#             type='MaxIoUAssigner',
#             pos_iou_thr=0.5,
#             neg_iou_thr=0.4,
#             min_pos_iou=0,
#             ignore_iof_thr=-1
#         ),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False
#     ),

#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=100
#     )
# )


# # -----------------------------
# # Dataloader
# # -----------------------------
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
# )


# # -----------------------------
# # Optimizer
# # -----------------------------
# optimizer = dict(
#     type='SGD',
#     lr=0.001,
#     momentum=0.9,
#     weight_decay=0.0001
# )

# optimizer_config = dict(
#     grad_clip=dict(max_norm=5, norm_type=2)
# )


# # -----------------------------
# # LR Schedule
# # -----------------------------
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.1,
#     step=[40, 55]
# )


# # -----------------------------
# # Training
# # -----------------------------
# runner = dict(
#     type='EpochBasedRunner',
#     max_epochs=60
# )

# evaluation = dict(interval=1, metric='bbox')
# checkpoint_config = dict(interval=1)
# log_config = dict(interval=20)

# work_dir = './work_dirs/htdet_channel_pruned'




_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True


# -----------------------------
# Model (Fixed Channel Pruning)
# -----------------------------
model = dict(
    type='RetinaNet',

    backbone=dict(
        type='HTDetMobileViTPruned',
        in_channels=3,
        width=16,
        width_mult=0.7,   # your current pruning
        init_cfg=None     # keep as-is (or add pretrained if available)
    ),

    # ✅ FIXED: Correct channels + strong FPN
    neck=dict(
        type='FPN',
        in_channels=[11, 22, 44, 88, 176],  # EXACT from your backbone
        out_channels=256,                   # RESTORED
        num_outs=5
    ),

    # ✅ FIXED: Strong detection head
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


# -----------------------------
# Dataloader
# -----------------------------
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)


# -----------------------------
# Optimizer (more stable)
# -----------------------------
optimizer = dict(
    type='SGD',
    lr=0.0005,   # LOWER for stability
    momentum=0.9,
    weight_decay=0.0001
)

optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2)
)


# -----------------------------
# LR Schedule
# -----------------------------
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[40, 55]
)


# -----------------------------
# Training
# -----------------------------
runner = dict(
    type='EpochBasedRunner',
    max_epochs=60
)

evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP'
)

checkpoint_config = dict(interval=1)

log_config = dict(interval=20)

work_dir = './work_dirs/htdet_channel_pruned_fixed'