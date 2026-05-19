# # configs/htdet/htdet_gpu.py

# _base_ = [
#     '../_base_/datasets/urpc_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]

# cudnn_benchmark = True


# # -----------------------------
# # Model
# # -----------------------------
# model = dict(
#     type='RetinaNet',
#     init_cfg=dict(
#     type='Pretrained',
#     checkpoint='torchvision://retinanet_resnet50_fpn'
#     ),

#     backbone=dict(
#         type='MobileNetV2',
#         out_indices=(1, 2, 4, 6),
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='torchvision://mobilenet_v2'
#         )
#     ),


#     neck=dict(
#         type='FPN',
#         in_channels=[24, 32, 96, 320],
#         out_channels=256,
#         num_outs=5
#     ),


#     bbox_head=dict(
#         type='RetinaHead',
#         num_classes=4,
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,

#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[2, 4, 8],
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
# # Dataloader (GPU)
# # -----------------------------
# data = dict(
#     samples_per_gpu=4,      # Increase batch size
#     workers_per_gpu=4,      # Parallel data loading
# )


# # -----------------------------
# # Optimizer
# # -----------------------------
# optimizer = dict(
#     _delete_=True,
#     type='SGD',
#     lr=0.02,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.5)
#         }
#     )
# )



# optimizer_config = dict(
#     _delete_=True,
#     grad_clip=dict(max_norm=5, norm_type=2)
# )


# # -----------------------------
# # LR Schedule
# # -----------------------------
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=0.00066667,
#     step=[16, 22]
# )



# # -----------------------------
# # Training
# # -----------------------------
# runner = dict(
#     type='EpochBasedRunner',
#     max_epochs=24
# )

# evaluation = dict(interval=1, metric='bbox')

# checkpoint_config = dict(interval=1)

# log_config = dict(interval=20)

# work_dir = './work_dirs/htdet_gpu'









_base_ = [
    '../_base_/datasets/urpc_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

cudnn_benchmark = True


# -----------------------------
# Model
# -----------------------------
model = dict(
    type='RetinaNet',

    backbone=dict(
        type='MobileNetV2',
        out_indices=(1, 2, 4, 6),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://mobilenet_v2'
        )
    ),



    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
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
            scales=[2, 4, 8],
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
# Dataloader (GPU)
# -----------------------------
data = dict(
    samples_per_gpu=4,      # Increase batch size
    workers_per_gpu=4,      # Parallel data loading
)


# -----------------------------
# Optimizer
# -----------------------------
optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.5)
        }
    )
)



optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=5, norm_type=2)
)


# -----------------------------
# LR Schedule
# -----------------------------
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.00066667,
    step=[16, 22]
)



# -----------------------------
# Training
# -----------------------------
runner = dict(
    type='EpochBasedRunner',
    max_epochs=24
)

evaluation = dict(interval=1, metric='bbox')

checkpoint_config = dict(interval=1)

log_config = dict(interval=20)

work_dir = './work_dirs/htdet_gpu'

