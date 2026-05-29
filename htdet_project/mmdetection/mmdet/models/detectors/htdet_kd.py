# Copyright (c) HTDet. FPN feature-mimicking knowledge distillation.
# Teacher: mobilevit_s + FPN(256-ch)  Student: mobilevit_s + FPN(192-ch)
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint

from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class HTDetKD(SingleStageDetector):
    """FPN feature-mimicking KD for HTDet.

    The student (192-ch FPN) learns to mimic the teacher (256-ch FPN)
    intermediate features via per-level L2 loss with a 192→256 adapter.

    Args:
        teacher_config (str): Path to teacher model config.
        teacher_ckpt (str): Path to teacher checkpoint.
        kd_weight (float): Weight for the FPN mimicking loss. Default 0.5.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt,
                 kd_weight=0.5,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.kd_weight = kd_weight

        # Build frozen teacher
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt:
            load_checkpoint(self.teacher_model, teacher_ckpt,
                            map_location='cpu')
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        # Per-level adapter: student 192-ch → teacher 256-ch (for L2 mimicking)
        student_ch = neck['out_channels']
        teacher_ch = teacher_config['model']['neck']['out_channels']
        num_outs = neck.get('num_outs', 5)
        self.kd_adapters = nn.ModuleList([
            nn.Conv2d(student_ch, teacher_ch, 1, bias=False)
            for _ in range(num_outs)
        ])

    # ------------------------------------------------------------------ #
    # Prevent teacher_model from registering as an nn.Module submodule
    # (keeps teacher params out of optimizer)
    # ------------------------------------------------------------------ #
    def __setattr__(self, name, value):
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def cuda(self, device=None):
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        self.teacher_model.train(False)
        super().train(mode)

    # ------------------------------------------------------------------ #
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None):
        student_feats = self.extract_feat(img)

        with torch.no_grad():
            teacher_feats = self.teacher_model.extract_feat(img)

        # Standard RetinaNet detection losses
        losses = self.bbox_head.forward_train(
            student_feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

        # FPN mimicking loss: L2(adapter(student_feat), teacher_feat)
        kd_loss = sum(
            F.mse_loss(self.kd_adapters[i](student_feats[i]),
                       teacher_feats[i].detach())
            for i in range(len(student_feats))
        ) / len(student_feats)
        losses['loss_kd'] = self.kd_weight * kd_loss

        return losses
