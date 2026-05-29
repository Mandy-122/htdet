"""
RetinaHeadDropout: RetinaHead with dropout after each stacked conv.

Standard RetinaHead has 4 stacked convs with no regularization between them.
With only 2901 training images, the head overfits easily — adding dropout
forces it to learn redundant, generalizable features instead of memorizing.

drop_rate=0.1 is conservative (10% neurons dropped). Use 0.2 for stronger reg.
"""
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RetinaHead


@HEADS.register_module()
class RetinaHeadDropout(RetinaHead):

    def __init__(self, *args, drop_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_rate = drop_rate
        self.cls_dropout = nn.ModuleList(
            [nn.Dropout2d(p=drop_rate) for _ in self.cls_convs])
        self.reg_dropout = nn.ModuleList(
            [nn.Dropout2d(p=drop_rate) for _ in self.reg_convs])

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv, cls_drop in zip(self.cls_convs, self.cls_dropout):
            cls_feat = cls_drop(cls_conv(cls_feat))
        for reg_conv, reg_drop in zip(self.reg_convs, self.reg_dropout):
            reg_feat = reg_drop(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
