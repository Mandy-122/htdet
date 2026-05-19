# mmdet/models/backbones/timm_backbone.py

import timm
import torch.nn as nn

from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class TIMMBackbone(nn.Module):
    """TIMM Backbone Wrapper for MMDetection"""

    def __init__(self,
                 model_name,
                 pretrained=True,
                 features_only=True,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            out_indices=out_indices
        )

    def forward(self, x):
        feats = self.model(x)

        # Make sure output is tuple (MMDet expects tuple/list)
        if not isinstance(feats, (list, tuple)):
            raise TypeError(
                f"TIMMBackbone output must be list/tuple, got {type(feats)}"
            )

        return tuple(feats)
