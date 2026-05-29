"""
UnderwaterCorrect: per-image color correction for underwater imagery.

Underwater scenes suffer from:
  1. Red channel attenuation — water absorbs red wavelengths fastest with depth.
     URPC dataset: R mean=82, G mean=143, B mean=125 — G is 74% brighter than R.
  2. Low local contrast — scattering flattens texture, objects blend into seafloor.
  3. Uneven global brightness — some frames near-blown-out green, others dark/muddy.

This transform applies three corrections in sequence:
  1. Gray-World white balance — scales each channel so their means match,
     effectively removing the per-image color cast without assuming fixed stats.
     clip_ratio controls how aggressively bright pixels are excluded from the mean
     estimate (prevents specular highlights from biasing the correction).
  2. Red-channel boost — after gray-world, applies an additional multiplicative
     boost to the R channel only. Underwater red is *structurally* attenuated, not
     just image-specific, so a fixed extra boost (default 1.2x) consistently helps.
  3. CLAHE (Contrast Limited AHE) — applied to the L channel in LAB space for
     local contrast enhancement without color shift. clip_limit=2.0 prevents
     noise amplification in the already-textured seafloor regions.

Usage in mmdet pipeline:
    custom_imports = dict(imports=['custom_modules.underwater_transform'], ...)

    dict(type='UnderwaterCorrect',
         clip_ratio=0.01,    # top/bottom 1% excluded from WB mean estimate
         red_boost=1.2,      # extra red boost after gray-world (1.0 = off)
         clahe_clip=2.0,     # CLAHE clip limit (0 = off)
         clahe_grid=(8, 8),  # CLAHE tile grid size
         p=1.0)              # apply probability (use <1.0 for stochastic aug)
"""

import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class UnderwaterCorrect:

    def __init__(self,
                 clip_ratio=0.01,
                 red_boost=1.2,
                 clahe_clip=2.0,
                 clahe_grid=(8, 8),
                 p=1.0):
        self.clip_ratio = clip_ratio
        self.red_boost = red_boost
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.p = p
        if clahe_clip > 0:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip,
                tileGridSize=clahe_grid)
        else:
            self.clahe = None

    def _gray_world_wb(self, img):
        """Per-image gray-world white balance. img: float32 BGR [0,255]."""
        result = img.copy()
        for c in range(3):
            ch = img[:, :, c]
            lo = np.percentile(ch, self.clip_ratio * 100)
            hi = np.percentile(ch, (1 - self.clip_ratio) * 100)
            mask = (ch >= lo) & (ch <= hi)
            mean = ch[mask].mean() if mask.any() else ch.mean()
            if mean > 1e-3:
                result[:, :, c] = np.clip(ch * (128.0 / mean), 0, 255)
        return result

    def _red_boost(self, img):
        """Extra multiplicative boost on R channel (index 2 in BGR)."""
        img[:, :, 2] = np.clip(img[:, :, 2] * self.red_boost, 0, 255)
        return img

    def _clahe_lab(self, img):
        """CLAHE on L channel in LAB space. img: float32 BGR."""
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    def __call__(self, results):
        if np.random.random() > self.p:
            return results

        img = results['img'].astype(np.float32)

        img = self._gray_world_wb(img)

        if self.red_boost > 1.0:
            img = self._red_boost(img)

        if self.clahe is not None:
            img = self._clahe_lab(img)

        results['img'] = img.astype(np.uint8)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'clip_ratio={self.clip_ratio}, '
                f'red_boost={self.red_boost}, '
                f'clahe_clip={self.clahe_clip}, '
                f'p={self.p})')
