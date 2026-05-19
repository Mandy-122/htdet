# Copyright (c) OpenMMLab. All rights reserved.
"""
Convert original model checkpoint to low GFLOPs model.

This script:
1. Loads the original epoch_42 checkpoint
2. Extracts backbone weights (100% reusable)
3. Initializes FPN and head with reduced channels (128 vs 256)
4. Saves a new checkpoint ready for fine-tuning

Usage:
    python tools/analysis_tools/convert_to_low_gflops.py \
        work_dirs/htdet_mobilevit_April21st_2/epoch_42.pth \
        work_dirs/low_gflops_init/epoch_42_low_gflops_init.pth
"""

import argparse
import torch
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert to Low GFLOPs checkpoint')
    parser.add_argument('src_checkpoint', help='Source checkpoint path')
    parser.add_argument('dst_checkpoint', help='Destination checkpoint path')
    args = parser.parse_args()
    return args


def convert_checkpoint(src_path, dst_path):
    """Convert original checkpoint to low GFLOPs format."""

    print(f"Loading source checkpoint: {src_path}")
    checkpoint = torch.load(src_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        src_state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        src_state_dict = checkpoint['model']
    else:
        src_state_dict = checkpoint

    print(f"Source checkpoint has {len(src_state_dict)} parameter tensors")

    # Create new state dict
    dst_state_dict = OrderedDict()

    # Copy backbone weights (these are compatible)
    backbone_keys = [k for k in src_state_dict.keys() if 'backbone' in k]
    print(f"\nCopying {len(backbone_keys)} backbone parameter tensors...")

    for key in backbone_keys:
        dst_state_dict[key] = src_state_dict[key]

    # Skip FPN and head weights (dimensions don't match)
    # They will be initialized by the model config

    # Count parameters
    total_params = sum(v.numel() for v in dst_state_dict.values())
    src_params = sum(v.numel() for v in src_state_dict.values())

    print(f"\nStatistics:")
    print(f"  Source parameters: {src_params:,}")
    print(f"  Destination parameters: {total_params:,}")
    print(f"  Backbone parameters: {total_params:,} ({total_params/src_params*100:.1f}%)")
    print(f"  FPN + Head (new): Will be initialized randomly")

    # Save new checkpoint
    import os
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    torch.save({
        'state_dict': dst_state_dict,
        'epoch': 0,
        'meta': {
            'source': src_path,
            'conversion': 'low_gflops',
            'backbone_loaded': True,
            'fpn_head_random': True
        }
    }, dst_path)

    print(f"\nSaved converted checkpoint to: {dst_path}")
    print(f"\nNext step: Fine-tune with:")
    print(f"  python tools/train.py configs/htdet/htdet_gpu_low_gflops_finetune.py \\")
    print(f"      --load-from {dst_path}")


def main():
    args = parse_args()
    convert_checkpoint(args.src_checkpoint, args.dst_checkpoint)


if __name__ == '__main__':
    main()
