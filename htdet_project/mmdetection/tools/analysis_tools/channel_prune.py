# Copyright (c) OpenMMLab. All rights reserved.
"""
Channel Pruning (Structured Pruning) for MMDetection Models

This script applies L1-norm based channel pruning to Conv2D layers.
It also scales BatchNorm layers accordingly for better accuracy recovery.

Key insight: When pruning output channels of a Conv layer, we need to:
1. Zero the pruned channel weights in Conv
2. Zero the corresponding BN gamma/beta/running_mean/running_var

Usage:
    python tools/analysis_tools/channel_prune.py \
        configs/htdet/htdet_gpu.py \
        work_dirs/htdet_mobilevit_April21st_2/epoch_42.pth \
        --prune-ratio 0.3 \
        --out-dir work_dirs/pruned_models
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from mmcv import Config
from mmdet.models import build_detector
from collections import OrderedDict
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Channel Pruning (Structured)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file to prune')
    parser.add_argument(
        '--prune-ratio',
        type=float,
        default=0.3,
        help='pruning ratio (0.0-1.0)')
    parser.add_argument(
        '--out-dir',
        default='./work_dirs/pruned_models',
        help='output directory for pruned models')
    parser.add_argument('--verbose', action='store_true', help='print details')
    parser.add_argument(
        '--method',
        type=str,
        default='l1',
        choices=['l1', 'random'],
        help='pruning method')
    args = parser.parse_args()
    return args


class ChannelPruner:
    """
    L1-norm based structured channel pruning with BatchNorm handling.
    """

    def __init__(self, model, prune_ratio=0.3, method='l1', verbose=False):
        self.model = model
        self.prune_ratio = prune_ratio
        self.method = method
        self.verbose = verbose
        self.conv_info = {}

    def compute_channel_importance(self, layer):
        """Compute L1-norm importance for each output channel."""
        weight = layer.weight.data
        importance = weight.abs().sum(dim=(1, 2, 3))
        return importance

    def get_conv_bn_pairs(self):
        """Get Conv2D layers with their following BatchNorm layers."""
        conv_layers = []
        layer_names = []
        bn_layers = {}

        # First pass: collect all BN layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layers[name] = module

        # Second pass: get Conv layers and find their BN
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Skip final prediction heads
                if any(skip in name for skip in ['retina_cls', 'retina_reg',
                                                  'cls_pred', 'bbox_pred']):
                    continue
                conv_layers.append(module)
                layer_names.append(name)

        return conv_layers, layer_names, bn_layers

    def find_bn_after_conv(self, conv_name, bn_layers):
        """Find BatchNorm layer that follows a Conv layer."""
        # Try common patterns
        patterns = [
            conv_name.replace('.conv', '.bn'),
            conv_name + '.bn',
            conv_name.replace('conv', 'bn'),
        ]
        for pattern in patterns:
            if pattern in bn_layers:
                return pattern, bn_layers[pattern]
        return None, None

    def analyze_and_prune(self):
        """Analyze model and apply pruning."""
        print("\n" + "="*60)
        print("Channel Pruning Analysis")
        print("="*60)
        print(f"Pruning ratio: {self.prune_ratio*100:.1f}%")
        print(f"Method: {self.method}")
        print("="*60)

        conv_layers, layer_names, bn_layers = self.get_conv_bn_pairs()
        print(f"\nFound {len(conv_layers)} Conv2D layers to prune\n")

        pruned_count = 0
        total_channels = 0

        for name, layer in zip(layer_names, conv_layers):
            importance = self.compute_channel_importance(layer)
            num_channels = layer.out_channels
            num_to_prune = int(num_channels * self.prune_ratio)
            num_to_keep = num_channels - num_to_prune

            total_channels += num_channels

            if self.method == 'l1':
                # Get indices of LEAST important channels (to prune)
                _, indices = torch.topk(importance, num_to_prune, largest=False)
                channels_to_prune = indices.cpu().numpy()
            else:  # random
                channels_to_prune = np.random.choice(
                    num_channels, num_to_prune, replace=False)

            channels_to_keep = sorted([i for i in range(num_channels)
                                       if i not in channels_to_prune])

            self.conv_info[name] = {
                'original_channels': num_channels,
                'channels_to_prune': channels_to_prune.tolist(),
                'channels_to_keep': channels_to_keep,
                'num_to_keep': num_to_keep,
            }

            # Apply pruning to Conv weights
            weight = layer.weight.data
            weight[channels_to_prune, :, :, :] = 0

            # Also prune corresponding BatchNorm
            bn_name, bn_layer = self.find_bn_after_conv(name, bn_layers)
            if bn_layer is not None:
                # Zero BN params for pruned channels
                bn_layer.weight.data[channels_to_prune] = 0
                bn_layer.bias.data[channels_to_prune] = 0
                bn_layer.running_mean.data[channels_to_prune] = 0
                bn_layer.running_var.data[channels_to_prune] = 1
                self.conv_info[name]['bn_name'] = bn_name

            pruned_count += num_to_prune

            if self.verbose:
                bn_info = f" + BN: {bn_name}" if bn_name else ""
                print(f"  {name}: pruning {num_to_prune}/{num_channels} channels{bn_info}")

        print(f"\nTotal channels pruned: {pruned_count}/{total_channels} "
              f"({pruned_count/total_channels*100:.1f}%)")

        return self.conv_info

    def get_stats(self):
        """Get pruning statistics."""
        if not self.conv_info:
            return {}

        total_original = sum(info['original_channels'] for info in self.conv_info.values())
        total_pruned = sum(len(info['channels_to_prune']) for info in self.conv_info.values())

        return {
            'total_original_channels': total_original,
            'total_pruned_channels': total_pruned,
            'channels_kept': total_original - total_pruned,
            'reduction_ratio': total_pruned / total_original if total_original > 0 else 0,
            'layers_pruned': len(self.conv_info)
        }


def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    print("Building model...")
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.eval()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Handle checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully")

    # Original stats
    original_params = count_parameters(model)
    print(f"\nOriginal model parameters: {original_params:,}")

    # Create pruner and apply pruning
    pruner = ChannelPruner(
        model,
        prune_ratio=args.prune_ratio,
        method=args.method,
        verbose=args.verbose
    )
    pruner.analyze_and_prune()

    # Stats after pruning
    pruned_params = count_parameters(model)
    print(f"\nModel parameters after pruning: {pruned_params:,}")

    # Get stats
    stats = pruner.get_stats()
    print(f"\nChannel pruning statistics:")
    print(f"  Total original channels: {stats.get('total_original_channels', 0):,}")
    print(f"  Channels pruned: {stats.get('total_pruned_channels', 0):,}")
    print(f"  Channels kept: {stats.get('channels_kept', 0):,}")
    print(f"  Channel reduction: {stats.get('reduction_ratio', 0) * 100:.2f}%")

    # Save pruned model
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    out_filename = f"{checkpoint_name}_channel_pruned_r{args.prune_ratio:.2f}.pth"
    out_path = os.path.join(args.out_dir, out_filename)

    print(f"\nSaving pruned model to: {out_path}")
    torch.save({
        'state_dict': model.state_dict(),
        'channel_masks': {k: v['channels_to_keep'] for k, v in pruner.conv_info.items()},
        'pruned_channels': {k: v['channels_to_prune'] for k, v in pruner.conv_info.items()},
        'pruning_stats': stats,
        'original_params': original_params,
        'prune_ratio': args.prune_ratio,
        'method': args.method,
        'config': args.config,
        'epoch': checkpoint.get('epoch', -1),
    }, out_path)

    print("Pruned model saved successfully!")

    # Save JSON stats
    stats_path = out_path.replace('.pth', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'config': args.config,
            'checkpoint': args.checkpoint,
            'prune_ratio': args.prune_ratio,
            'method': args.method,
            'original_params': original_params,
            'channel_stats': stats,
            'layer_details': {
                name: {
                    'original': info['original_channels'],
                    'pruned': len(info['channels_to_prune']),
                    'kept': info['channels_to_keep']
                }
                for name, info in pruner.conv_info.items()
            }
        }, f, indent=2)

    print(f"Stats saved to: {stats_path}")

    # Print next steps
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print(f"""
Channel pruning zeros weights - the model needs fine-tuning to recover.

1. Fine-tune the pruned model:
   python tools/train.py configs/htdet/htdet_gpu_pruned_finetune.py \\
       --load-from {out_path}

2. Evaluate after fine-tuning:
   python tools/test.py configs/htdet/htdet_gpu.py <finetuned_ckpt> --eval bbox

Expected: Accuracy will be low initially, then recover after fine-tuning.
""")
    print("="*60)

    return out_path, stats


if __name__ == '__main__':
    main()
