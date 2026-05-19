# Copyright (c) OpenMMLab. All rights reserved.
"""
Unstructured Weight Pruning for MMDetection Models

This script applies magnitude-based unstructured pruning to model weights.
It supports different pruning ratios and saves the pruned model for re-training.

Usage:
    python tools/analysis_tools/prune_weights.py \
        configs/htdet/htdet_gpu.py \
        path/to/checkpoint.pth \
        --prune-ratio 0.3 \
        --out-dir ./pruned_models
"""

import argparse
import os
import torch
import torch.nn as nn
from mmcv import Config
from mmdet.models import build_detector
from torch.nn.utils import prune


def parse_args():
    parser = argparse.ArgumentParser(description='Unstructured Weight Pruning')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file to prune')
    parser.add_argument(
        '--prune-ratio',
        type=float,
        default=0.3,
        help='pruning ratio (0.0-1.0), e.g., 0.3 means 30%% of weights will be zeroed')
    parser.add_argument(
        '--out-dir',
        default='./pruned_models',
        help='output directory for pruned models')
    parser.add_argument(
        '--prune-type',
        type=str,
        default='l1',
        choices=['l1', 'random'],
        help='pruning type: l1 (magnitude-based) or random')
    parser.add_argument(
        '--layer-types',
        type=str,
        nargs='+',
        default=['Conv2d', 'Linear'],
        help='layer types to prune')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='override some settings in config')
    args = parser.parse_args()
    return args


def get_layers_by_type(model, layer_types=['Conv2d', 'Linear']):
    """Get all layers of specified types from the model."""
    layers_to_prune = []
    layer_names = []

    for name, module in model.named_modules():
        if type(module).__name__ in layer_types:
            # Skip certain layers that shouldn't be pruned
            # (e.g., output layers, classification heads)
            if any(skip in name for skip in ['cls_pred', 'bbox_pred']):
                print(f"  Skipping {name} (output layer)")
                continue
            layers_to_prune.append(module)
            layer_names.append(name)

    return layers_to_prune, layer_names


def apply_unstructured_pruning(model, prune_ratio=0.3, prune_type='l1', layer_types=['Conv2d', 'Linear']):
    """
    Apply unstructured pruning to the model.

    Args:
        model: PyTorch model
        prune_ratio: Fraction of weights to prune (0.0-1.0)
        prune_type: 'l1' for magnitude-based, 'random' for random pruning
        layer_types: List of layer type names to prune

    Returns:
        pruned_model: Model with pruning applied
        stats: Dictionary with pruning statistics
    """
    print("\n" + "="*60)
    print("Starting Unstructured Weight Pruning")
    print("="*60)
    print(f"Pruning ratio: {prune_ratio*100:.1f}%")
    print(f"Pruning type: {prune_type}")
    print(f"Layer types: {layer_types}")
    print("="*60)

    # Get layers to prune
    layers, layer_names = get_layers_by_type(model, layer_types)
    print(f"\nFound {len(layers)} layers to prune:\n")

    stats = {
        'total_weights': 0,
        'pruned_weights': 0,
        'layer_stats': []
    }

    # Apply pruning to each layer
    for name, layer in zip(layer_names, layers):
        # Get weight statistics
        weight = layer.weight.data
        total_weights = weight.numel()
        stats['total_weights'] += total_weights

        # Apply pruning
        if prune_type == 'l1':
            prune.l1_unstructured(layer, name='weight', amount=prune_ratio)
        elif prune_type == 'random':
            prune.random_unstructured(layer, name='weight', amount=prune_ratio)

        # Count pruned weights
        pruned_mask = layer.weight_mask.sum()
        pruned_count = total_weights - pruned_mask.item()
        stats['pruned_weights'] += pruned_count

        layer_prune_ratio = pruned_count / total_weights * 100
        stats['layer_stats'].append({
            'name': name,
            'total': total_weights,
            'pruned': pruned_count,
            'ratio': layer_prune_ratio
        })

        print(f"  {name}: {layer_prune_ratio:.2f}% pruned ({pruned_count}/{total_weights})")

    # Make pruning permanent (remove re-parameterization)
    print("\nMaking pruning permanent...")
    for name, layer in zip(layer_names, layers):
        prune.remove(layer, 'weight')

    # Calculate final statistics
    overall_prune_ratio = stats['pruned_weights'] / stats['total_weights'] * 100 if stats['total_weights'] > 0 else 0

    print("\n" + "="*60)
    print("Pruning Summary")
    print("="*60)
    print(f"Total weights: {stats['total_weights']:,}")
    print(f"Pruned weights: {stats['pruned_weights']:,}")
    print(f"Overall pruning ratio: {overall_prune_ratio:.2f}%")
    print("="*60)

    return model, stats


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build model
    print("Building model...")
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully")

    # Apply pruning
    pruned_model, prune_stats = apply_unstructured_pruning(
        model,
        prune_ratio=args.prune_ratio,
        prune_type=args.prune_type,
        layer_types=args.layer_types
    )

    # Save pruned model
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    out_filename = f"{checkpoint_name}_pruned_{args.prune_type}_ratio{args.prune_ratio:.2f}.pth"
    out_path = os.path.join(args.out_dir, out_filename)

    print(f"\nSaving pruned model to: {out_path}")
    torch.save({
        'state_dict': pruned_model.state_dict(),
        'prune_stats': prune_stats,
        'config': args.config,
        'prune_ratio': args.prune_ratio,
        'prune_type': args.prune_type
    }, out_path)

    print(f"\nPruned model saved successfully!")
    print(f"Output path: {out_path}")

    # Save pruning statistics
    stats_path = out_path.replace('.pth', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Unstructured Weight Pruning Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Pruning ratio: {args.prune_ratio*100:.1f}%\n")
        f.write(f"Pruning type: {args.prune_type}\n")
        f.write(f"Layer types: {args.layer_types}\n\n")
        f.write("="*60 + "\n")
        f.write("Overall Statistics\n")
        f.write("="*60 + "\n")
        f.write(f"Total weights: {prune_stats['total_weights']:,}\n")
        f.write(f"Pruned weights: {prune_stats['pruned_weights']:,}\n")
        f.write(f"Overall pruning ratio: {prune_stats['pruned_weights']/prune_stats['total_weights']*100:.2f}%\n\n")
        f.write("="*60 + "\n")
        f.write("Per-Layer Statistics\n")
        f.write("="*60 + "\n")
        for layer_stat in prune_stats['layer_stats']:
            f.write(f"{layer_stat['name']}: {layer_stat['pruned']}/{layer_stat['total']} ({layer_stat['ratio']:.2f}%)\n")

    print(f"Pruning statistics saved to: {stats_path}")

    return out_path, prune_stats


if __name__ == '__main__':
    main()
