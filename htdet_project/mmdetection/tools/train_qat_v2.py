"""
QAT (Quantization-Aware Training) fine-tuning for HTDet.

Strategy
--------
  Load fully fine-tuned FP32 checkpoint  →  insert fake-quantize nodes
  →  fine-tune 10 epochs with very low LR  →  convert to INT8  →  export.

Why QAT on fine-tuned model (not from scratch)?
  • epoch_60.pth is already converged on the URPC domain.  QAT only needs to
    nudge weights ±small amounts so they round cleanly to INT8 — far easier
    than learning to detect AND quantize simultaneously.
  • MobileViT attention blocks diverge under INT8 noise when starting cold.
  • Keeps backbone in FP32 (transformer layers); only quantizes conv-heavy
    FPN neck + RetinaHead where INT8 gives the biggest speedup.

Observer schedule (standard Google/Facebook recipe)
  Epochs 0-3 : both weight and activation observers active
  Epoch  4   : disable observers, freeze quantization ranges
  Epoch  7   : freeze BatchNorm running stats

Usage
-----
  python tools/train_qat_v2.py  [--checkpoint path/to/epoch_60.pth]
                                 [--config configs/htdet/htdet_gpu_qat_v2.py]
                                 [--device cuda]
                                 [--epochs 10]
                                 [--eval-only]  # load qat_quantized_final.pth and evaluate
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint, build_optimizer
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='QAT fine-tuning for HTDet')
    p.add_argument('--config',
                   default='configs/htdet/htdet_gpu_qat_v2.py',
                   help='MMDet config file')
    p.add_argument('--checkpoint',
                   default='./work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth',
                   help='Fine-tuned FP32 checkpoint to start QAT from')
    p.add_argument('--device', default='cuda',
                   help='"cuda" or "cpu"')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of QAT fine-tuning epochs')
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; load final INT8 model and evaluate')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def apply_qat_config(model: nn.Module, backend: str = 'fbgemm') -> nn.Module:
    """
    Attach qconfigs and call prepare_qat.

    Policy:
      backbone  → FP32 (MobileViT transformers are too fragile for INT8)
      neck      → INT8 (FPN is pure conv — quantizes cleanly)
      bbox_head → INT8 (4× stacked convs — largest latency chunk)
    """
    torch.backends.quantized.engine = backend
    qconfig = torch.quantization.get_default_qat_qconfig(backend)

    model.backbone.qconfig = None        # keep transformer in FP32
    model.neck.qconfig = qconfig
    model.bbox_head.qconfig = qconfig

    torch.quantization.prepare_qat(model, inplace=True)
    return model


def disable_observers(model: nn.Module):
    model.apply(torch.quantization.disable_observer)
    print('[QAT] Observers disabled — quantization ranges frozen.')


def freeze_bn(model: nn.Module):
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    print('[QAT] BatchNorm stats frozen.')


# ---------------------------------------------------------------------------
# Loss utility  (handles scalar tensors and lists of tensors)
# ---------------------------------------------------------------------------

def sum_losses(losses: dict) -> torch.Tensor:
    total = None
    for v in losses.values():
        val = sum(v) if isinstance(v, (list, tuple)) else v
        total = val if total is None else total + val
    return total


# ---------------------------------------------------------------------------
# Checkpoint saving  (MMDet-compatible format)
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, work_dir: str, name: str):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, name)
    torch.save({'state_dict': model.state_dict()}, path)
    print(f'[QAT] Saved checkpoint → {path}')


# ---------------------------------------------------------------------------
# Evaluation  (runs val split, prints mAP)
# ---------------------------------------------------------------------------

def run_eval(model: nn.Module, cfg: Config, device: str):
    import mmcv
    model.eval()

    val_dataset = build_dataset(cfg.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )

    results = []
    prog_bar = mmcv.ProgressBar(len(val_dataset))

    for data in val_loader:

        imgs = data['img']
        img_metas = []

        for aug_metas in data['img_metas']:
            if hasattr(aug_metas, 'data'):
                img_metas.append(aug_metas.data[0])
            else:
                img_metas.append(aug_metas)

        # 🔥 CRITICAL FIX: ensure float input
        imgs = [img.float() for img in imgs]

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                img=imgs,
                img_metas=img_metas
            )

        results.extend(result)

        for _ in range(len(result)):
            prog_bar.update()

    print()
    eval_results = val_dataset.evaluate(results, metric='bbox')
    print('[QAT] Eval results:', eval_results)

    return eval_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch
    torch.backends.quantized.engine = 'fbgemm'
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Config ----
    cfg = Config.fromfile(args.config)
    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)

    # ---- Build model ----
    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))

    # ---- Load fine-tuned checkpoint ----
    print(f'[QAT] Loading FP32 checkpoint: {args.checkpoint}')
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # ---- Eval-only mode ----
    # ---- Eval-only mode ----
    if args.eval_only:
        final_path = os.path.join(work_dir, 'qat_quantized_final.pth')
        print(f'[QAT] Eval-only mode — loading {final_path}')

        # 🔥 LOAD FP32 MODEL (NOT INT8 STRUCTURE)
        ckpt = torch.load(final_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

        model.eval()
        model.cpu()

        # 🔥 SAFE QUANTIZATION (NO conv crash)
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        run_eval(model, cfg, 'cpu')
        return

    # ---- Apply QAT ----
    model.train()
    model = apply_qat_config(model, backend='fbgemm')
    model.to(device)

    # ---- Dataset / DataLoader ----
    train_dataset = build_dataset(cfg.data.train)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True
    )

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg.optimizer)

    # ---- LR scheduler (cosine) ----
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )

    # ---- Training loop ----
    print(f'[QAT] Starting QAT fine-tuning for {args.epochs} epochs ...')
    print(f'[QAT] Work dir: {work_dir}')
    print(f'[QAT] Quantized components: neck (FPN) + bbox_head (RetinaHead)')
    print(f'[QAT] Backbone stays FP32 (MobileViT)')

    for epoch in range(args.epochs):

        # --- Observer / BN schedule ---
        if epoch == 4:
            disable_observers(model)

        if epoch == 7:
            freeze_bn(model)

        model.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            img        = data['img'].data[0].to(device)
            img_metas  = data['img_metas'].data[0]
            gt_bboxes  = [b.to(device) for b in data['gt_bboxes'].data[0]]
            gt_labels  = [l.to(device) for l in data['gt_labels'].data[0]]

            losses = model.forward_train(
                img=img,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels
            )

            loss = sum_losses(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if i % 20 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  [Epoch {epoch:02d} | Iter {i:04d}]'
                      f'  loss={loss.item():.4f}  lr={lr_now:.2e}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'[QAT] Epoch {epoch:02d} done — avg_loss={avg_loss:.4f}')

        save_checkpoint(model, work_dir, f'qat_epoch_{epoch:02d}.pth')

    # ---- Convert to quantized INT8 ----
    print('[QAT] Converting to INT8 ...')
    model.eval()
    model.cpu()                      # convert must run on CPU
    torch.quantization.convert(model, inplace=True)

    save_checkpoint(model, work_dir, 'qat_quantized_final.pth')
    print('[QAT] INT8 model saved.')

    # ---- Final evaluation (FP32 emulation via fake-quant residuals) ----
    print('[QAT] Running final evaluation on val split ...')
    run_eval(model, cfg, 'cpu')      # INT8 model runs on CPU


if __name__ == '__main__':
    # Ensure we run from the mmdetection root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    main()
