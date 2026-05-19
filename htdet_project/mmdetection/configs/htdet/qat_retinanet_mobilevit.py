import torch
import os
from mmcv import Config
from mmcv.runner import load_checkpoint, build_optimizer
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

# -----------------------------
# 0. Work directory (DEFINE FIRST)
# -----------------------------
work_dir = './work_dirs/qat_mobilevit_run'
os.makedirs(work_dir, exist_ok=True)

# -----------------------------
# 1. Load config
# -----------------------------
config_file = './configs/htdet/htdet_gpu.py'
checkpoint_file = './work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth'

cfg = Config.fromfile(config_file)

# Lower LR for QAT stability
cfg.optimizer['lr'] = 5e-6

qat_epochs = 8

# -----------------------------
# 2. Build model
# -----------------------------
model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg')
)

load_checkpoint(model, checkpoint_file, map_location='cpu')

model.train()

# -----------------------------
# 3. QAT Setup
# -----------------------------
torch.backends.quantized.engine = 'fbgemm'
qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

model.backbone.qconfig = None
model.neck.qconfig = qconfig
model.bbox_head.qconfig = qconfig

torch.quantization.prepare_qat(model, inplace=True)

# -----------------------------
# 4. Dataset
# -----------------------------
dataset = build_dataset(cfg.data.train)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=2,
    workers_per_gpu=2,
    dist=False,
    shuffle=True
)

# -----------------------------
# 5. Optimizer
# -----------------------------
optimizer = build_optimizer(model, cfg.optimizer)

# -----------------------------
# 6. QAT Training Loop
# -----------------------------
print("Starting QAT fine-tuning...")

for epoch in range(qat_epochs):

    if epoch == 3:
        print("Disabling observers...")
        model.apply(torch.quantization.disable_observer)

    if epoch == 6:
        print("Freezing BN stats...")
        model.apply(torch.quantization.freeze_bn_stats)

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        img = data['img'].data[0]
        img_metas = data['img_metas'].data[0]

        gt_bboxes = data['gt_bboxes'].data[0]
        gt_labels = data['gt_labels'].data[0]

        losses = model.forward_train(
            img=img,
            img_metas=img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        loss = sum(
            sum(v) if isinstance(v, list) else v
            for v in losses.values()
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        if i % 20 == 0:
            print(f"[Epoch {epoch} | Iter {i}] Loss: {loss.item():.4f}")

    # ✅ Save intermediate checkpoints
    torch.save(
        model.state_dict(),
        os.path.join(work_dir, f'qat_epoch_{epoch}.pth')
    )

print("QAT training completed.")

# -----------------------------
# 7. Convert to quantized model
# -----------------------------
model.eval()
torch.quantization.convert(model, inplace=True)

# -----------------------------
# 8. Save final model
# -----------------------------
torch.save(
    model.state_dict(),
    os.path.join(work_dir, 'qat_quantized_final.pth')
)

print("QAT quantized model saved successfully.")