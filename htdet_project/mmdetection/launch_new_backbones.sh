#!/bin/bash
# Sequential launcher: ConvNeXt backbone experiments
# Runs after both htdet_s_256_fafpn_scratch (PID 1973358) and
# htdet_effb3_scratch (PID 1977787) finish.
#
# Order:
#   1. ConvNeXt-Tiny + plain FPN  (htdet_convnext_scratch)
#   2. ConvNeXt-Tiny + FA-FPN     (htdet_convnext_fafpn_scratch)
#
# EfficientNet FA-FPN can be queued after effb3_scratch finishes.

PYTHON=/workspace/ckarfa/anaconda3/envs/mani/bin/python3.9
cd /workspace/ckarfa/htdet/htdet_project/mmdetection

echo "=== Waiting for fafpn_scratch (1973358) and effb3_scratch (1977787) to finish ==="
while kill -0 1973358 2>/dev/null || kill -0 1977787 2>/dev/null; do
    sleep 60
done
echo "=== Both done. Starting ConvNeXt experiments ==="

# 1. ConvNeXt-Tiny + plain FPN
echo "=== [1/2] ConvNeXt-Tiny + plain FPN ==="
$PYTHON tools/train.py configs/htdet/htdet_convnext_scratch.py \
    --work-dir work_dirs/htdet_convnext_scratch \
    --gpu-ids 0 \
    2>&1 | tee work_dirs/htdet_convnext_scratch/train_stdout.log
echo "=== [1/2] ConvNeXt-Tiny FPN done ==="

# 2. ConvNeXt-Tiny + FA-FPN
echo "=== [2/2] ConvNeXt-Tiny + FA-FPN ==="
$PYTHON tools/train.py configs/htdet/htdet_convnext_fafpn_scratch.py \
    --work-dir work_dirs/htdet_convnext_fafpn_scratch \
    --gpu-ids 0 \
    2>&1 | tee work_dirs/htdet_convnext_fafpn_scratch/train_stdout.log
echo "=== [2/2] ConvNeXt-Tiny FA-FPN done ==="

echo "=== ConvNeXt experiments complete ==="
