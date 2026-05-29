#!/bin/bash
# Evaluate all epoch checkpoints from htdet_mobilevit_April21st_2 on the TRAINING set.
# Outputs one mAP line per epoch so you can compare train vs val curves.

PYTHON=/workspace/ckarfa/anaconda3/envs/mani/bin/python3.9
CONFIG=work_dirs/htdet_mobilevit_April21st_2/htdet_gpu.py
CKPT_DIR=work_dirs/htdet_mobilevit_April21st_2
OUT=work_dirs/htdet_mobilevit_April21st_2/train_map_results.txt

cd /workspace/ckarfa/htdet/htdet_project/mmdetection

echo "Epoch,train_mAP" > $OUT

for ep in $(seq 1 60); do
    CKPT=$CKPT_DIR/epoch_${ep}.pth
    if [ ! -f "$CKPT" ]; then
        echo "$ep,N/A" >> $OUT
        continue
    fi

    MAP=$($PYTHON tools/test.py $CONFIG $CKPT \
        --eval bbox \
        --cfg-options \
            data.test.ann_file=data/urpc/annotations/instances_train2018.json \
            data.test.img_prefix=data/urpc/train2018/images/ \
        2>/dev/null | grep "bbox_mAP:" | tail -1 | grep -oP "bbox_mAP: \K[0-9.]+")

    echo "Ep $ep: train_mAP=$MAP"
    echo "$ep,$MAP" >> $OUT
done

echo "Done. Results saved to $OUT"
