#!/bin/bash
# Monitors KD mAP after each epoch. Once epoch 5-6 shows bbox_mAP >= 0.36,
# launches v2_b4 training in parallel. Logs to /tmp/kd_monitor.log.
# Run from mmdetection/ root.

KD_LOG="work_dirs/htdet_xs_192_kd"
V2_LOG="work_dirs/htdet_xs_192_v2_b4"
KD_CFG="configs/htdet/htdet_xs_192_kd.py"
V2_CFG="configs/htdet/htdet_xs_192_v2_b4.py"
PYTHON="/workspace/ckarfa/anaconda3/envs/mani/bin/python3.9"
MIN_EPOCH=5          # Don't act before this epoch
MAP_THRESHOLD=0.36   # Launch v2_b4 if KD mAP exceeds this (baseline was 37.32%)
LOG="/tmp/kd_monitor.log"

echo "[$(date)] Monitor started. Will check KD mAP from epoch $MIN_EPOCH." | tee -a $LOG

last_seen_epoch=0
v2_launched=false

while true; do
    # Parse best mAP so far and last completed epoch from KD json logs
    read BEST_MAP BEST_EP LAST_EP < <(
        python3 -c "
import json, glob, os
logs = sorted(glob.glob('${KD_LOG}/*.log.json'))
best, best_ep, last_ep = 0, 0, 0
for lf in logs:
    with open(lf) as f:
        for line in f:
            try:
                d = json.loads(line)
                ep = d.get('epoch', 0)
                if ep > last_ep: last_ep = ep
                v = d.get('bbox_mAP', 0)
                if v > best: best, best_ep = v, ep
            except: pass
print(best, best_ep, last_ep)
" 2>/dev/null
    )

    BEST_MAP=${BEST_MAP:-0}
    LAST_EP=${LAST_EP:-0}

    # Only act on newly completed epochs
    if [ "$LAST_EP" -gt "$last_seen_epoch" ]; then
        last_seen_epoch=$LAST_EP
        echo "[$(date)] KD epoch $LAST_EP done — best bbox_mAP = $BEST_MAP (epoch $BEST_EP)" | tee -a $LOG

        # After MIN_EPOCH: decide
        if [ "$LAST_EP" -ge "$MIN_EPOCH" ] && [ "$v2_launched" = false ]; then
            MAP_OK=$(python3 -c "print(1 if float('${BEST_MAP}') >= ${MAP_THRESHOLD} else 0)" 2>/dev/null)
            if [ "$MAP_OK" = "1" ]; then
                echo "[$(date)] KD mAP $BEST_MAP >= $MAP_THRESHOLD  -> Launching v2_b4!" | tee -a $LOG
                mkdir -p $V2_LOG
                PYTHONPATH=. nohup $PYTHON tools/train.py $V2_CFG \
                    --gpu-id 0 --work-dir $V2_LOG \
                    > ${V2_LOG}/train_stdout.log 2>&1 &
                echo "[$(date)] v2_b4 launched (PID $!)" | tee -a $LOG
                v2_launched=true
            else
                echo "[$(date)] KD mAP $BEST_MAP < $MAP_THRESHOLD — monitoring continues." | tee -a $LOG
                if [ "$LAST_EP" -ge 8 ]; then
                    echo "[$(date)] Epoch $LAST_EP reached without target mAP. Launching v2_b4 anyway." | tee -a $LOG
                    mkdir -p $V2_LOG
                    PYTHONPATH=. nohup $PYTHON tools/train.py $V2_CFG \
                        --gpu-id 0 --work-dir $V2_LOG \
                        > ${V2_LOG}/train_stdout.log 2>&1 &
                    echo "[$(date)] v2_b4 launched (PID $!)" | tee -a $LOG
                    v2_launched=true
                fi
            fi
        fi
    fi

    # Exit once v2 is launched and KD is done
    if [ "$v2_launched" = true ] && [ "$LAST_EP" -ge 60 ]; then
        echo "[$(date)] KD finished and v2_b4 is running. Monitor exiting." | tee -a $LOG
        break
    fi

    sleep 120  # check every 2 minutes
done
