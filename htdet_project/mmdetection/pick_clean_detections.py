"""
Run baseline model on val images, pick clean results (no overlap, score>=0.5),
save visualizations for the report.
"""
import os, sys, json
import numpy as np
import cv2

sys.path.insert(0, '/workspace/ckarfa/htdet/htdet_project/mmdetection')
os.chdir('/workspace/ckarfa/htdet/htdet_project/mmdetection')

from mmdet.apis import init_detector, inference_detector

CONFIG   = 'configs/htdet/htdet_gpu.py'
CKPT     = 'work_dirs/htdet_mobilevit_April21st_2/epoch_47.pth'
IMG_DIR  = 'data/urpc/val2018/images/'
OUT_DIR  = 'MTP2_Mani_Deep_G/clean_detections/'
SCORE_THR = 0.50
MAX_OVERLAP_IOU = 0.25  # reject images with any pair IoU above this

CLASS_NAMES  = ['holothurian', 'echinus', 'scallop', 'starfish']
CLASS_COLORS = {
    'holothurian': (0,   0,   220),   # red (BGR)
    'echinus':     (220, 0,   0),     # blue
    'scallop':     (0,   180, 0),     # green
    'starfish':    (0,   165, 255),   # orange
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(OUT_DIR, exist_ok=True)

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def max_pairwise_iou(boxes):
    if len(boxes) < 2:
        return 0.0
    mx = 0.0
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            mx = max(mx, box_iou(boxes[i], boxes[j]))
    return mx

def draw_detections(img, result):
    out = img.copy()
    all_boxes = []
    for cls_id, bboxes in enumerate(result):
        name  = CLASS_NAMES[cls_id]
        color = CLASS_COLORS[name]
        for *xyxy, score in bboxes:
            if score < SCORE_THR:
                continue
            x1,y1,x2,y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
            all_boxes.append([x1,y1,x2,y2])
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            label = f'{name} {score:.2f}'
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            cv2.rectangle(out, (x1, y1-th-4), (x1+tw+2, y1), color, -1)
            cv2.putText(out, label, (x1+1, y1-3), FONT, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out, all_boxes

print("Loading model...")
model = init_detector(CONFIG, CKPT, device='cpu')

imgs = sorted(os.listdir(IMG_DIR))
print(f"Scanning {min(200, len(imgs))} images...")

candidates = []
for fname in imgs[:200]:
    if not fname.endswith('.jpg'):
        continue
    path = os.path.join(IMG_DIR, fname)
    result = inference_detector(model, path)

    # Collect all above-threshold boxes + unique classes
    all_boxes = []
    classes_seen = set()
    total_dets = 0
    for cls_id, bboxes in enumerate(result):
        for *xyxy, score in bboxes:
            if score >= SCORE_THR:
                all_boxes.append(xyxy)
                classes_seen.add(cls_id)
                total_dets += 1

    if total_dets < 2:
        continue

    max_iou = max_pairwise_iou(all_boxes)
    candidates.append({
        'fname': fname, 'path': path, 'result': result,
        'n_dets': total_dets, 'n_classes': len(classes_seen),
        'max_iou': max_iou, 'classes': classes_seen,
    })

print(f"Found {len(candidates)} images with >=2 detections at score>=0.5")

# Sort: prefer more classes, fewer overlaps, reasonable number of detections
candidates.sort(key=lambda x: (-x['n_classes'], x['max_iou'], -x['n_dets']))

# Keep only those with low overlap
clean = [c for c in candidates if c['max_iou'] < MAX_OVERLAP_IOU]
print(f"Clean (max_iou < {MAX_OVERLAP_IOU}): {len(clean)} images")

# Save top 12 for review
saved = []
for c in clean[:12]:
    img = cv2.imread(c['path'])
    vis, _ = draw_detections(img, c['result'])
    out_path = os.path.join(OUT_DIR, c['fname'])
    cv2.imwrite(out_path, vis)
    saved.append(c)
    print(f"  {c['fname']}: {c['n_dets']} dets, {c['n_classes']} classes, max_iou={c['max_iou']:.3f}")

print(f"\nSaved {len(saved)} images to {OUT_DIR}")
