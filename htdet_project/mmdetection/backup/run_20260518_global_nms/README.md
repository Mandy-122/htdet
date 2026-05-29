# Backup: run_20260518_global_nms

**Date:** 2026-05-18  
**Checkpoint:** `work_dirs/htdet_mobilevit_April21st_2/epoch_60.pth`  
**Input image:** `data/urpc/val2018/images/CHN083846_0270.jpg` (640×640, ImageNet normalised)

## What was fixed in this run

### Bug 1 — MobileViT norm order in export_weights.py (root cause of 0% match)
`export_weights.py :: add_mobilevit()`: the block-level `norm.weight / norm.bias` was
written **before** the transformer blocks in the binary stream. The C code reads them
**after** all transformer blocks. Moving the two lines after the transformer loop fixed
the backbone completely (C1–C4, P2–P6 all went to cos=1.000, rel_err=0.000).

### Bug 2 — Per-level NMS vs global NMS (improved 73% → 87% match)
`retina_head.h :: retina_head()`: original code ran NMS independently per FPN level,
then merged survivors. MMDetection merges all 5 levels first, then runs one global NMS.
Changed to collect all candidates into a single buffer and run NMS once at the end.
Original per-level NMS code is preserved as commented-out blocks directly above the new code.

## Results

| Metric | Per-level NMS (before) | Global NMS (this run) |
|---|---|---|
| Backbone features (C1–C4) | cos=1.000 ✅ | cos=1.000 ✅ |
| FPN features (P2–P6) | cos=1.000 ✅ | cos=1.000 ✅ |
| Head logits (P3) | cos=1.000 ✅ | cos=1.000 ✅ |
| Python detections | 75 | 75 |
| C-sim detections | 89 | **83** |
| Matched pairs | 65 (73%) | **72 (87%)** |
| Python-only | 10 | **3** |
| C-sim-only | 24 | **11** |

## Directory layout

```
weights/          — exported float32 binary weights (epoch_60.pth, BN fused)
                    backbone_w.bin (19MB), fpn_w.bin (10MB),
                    cls/reg_conv_w.bin (9MB each), cls/reg_pred_w/b.bin, sizes.txt

csim_validation/  — all intermediate and final outputs
  input_image.bin/txt        — preprocessed 640×640 input (CHW float32)
  python_detections.txt      — 75 detections from PyTorch model
  csim_detections.txt        — 83 detections from C-simulation
  combined_detections.json   — both sets merged: matched pairs, python-only, csim-only
  comparison_table.txt       — human-readable side-by-side table
  c1/c2/c3/c4_*.bin         — backbone stage outputs (csim + python)
  p2..p6_*.bin               — FPN level outputs (csim + python)
  cls/reg_feat_p3_*.bin      — head stacked conv outputs at P3
  cls_logits/reg_deltas_*.bin — head prediction conv outputs at P3
  vis_output.jpg             — visualised Python detections on original image
  detect_python.py           — script that produced python_detections.txt
  compare_results.py         — IoU-based comparison script

sources/          — snapshot of all HLS source files + compiled binary
  export_weights.py          — weight exporter (bug 1 fixed here)
  retina_head.h              — detection head (bug 2 fixed here, old code commented out)
  mobilevit_backbone.h       — backbone HLS implementation
  fpn_neck.h                 — FPN HLS implementation
  fpga_types.h / fpga_utils.h — shared types and primitives
  htdet_top.cpp / .h         — top-level kernel
  testbench.cpp              — C-sim testbench source
  testbench_csim             — compiled binary (g++ -O2, DEBUG_HEAD_DUMP=1)
```

## Rebuild command

```bash
cd fpga_temp/
g++ -O2 -std=c++14 -I. -I./hls_stubs -DDEBUG_HEAD_DUMP=1 \
    testbench.cpp htdet_top.cpp -lm -o testbench_csim

./testbench_csim ../weights/ ../csim_validation/input_image.bin --dump ../csim_validation
```
