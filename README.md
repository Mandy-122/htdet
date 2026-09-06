# HTDet — An FPGA Accelerator for Hybrid Transformer Underwater Object Detection

A hybrid CNN-Transformer object detector for underwater imagery, compressed and rewritten in
synthesisable C/C++ so that it can run on a Xilinx Zynq UltraScale+ FPGA instead of a GPU.

The detector finds four marine species — **holothurian** (sea cucumber), **echinus** (sea urchin),
**scallop**, and **starfish** — in the URPC (Underwater Robot Picking Contest) benchmark. It reaches **76.34% mAP50** in its full PyTorch form, and **72.80% mAP50** after
the compression needed to fit an FPGA.

This is the code accompanying an M.Tech thesis at IIT Guwahati (Mani Deep G, 244101027, supervised by
Dr. Chandan Karfa). The full write-up is at
[`htdet_project/mmdetection/MTP2_Mani_Deep_G/thesis.pdf`](htdet_project/mmdetection/MTP2_Mani_Deep_G/thesis.pdf).

---

## Why build this?

Autonomous underwater vehicles have to detect objects *onboard*. Acoustic links to the surface run at
1–10 kbps, which is far too slow to stream frames off the vehicle, so cloud inference is not an
option. Meanwhile a desktop GPU draws 150–350 W and needs forced airflow, neither of which survives a
sealed, battery-powered pressure enclosure with a 500 Wh–2 kWh budget.

That pushes inference onto a low-power FPGA (5–15 W, deterministic latency, no fans). The problem is
that FPGAs cannot run PyTorch. Getting a modern transformer-based detector onto one means
co-designing the model and the hardware: shrinking the network until it fits the on-chip memory and
DSP budget, then hand-writing the entire inference graph in a restricted subset of C++ that Vitis HLS
can compile to RTL.

This repository is the record of that process end to end — training, compression, C conversion,
verification, and synthesis.

---

## Results

**Detection accuracy on URPC val2018**, compared against standard detectors on the same benchmark:

| Model | mAP@[.50:.95] | mAP@0.50 |
|---|---|---|
| Faster R-CNN | 0.358 | 0.698 |
| FCOS | 0.347 | 0.697 |
| YOLOv3 | 0.378 | 0.721 |
| HTDet base paper (MobileViT-S + FG-FPN) | 0.385 | 0.723 |
| **This work (MobileViT-S + FPN-256)** | **0.4124** | **0.7634** |

**The compression path from that baseline down to the deployed FPGA model.** Each row is a trained
and evaluated model, not an estimate:

| Configuration | Params | GFLOPs | GFLOPs saved | mAP@0.50 |
|---|---|---|---|---|
| Baseline (FPN/head at 256 ch) | 12.4 M | 198.9 | — | 0.7634 |
| 30% unstructured weight pruning + fine-tune | 12.4 M | 198.9 | 0% | 0.7618 |
| Channel-224 | 10.7 M | 155.7 | 21.7% | 0.7337 |
| **Channel-192 ← selected** | **9.2 M** | **118.1** | **40.6%** | **0.7287** |
| Channel-128 | 6.9 M | 59.9 | 69.9% | 0.6394 |
| Channel-192 + INT8 post-training quantization | 9.2 M | 118.1 | 40.6% | 0.7280 |

Two things drove the choice of Channel-192. Weight pruning zeroes individual weights, which saves
nothing on hardware that still performs dense matrix multiplies — it was run only as a diagnostic,
and the fact that 30% of weights could be removed with a 0.16-point accuracy loss confirmed the model
had capacity to spare. Structured channel reduction, by contrast, physically shrinks the tensors, so
its savings are real on any device. At 192 channels the model gives up 4.5 points of mAP50 for 40.6%
fewer GFLOPs; at 128 channels the loss jumps to 12.8 points, mostly on scallop, the hardest class.

Quantizing the 192-channel weights to INT8 costs essentially nothing: mean SQNR of 43.8 dB and cosine
similarity of 1.000 across all 53 convolutional layers, for a 0.07-point mAP50 drop and a 4×
reduction in weight storage.

### Things that were tried and rejected

Kept here because knowing what *didn't* work saves the next person the experiment:

| Variant | mAP@0.50 | Outcome |
|---|---|---|
| White-balance preprocessing (URPC channel stats + tone curve) | 0.7638 | Rejected — the gain is within noise. After 60 epochs the backbone adapts to the underwater colour distribution regardless of the initial normalisation. |
| MobileViT-XS backbone + FPN-192 | 0.6865 | Rejected — 10.1 points below baseline. The XS backbone shrinks C4 from 640 to 384 channels, but the FPN equalises every level to 192 anyway, so the P2 buffer that actually dominates BRAM is unchanged. All cost, no benefit. |
| URPC 2020 (16,938 train images, own val split) | 0.7995 | Not a rejection — confirms the architecture scales with 5.8× more data without modification. |

---

## How the detector works

```
  640×640×3 RGB
        │
        ▼
┌───────────────────────┐
│  MobileViT-S backbone │   MBConv blocks for cheap local features at high
│  (5.58 M params)      │   resolution; unfold → transformer → fold blocks at
└───────────────────────┘   stages S3/S4/S5 for global context
        │
        │  C1 160×160×64   C2 80×80×96   C3 40×40×128   C4 20×20×640
        ▼
┌───────────────────────┐
│  FPN neck             │   1×1 laterals project every level to 192 channels,
│  (192 ch × 5 levels)  │   top-down upsampling fuses semantics into detail
└───────────────────────┘
        │
        │  P2 160² · P3 80² · P4 40² · P5 20² · P6 10²  (all 192 ch)
        ▼
┌───────────────────────┐
│  RetinaNet head       │   4 stacked 3×3 convs → classification subnet (4
│  (shared across P2–P6)│   classes × 9 anchors) + regression subnet (4 × 9)
└───────────────────────┘
        │
        ▼
    greedy NMS  →  (class, score, x1, y1, x2, y2)
```

MobileViT is the interesting part. CNNs are efficient and data-efficient but have a limited receptive
field; vision transformers see globally but cost O(N²) and need huge datasets. A MobileViT block gets
both: it extracts local features convolutionally, unfolds the feature map into non-overlapping 2×2
patches, runs a standard transformer encoder over those token sequences, then folds the result back
and fuses it with the local branch. Every stage therefore has a global receptive field while keeping
the spatial layout that the following convolutions need.

That matters here because underwater targets are camouflaged against seabed texture, low-contrast in
turbid water, colour-shifted toward blue-green as water absorbs red, and span a wide range of
apparent sizes. Local texture alone is not enough to separate a holothurian from the rock it is lying
on.

---

## Repository layout

The repository is a fork of [MMDetection v2.28.2](https://github.com/open-mmlab/mmdetection) with the
project's own work added on top. **Almost everything under `htdet_project/mmdetection/` that is not
listed below is unmodified upstream MMDetection** — including its `README.md`, its ~670 configs for
other detectors, and the `mmdet/` package itself.

The directories that contain this project's work:

```
data/urpc/                         URPC 2018 dataset, COCO format (2,901 train / 800 val)

htdet_project/mmdetection/
├── configs/htdet/                 all training configs for this project (see below)
├── configs/_base_/datasets/urpc_detection.py    URPC dataset definition
├── custom_modules/                experimental necks and hooks (FA-FPN, UA-FPN, freeze hook)
│
├── fpga_ptq_192/     ★ the deployed design — full pipeline, INT8 weights, 640×640
├── fpga_support/       weight export, PTQ calibration, host app, build scripts
├── weights_ptq/        exported float32 weight binaries consumed by the C testbench
├── ptq_results_192/    PTQ calibration scales and per-layer SQNR report
├── csim_validation/    Python-vs-C comparison harness and the 5 validation images
│
├── fpga_cpp/           earlier full-pipeline C implementation at 256 channels
├── fpga_192/           full-pipeline C implementation at 192 channels, float32 weights
├── w8a32/, w8a8/       quantization-scheme variants of the full pipeline
│
├── fpga_stem/          ── per-module HLS synthesis projects. Each is a self-contained
├── fpga_mbconv_*/         Vitis HLS project with its own fpga_types.h, testbench,
├── fpga_t_blk_s{2,3,4}/   run_hls_*.tcl, and SYNTHESIS_*.md report.
├── fpga_mvit_blk_s*/
├── fpga_fpn_192/
├── fpga_retina_head/
├── fpga_nms/
│
├── ARCHITECTURE_EXPLANATION.md    detailed walkthrough of the three-stage model
├── HTDET_RESEARCH_PAPER.md        paper-style write-up
└── MTP2_Mani_Deep_G/thesis.pdf    full thesis (the authoritative reference)
```

### Which config is which

All under `configs/htdet/`. These are the ones that produced the numbers above:

| Config | What it is |
|---|---|
| `htdet_gpu.py` | Baseline: MobileViT-S + FPN-256 + RetinaNet, 60 epochs, 640×640 |
| `htdet_gpu_low_gflops_224.py` | Channel-224 variant |
| `htdet_gpu_low_gflops_192.py` | **Channel-192 — the deployed model** |
| `htdet_gpu_low_gflops.py` | Channel-128 variant |
| `htdet_gpu_pruned_finetune.py` | Fine-tuning recipe after 30% unstructured pruning |
| `htdet_gpu_whitebal.py` | White-balance preprocessing ablation |
| `htdet_xs_192.py` | MobileViT-XS backbone ablation |

Other files in that directory (`*_fafpn*`, `*_ucafpn*`, `*_kd*`, `*_detr*`, `htdet_effb3*`,
`htdet_convnext*`, `htdet_yolox*`) are exploratory experiments that were not carried forward.

---

## Getting started

### 1. Environment

Needs Python 3.7–3.10, PyTorch with CUDA, and MMCV in the 1.3.17–1.8.0 range (MMDetection 2.x is
*not* compatible with mmcv 2.x).

```bash
conda create -n htdet python=3.8 -y && conda activate htdet
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -U openmim && mim install "mmcv-full>=1.3.17,<1.8.0"
pip install timm

cd htdet_project/mmdetection
pip install -v -e .
```

### 2. Point MMDetection at the dataset

The configs resolve the dataset as `data/urpc/`, relative to the working directory. The data lives at
the repository root but MMDetection runs from `htdet_project/mmdetection/`, so link it:

```bash
cd htdet_project/mmdetection
ln -s ../../data data
```

### 3. Train and evaluate

```bash
# baseline, 256 channels
python tools/train.py configs/htdet/htdet_gpu.py --gpu-id 0

# the deployed 192-channel model
python tools/train.py configs/htdet/htdet_gpu_low_gflops_192.py --gpu-id 0

# evaluate a checkpoint
python tools/test.py configs/htdet/htdet_gpu_low_gflops_192.py \
    work_dirs/htdet_low_gflops_192/epoch_47.pth --eval bbox
```

Checkpoints are written to `work_dirs/` and are **not** committed to this repository — you will need
to train from scratch, or obtain the `.pth` files from the author. Note that
`htdet_gpu_low_gflops_192.py` sets `load_from` to a warm-start checkpoint at
`work_dirs/low_gflops_init/epoch_42_low_gflops_init.pth`; comment that line out to train from the
ImageNet-pretrained backbone instead.

---

## The FPGA path

Four stages take a trained `.pth` to synthesised RTL.

### 1. Quantize and export weights

PyTorch tensors become flat binary files that the C code reads sequentially. Per-layer INT8 scales
come from a min/max calibration pass over 200 validation images, and batch-norm is folded into the
preceding convolution at export time so the hardware does one multiply-add instead of a full BN pass.

Both scripts run from `htdet_project/mmdetection/` and read the trained checkpoint from
`work_dirs/htdet_low_gflops_192/latest.pth`:

```bash
cd htdet_project/mmdetection

# calibrate INT8 scales over 200 validation images → ptq_results_192/ptq_scales.json
python fpga_support/ptq_calibrate_192.py --num-cal 200 --num-det 20 --outdir ptq_results_192

# export INT8 weights + float32 scale/bias metadata binaries
PYTHONPATH=. python3 fpga_ptq_192/run_export_meta.py
```

The export order is load-bearing: the C testbench reads the binaries as one contiguous stream, so any
mismatch between the Python traversal order and the C read order silently assigns weights to the
wrong layers. This caused a bug that produced 0% backbone match and was only found by dumping and
comparing intermediate feature maps.

### 2. Verify in C simulation

`csim_validation/run_multi_image.py` runs each image through PyTorch, then shells out to a compiled
C testbench, then compares the two sets of detections by IoU and score. Compile the testbench first —
it needs the `hls_stubs/` headers so it can build without Vitis installed:

```bash
cd htdet_project/mmdetection/fpga_temp
g++ -O2 -std=c++14 -I. -Ihls_stubs testbench.cpp htdet_top.cpp -o testbench_csim

cd .. && python csim_validation/run_multi_image.py
```

Expect roughly 8–10 minutes per image. Note that this harness is wired to the **float32,
256-channel** pipeline — it loads `configs/htdet/htdet_gpu.py` on the Python side and runs
`fpga_temp/` on the C side. See the limitations section below regarding the weights directory it
expects. The INT8 192-channel design is verified separately through `fpga_ptq_192/testbench.cpp`.

Result across five validation images: **148 of 157 detections matched at IoU ≥ 0.5 (94.3%)**, with
three of the five images at 100%. The two that fall short (89.9% and 93.3%) are dense scenes where
borderline boxes sit right at the NMS threshold and IEEE 754 rounding pushes them across differently
in C than in Python. 94% of matched detections differ by less than 5×10⁻³ in score.

Activations are deliberately kept in float32 during simulation. That way any C-versus-Python
discrepancy is a logic error, not precision loss, which makes debugging tractable.

### 3. Synthesise

```bash
source /tools/Xilinx/Vitis/2022.2/settings64.sh
cd fpga_ptq_192 && vitis_hls -f run_hls_ptq.tcl
```

Target is the Xilinx ZCU102 (`xczu9eg-ffvb1156-2-e`) at 200 MHz. Expect roughly 20–40 minutes for
C-simulation and 2–4 hours for synthesis. Each per-module directory has its own `run_hls_*.tcl` if you
want to synthesise one block in isolation.

### 4. Synthesis results

All nine modules meet timing at 200 MHz on the ZCU102 (274,080 LUT / 548,160 FF / 1,824 BRAM18 /
2,520 DSP available):

| Module | BRAM18 | DSP | FF | LUT | Latency |
|---|---|---|---|---|---|
| `mbconv_80_top` | 1,500 | 253 | 79,601 | 65,480 | 780 ms |
| `mbconv_40_top` | 900 | 313 | 147,021 | 111,967 | 252 ms |
| `mbconv_20_top` | 600 | 244 | — | — | 104 ms |
| `t_blk_s2_top` | 1,256 | 260 | 317,871 | 269,309 | 470 ms |
| `t_blk_s3_top` | 426 | 338 | 288,559 | 262,703 | 325 ms |
| `t_blk_s4_top` | 274 | 357 | 276,387 | 270,700 | 90 ms |
| `fpn_full_top` | 2,900 | 29 | 13,014 | 22,405 | 4.63 s |
| `retina_head_top` | 192 | 35 | 78,762 | 148,671 | 1.721 s |
| `nms_top` | 41 | 14 | 6,759 | 12,831 | 28.3 ms |

Summed sequentially this comes to roughly **8.4 s per frame**. Read the caveats in the next section
before quoting that number.

HLS pragmas are what make this tractable at all — `PIPELINE`, `ARRAY_PARTITION`, and `UNROLL`
together buy 7–21× over the same C code with all directives removed. Two findings were worth the
effort:

- **The FPN was halved by fixing an interaction between tiling and the pipeline target.** Its inner
  pipeline II is pinned at 5 by the DSP's latency-4 dependency chain. Raising `OC_TILE` from 4 to 8
  halves the outer trip count, but only if `PIPELINE II=5` is stated explicitly: with `II=1` and 8
  accumulator chains, HLS allocates a single DSP and quietly relaxes to II=8, cancelling the gain.
  Forcing II=5 makes it allocate two DSPs and the full 2× materialises — 18.5 s down to 9.3 s, and
  4.63 s at `OC_TILE=16`.
- **BRAM, not compute, is the binding constraint.** `fpn_full_top` needs 2,900 BRAM18 tiles against
  1,824 available, almost entirely for the 192×80×80 lateral buffer — while using 29 of 2,520 DSPs.

---

## Project status and known limitations

**This is a validated functional proof-of-concept, not a deployable real-time system.** Being
specific about where it stands:

- **Nothing has run on physical hardware.** Every number here is from C-simulation and HLS
  C-synthesis. Place-and-route in Vivado has not been done, so timing closure after routing, real
  on-board latency, and actual power draw are all unconfirmed.
- **The design does not currently fit.** `fpn_full_top` alone requests 2,900 BRAM18 tiles on a device
  with 1,824. Streaming or tiling the P2 buffer is required before placement is possible.
- **8.4 s per frame is ~100× too slow for the target.** Onboard ROV control loops need 33–100 ms.
  Closing that gap needs spatial tiling, deeper unrolling, and a move to full W8A8 fixed-point to
  eliminate the remaining FP32 activation paths.
- **Read the per-module latencies carefully — they are not all at the same input resolution.** The
  MBConv and FPN projects (`fpga_mbconv_*`, `fpga_fpn_192`) declare `INPUT_H/W = 320`, and the
  transformer blocks use `TB_SEQ = 400`, which also corresponds to a 320×320 input. Only the
  full-pipeline design in `fpga_ptq_192` is built at 640×640. So the 8.4 s total is a sum over modules
  configured at 320×320, and should not be read as a 640×640 end-to-end figure.
- **`retina_head_top` is one pyramid level, not the whole head.** `fpga_retina_head/fpga_types.h` sets
  `TEST_H = TEST_W = 10`; the 1.721 s covers a single 10×10 level, not all five. The thesis notes a
  predicted ~9× improvement to ≈500 ms from removing an erroneous `KH/KW` unroll that inflates weight
  BRAM traffic.
- **W8A8 is the eventual target; W8A32 is what is validated.** Full INT8 reaches the same 72.80%
  mAP50, but the C/HLS implementation currently keeps activations in FP32. The fixed-point `ap_fixed`
  type mapping is specified in the thesis (Table 5.4) but not yet applied.
- **Activations use exact `expf`/`sqrtf` in simulation.** Piecewise-linear approximations for SiLU
  and sigmoid are written but commented out in `fpga_utils.h`; LayerNorm still needs a fixed-point
  integer square root for synthesis.
- **The INT8 weight binaries are not in the repository.** `fpga_ptq_192/testbench.cpp` reads from
  `ptq_results_192/ptq_int8_weights/`, which is not committed — only the calibration scales and the
  report are. `weights_ptq/` contains float32 weights (4,937,632 backbone elements at 4 bytes each),
  despite the directory name. Re-run the export step in stage 1 to regenerate the INT8 set.
- **Some tooling points at paths that no longer exist.** `csim_validation/run_multi_image.py` invokes
  `./testbench_csim` with a weights directory of `../weights/`, but the committed weights are in
  `weights_ptq/` — either rename, symlink, or edit the call. Similarly, `fpga_support/README.md`
  refers to an `fpga_rtl_src/` directory that is not in the repository.

---

## Reference material

| Document | Contents |
|---|---|
| [`MTP2_Mani_Deep_G/thesis.pdf`](htdet_project/mmdetection/MTP2_Mani_Deep_G/thesis.pdf) | The authoritative reference — architecture, all experiments, C/HLS conversion, synthesis |
| [`ARCHITECTURE_EXPLANATION.md`](htdet_project/mmdetection/ARCHITECTURE_EXPLANATION.md) | Component-by-component walkthrough of backbone, neck, and head |
| [`fpga_support/quantization_reference.md`](htdet_project/mmdetection/fpga_support/quantization_reference.md) | Quantization scheme and calibration details |
| [`ptq_results/PTQ_COMPARISON.md`](htdet_project/mmdetection/ptq_results/PTQ_COMPARISON.md) | Float32 vs quantized accuracy comparison |
| [`w8a8/W8A8_COMPARISON.md`](htdet_project/mmdetection/w8a8/W8A8_COMPARISON.md) | W8A32 vs W8A8 analysis |
| `fpga_*/SYNTHESIS_*.md` | Per-module synthesis runs, pragma experiments, and resource reports |

---

## Acknowledgements and licence

Built on [MMDetection](https://github.com/open-mmlab/mmdetection) (Apache 2.0), [TIMM](https://github.com/huggingface/pytorch-image-models),
PyTorch, and Xilinx Vitis HLS. The vendored MMDetection tree retains its original Apache 2.0
`LICENSE`.

Key references:

1. G. Chen, Z. Mao, K. Wang, J. Shen. *HTDet: A Hybrid Transformer-Based Approach for Underwater Small
   Object Detection.* Remote Sensing 15(4), 1076, 2023.
2. S. Mehta, M. Rastegari. *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision
   Transformer.* ICLR 2022.
3. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. *Focal Loss for Dense Object Detection.* ICCV 2017.
4. T.-Y. Lin et al. *Feature Pyramid Networks for Object Detection.* CVPR 2017.
