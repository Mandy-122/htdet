# Float32 CSIM vs Python Reference — GOPR0293_10229

**Date:** 2026-05-20  
**CSIM type:** Float32 (all weights and activations in float32)  
**Python reference:** mmdetection PyTorch model (float32, URPC dataset)  
**Image:** GOPR0293_10229 (640×640)

## Summary

| Metric | Python (mmdetection) | CSIM (Vitis HLS float32) |
|--------|---------------------|--------------------------|
| Total detections | 19 | 67 |
| Score range | 0.224 – 0.996 | 0.200 – 0.936 |
| Top score | 0.995941 | 0.935511 |
| Score threshold | ~0.05 | ~0.20 |

## Image Identification

Image was identified as GOPR0293_10229 by matching box coordinates:
- Python det #4: class 1, score=0.9703, box=[495.89, 19.16, 525.81, 50.03]
  → CSIM det[0]: echinus, score=0.9355, box=[498.09, 20.23, 528.80, 51.21]
- Python det #5: class 1, score=0.9555, box=[517.14, 53.85, 547.89, 81.62]
  → CSIM det[1]: echinus, score=0.5396, box=[515.94, 54.11, 551.39, 85.46]

## Detection-by-Detection Match

| Python Rank | Python Class | Python Score | Python Box | CSIM Class | CSIM Score | CSIM Box | Status |
|-------------|-------------|-------------|------------|-----------|-----------|---------|--------|
| #1 | echinus | 0.9959 | [523.59, 209.68, 557.50, 240.24] | holothurian | 0.5009 | [523.35, 210.78, 560.05, 240.89] | Class mismatch + score −49% |
| #2 | starfish | 0.9833 | [236.07, 100.26, 268.44, 130.33] | — | — | — | Missing / buried below threshold |
| #3 | echinus | 0.9794 | [477.91, 269.01, 504.28, 295.83] | — | — | — | Missing |
| #4 | echinus | 0.9703 | [495.89, 19.16, 525.81, 50.03] | echinus | 0.9355 | [498.09, 20.23, 528.80, 51.21] | ✓ Match, score −3.5% |
| #5 | echinus | 0.9555 | [517.14, 53.85, 547.89, 81.62] | echinus | 0.5396 | [515.94, 54.11, 551.39, 85.46] | ✓ Box match, score −43% |
| #6 | starfish | 0.9037 | [566.86, 150.78, 634.97, 182.13] | — | — | — | Missing |
| #7 | echinus | 0.8884 | [31.84, 300.65, 59.12, 326.98] | echinus | 0.3299 | [35.02, 300.35, 65.20, 332.12] | ✓ Box match, score −63% |
| #8 | echinus | 0.8797 | [145.73, 296.93, 184.24, 332.04] | — | — | — | Possibly buried |

## Bugs Identified

### Bug 1: Classification Score Under-Estimation (Critical)
CSIM sigmoid/classification outputs are significantly and non-uniformly lower than Python:

| Detection | Python Score | CSIM Score | Delta |
|-----------|-------------|-----------|-------|
| echinus @ [498, 20] | 0.9703 | 0.9355 | −3.5% |
| echinus @ [517, 54] | 0.9555 | 0.5396 | −43% |
| echinus @ [523, 210] | 0.9959 | 0.4818 (+ wrong class) | −52% |
| echinus @ [35, 300] | 0.8884 | 0.3299 | −63% |

The non-uniform deflation rules out a simple scale factor error.
**Likely cause:** Sigmoid approximation error in HLS classification head, or numerical issue in the exp() computation for larger logit magnitudes.

### Bug 2: Duplicate Boxes / NMS Failure (Critical)
CSIM detections [2] and [3] have **identical boxes** `[523.35, 210.78, 560.05, 240.89]`
with different classes (holothurian and echinus). Python NMS suppresses this to one detection.

- CSIM det[2]: holothurian, score=0.5009, box=[523.35, 210.78, 560.05, 240.89]
- CSIM det[3]: echinus,     score=0.4818, box=[523.35, 210.78, 560.05, 240.89]

**Likely cause:** NMS is run per-class (correct) but duplicate anchors from the same spatial location
on different FPN levels are not being suppressed by inter-class IoU checks.

### Bug 3: Spurious Large Bounding Boxes (Major)
CSIM produces many large boxes that Python NMS eliminates entirely:

| CSIM Box | Area | Score | Class |
|---------|------|-------|-------|
| [43.47, 0, 640, 321.51] | 596×322 | 0.3665 | holothurian |
| [41.42, 53.49, 640, 506.90] | 599×453 | 0.3641 | scallop |
| [69.04, 0, 421.12, 178.86] | 352×179 | 0.3544 | scallop |
| [0, 0, 640, 258.59] | 640×259 | 0.2521 | scallop |

**Likely cause:** Regression delta `exp(dw)` / `exp(dh)` decoding is not clamped,
allowing large width/height multipliers. Python clips delta values before exp (e.g., max=4.135).
Check `retina_head.h` decode_boxes() for the clamp before exp.

### Bug 4: Missing High-Confidence Detections
Python's top-2 detections (starfish at 0.983, echinus at 0.979) are absent from CSIM output.
Combined with Bug 1, their CSIM scores may fall below the 0.2 threshold and get filtered out.

## Files in this Directory

- `csim_detections.txt` — raw CSIM output (67 detections, float32 CSIM)
- `comparison_report.md` — this file

## Reference Files

- Python reference: `../../../multi_image/GOPR0293_10229/python_detections.txt` (19 detections)
- W8A32 CSIM reference: `../../w8a32/GOPR0293_10229/csim_detections.txt` (19 detections)
- W8A8 CSIM reference: `../../w8a8/GOPR0293_10229/csim_detections.txt` (19 detections)
