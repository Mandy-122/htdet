# HTDet PTQ — Viva Questions & Answers

**Model:** RetinaNet + MobileViT-S + FPN (URPC underwater detection, 4 classes)  
**Topic:** Post-Training Quantization (PTQ) for FPGA deployment  
**Strategy:** W8A32 — INT8 per-channel symmetric weights, FP32 activations

---

## SECTION 1: Fundamentals

**Q: What is Post-Training Quantization (PTQ)?**
> Quantization reduces the numerical precision of a model's weights and/or activations — from 32-bit floats down to 8-bit integers. Post-Training means you do this after training is complete, without any retraining. You load the trained checkpoint, collect statistics from a small calibration set, compute quantization scales, and produce a quantized model — all in minutes.

**Q: What is the difference between PTQ and QAT?**
> PTQ (Post-Training Quantization) runs after training — fast (minutes), no GPU training needed, slightly more accuracy loss. QAT (Quantization-Aware Training) simulates quantization during training so the model learns to compensate for rounding errors — slower (days of retraining), marginally better accuracy. For this model, PTQ gave only −0.2% mAP loss, making QAT unnecessary.

**Q: Why quantize at all — what is the motivation?**
> Three reasons specific to FPGA deployment: (1) FPGAs have limited on-chip BRAM — INT8 weights use 4× less BRAM than FP32. (2) DSP48E2 primitives on Xilinx FPGAs pack 4 INT8 MACs into one DSP vs. 3 DSPs for a single FP32 multiply — ~12× DSP efficiency gain for full INT8. (3) Memory bandwidth is the main bottleneck — INT8 fetches 4 weights per memory access vs. 1 for FP32, giving 4× effective throughput.

**Q: Why INT8 specifically — why not INT4 or INT16?**
> INT8 is the hardware sweet spot: it fits natively into DSP48E2 primitives on Xilinx FPGAs (4 INT8 MACs per DSP). SQNR analysis shows all 53 layers quantize cleanly to 8 bits with SQNR > 36 dB. INT4 would cause larger accuracy loss especially for the dynamic-range-challenged early depthwise layers. INT16 wastes DSP resources with negligible accuracy benefit over INT8.

---

## SECTION 2: The Quantization Math

**Q: How does symmetric per-channel INT8 quantization work mathematically?**
> For each output channel `c` of a Conv2d layer, compute the scale:
> ```
> scale[c] = max(|weight[c, :, :, :]|) / 127
> ```
> Quantize (float → INT8):
> ```
> w_int8[c] = round(weight[c] / scale[c]).clamp(−128, 127)
> ```
> Dequantize (INT8 → float, for simulation):
> ```
> w_dequant[c] = w_int8[c] × scale[c]
> ```
> 127 is used as the denominator (not 128) to keep the range symmetric. The rounding introduces a small error — this is the quantization noise.

**Q: What is SQNR and why does it matter?**
> Signal-to-Quantization-Noise Ratio measures how much of the weight signal is preserved vs. the rounding error introduced:
> ```
> SQNR = 10 × log10( mean(w²) / mean((w − w_dequant)²) )
> ```
> 40 dB means the error power is 10,000× smaller than the signal — invisible to the model. Layers below ~30 dB need special treatment (mixed precision or finer quantization). All 53 layers in this model are safely above 36 dB (min 36.8, mean 43.3, max 49.1 dB).

**Q: What is Cosine Similarity in this context and what do your values mean?**
> Cosine similarity measures the angular difference between the original and quantized weight vectors — 1.0 means they point in exactly the same direction (same relative pattern, possibly different scale). Values of 0.9999–1.0000 across all layers mean the quantized weights have essentially identical directional structure to the originals. Even the worst layer (0.9999) is negligibly different.

**Q: Why per-channel quantization instead of per-tensor?**
> Per-tensor uses one scale for the entire weight matrix. If one channel has weights in [−100, 100] and another in [−0.1, 0.1], a single scale of ~0.79 leaves the small-weight channel with almost all values rounded to 0 or ±1, destroying it. Per-channel gives each output channel its own scale, preserving small-magnitude channels faithfully. This is especially important in depthwise separable convolutions where channel magnitudes vary widely.

---

## SECTION 3: Implementation Details

**Q: What is the fake-quantize approach and why use it?**
> Instead of storing values as actual INT8 bytes, you round to the INT8 range then immediately convert back to float, introducing the same rounding error as real INT8 would:
> ```python
> w_int8  = round(w / scale).clamp(−128, 127)  # quantize
> w_deq   = w_int8 × scale                      # dequantize back to float
> ```
> The model runs with `w_deq` instead of `w`. Because you run the full forward pass with these fake-quantized weights, the error propagates through all layers realistically — giving accurate prediction of real INT8 inference accuracy without needing custom INT8 CUDA kernels.

**Q: What is calibration and why is it needed?**
> Calibration runs a small set of representative images (200 in this case) through the float model with forward hooks attached to every Conv2d layer. Each hook records the running min and max of the layer's output activations. These ranges are used to: (1) compute activation quantization scales for W8A8, and (2) determine how many integer bits each layer's `ap_fixed` type needs for FPGA synthesis. Without calibration, you would have to guess activation ranges, which can cause overflow (clipping) or waste precision bits.

**Q: How do you choose the number of integer bits for ap_fixed on the FPGA?**
> From calibration data — collect the max absolute activation value for each layer, then:
> ```
> INT_BITS = ceil(log2(ActMax + 1)) + 1
> ```
> The +1 is for the sign bit. Examples from this model:
> - `retina_reg` output: ActMax = 0.88 → INT_BITS = 2 → `ap_fixed<8,2>` (6 fractional bits)
> - Most backbone 1×1 convs: ActMax ~13 → INT_BITS = 5 → `ap_fixed<8,5>` (3 fractional bits)
> - Early depthwise convs: ActMax = 232 → INT_BITS = 9 → `ap_fixed<8,9>` (only −1 fractional bits — a known outlier)

**Q: Why do you keep transformer layers in FP32?**
> The attention mechanism computes Q·Kᵀ (dot products of query and key vectors) then applies softmax. Softmax is extremely sensitive to input range — a small perturbation can dramatically shift the attention distribution. Additionally, attention scores have highly dynamic range that is difficult to calibrate statically. FP32 for these layers adds negligible overhead (they are a small fraction of total compute) while protecting the most range-sensitive operations. The MobileViT-S transformer Linear layers (attn.qkv, attn.proj, mlp.fc*) are all kept FP32.

---

## SECTION 4: BN Fusion

**Q: What is Batch Normalization fusion and why do you need it?**
> Every Conv2d in the backbone is followed by BatchNorm (BN). BN applies a per-channel affine transform to the conv output: `y = (x − μ) / √(σ² + ε) × γ + β`. For FPGA deployment, you fold this into the conv weight before quantizing:
> ```
> w_fused[c] = w_conv[c] × (γ[c] / √(σ²[c] + ε))
> fused_bias[c] = β[c] − μ[c] × γ[c] / √(σ²[c] + ε)
> ```
> At inference the FPGA just does `conv(input, w_fused) + fused_bias` — BN runs for free (no extra multiply). This removes one operator from the datapath and reduces latency.

**Q: Why do you use different quantization bases for the Python checkpoint vs. the FPGA binary files?**
> The Python `.pth` checkpoint (used for mAP evaluation) must work with MMDetection's standard model graph where Conv and BN are separate layers. So you fake-quantize the *raw* conv weight using a scale from the raw weight, and let BN run normally after. The FPGA INT8 binary files target synthesized hardware where BN is pre-fused — so you fold BN into the conv weight first (`w_fused`), then quantize. Using the wrong base in either case causes errors: using fused-weight scale to quantize raw weights produces scales ~10–100× too large, collapsing all conv outputs to near zero (this was an actual bug found and fixed during development).

**Q: What is the fused bias and where is it stored?**
> `fused_bias[c] = β[c] − μ[c] × γ[c] / √(σ²[c] + ε)`. This replaces both the BN mean subtraction and beta shift. It is stored per-layer in `ptq_int8_weights/scales.json` under the key `layername.bn_bias`. It is a float32 vector (one value per output channel) added to the conv output after dequantizing the INT8 weights.

---

## SECTION 5: The Outlier Problem

**Q: What are the activation range outliers you found and what do they mean for FPGA design?**
> Two early depthwise convolutions have unusually large activation ranges:
> - `backbone.model.stages_0.0.conv2_kxk`: ActMax = 232, needs `ap_fixed<8,9>`
> - `backbone.model.stages_1.0.conv2_kxk`: ActMax = 196, needs `ap_fixed<8,9>`
> `ap_fixed<8,9>` has 9 integer bits and −1 fractional bits, meaning each step represents 2 units — very coarse. These are depthwise separable convolutions in early MBConv blocks with a 4× channel expand ratio. They see large pre-BN values because they operate in the expanded channel space before projection back down.

**Q: How would you handle those two outlier layers in a real W8A8 design?**
> Three options: (1) **Mixed precision** — keep just those two layers in FP32, quantize everything else INT8. This costs minimal BRAM but avoids the precision problem. (2) **Per-channel activation quantization** — instead of one scale per tensor, use one scale per channel, which handles the wide inter-channel variance better. (3) **Input clipping** — clip activations to a narrower range (e.g., ±64) before quantizing, accepting a small accuracy loss in exchange for more fractional bits. Option 1 is simplest and most practical here.

---

## SECTION 6: Results & Accuracy

**Q: What were your PTQ accuracy results and how do you interpret them?**
> Float32 baseline (epoch 60): mAP = 0.408, mAP@50 = 0.755.
> PTQ W8A32: mAP = 0.407, mAP@50 = 0.753.
> Drop: −0.001 mAP (−0.2%), −0.002 mAP@50 (−0.3%). This is negligible — well within the noise of evaluation variance across runs. The model is production-ready without any retraining.

**Q: Why did the detection count go up (+3.3%) while mAP went slightly down?**
> This is counterintuitive but explainable. Quantization noise acts as a very slight regularizer on the weight values, occasionally pushing borderline scores above the 0.20 threshold — so more boxes pass the score filter. However, those extra detections are lower quality (lower IoU with ground truth), which hurts precision but inflates raw count. mAP penalizes false positives through the precision-recall curve, so it decreases even as raw count increases.

**Q: What would you do if the PTQ accuracy drop was unacceptable?**
> Three options in order of increasing effort: (1) **Mixed precision** — keep the most sensitive layers in FP32. (2) **Per-channel activation quantization** — finer granularity for activation scales. (3) **QAT** — fine-tune for a few epochs with fake-quantization enabled so the model learns to compensate for rounding error. Given the −0.2% mAP drop here, none of these are needed.

---

## SECTION 7: FPGA Hardware Impact

**Q: How does W8A32 reduce FPGA latency if the MAC count is unchanged?**
> The bottleneck is weight fetch bandwidth from BRAM to compute units, not raw MAC throughput. Each clock cycle, BRAM delivers a fixed number of bits. FP32 weights are 32 bits each → 1 weight per access. INT8 weights are 8 bits each → 4 weights per access. The convolution PEs receive 4× more operands per memory cycle, staying busier and reducing stall time. Estimated FPGA latency drops from ~500 ms (FP32) to ~200 ms (W8A32) — a ~2.5× speedup without touching the MAC architecture.

**Q: What additional gains would W8A8 (full INT8) give over W8A32?**
> W8A8 adds activation quantization on top of weight quantization. Benefits: (1) DSP savings — DSP48E2 packs 4 INT8 MACs into one primitive vs. 3 DSPs for a single FP32 multiply, ~12× fewer DSPs per PE. (2) Activation buffer savings — 4× smaller feature map buffers in BRAM/URAM. (3) MAC throughput — 4× more INT8 operations per DSP per clock, pushing estimated latency to ~50–80 ms. The cost is slightly higher accuracy risk, especially for the two outlier depthwise layers.

**Q: How much BRAM does quantization save?**
> Conv weight storage drops from 49.7 MB (FP32) to 9.5 MB (INT8) — a 5.2× reduction. In BRAM36 terms (36 Kb each): ~11,310 BRAM36 for FP32 vs. ~2,160 BRAM36 for INT8 — an 81% reduction. This is significant because mid-range Xilinx FPGAs (e.g., ZCU104) have ~312 BRAM36 on-chip; storing all weights on-chip requires off-chip DDR for FP32, whereas a selective INT8 approach can fit key weights on-chip, eliminating DDR latency.

**Q: What is the deployment pipeline from Python to FPGA?**
> ```
> epoch_60.pth (FP32, 96 MB)
>       │
>       ▼ ptq_calibrate.py
>       │
>       ├─► ptq_model.pth          → MMDetection eval (mAP verification)
>       ├─► ptq_int8_weights/*.bin → HLS testbench weight loading
>       ├─► scales.json            → per-channel scales + BN biases
>       └─► ptq_scales.json        → activation ranges
>                  │
>                  ▼ fpga_types.h  (ap_fixed<8,N> per layer)
>             HLS synthesis (Vitis HLS)
>                  │
>                  ▼
>             Synthesized IP core (Vivado implementation)
> ```

---

## SECTION 8: Model Architecture Context

**Q: Why is MobileViT-S a good backbone for FPGA deployment?**
> MobileViT-S is a hybrid CNN-Transformer: depthwise separable MBConv blocks for local features (low parameter count, low FLOPs) combined with lightweight MobileViT blocks for global attention (small token sequences, low transformer overhead). It has only 12.4 M parameters and 198.9 GFLOPs — compact enough for FPGA deployment while matching heavier CNN backbones in accuracy. The depthwise convolutions are especially hardware-friendly: they have low channel-to-channel connectivity (group convolutions with groups = channels), making them easier to pipeline in HLS.

**Q: Why RetinaNet as the detection head?**
> RetinaNet is a single-stage detector — no region proposal network (RPN) stage, no two-stage processing. Single-stage detectors are much simpler to implement in HLS because the control flow is a fixed pipeline: backbone → FPN → parallel cls/reg heads → NMS. There are no dynamic data-dependent branches. The anchor-based design also gives fixed-size output tensors per FPN level, making buffer allocation in `fpga_types.h` straightforward.

**Q: What is the FPN and why does it matter for multi-scale detection?**
> Feature Pyramid Network (FPN) takes multi-scale backbone features (C1–C4 at stride 4, 8, 16, 32) and combines them top-down with lateral connections to produce P2–P6 feature maps, all with 256 channels. This allows the detector to use fine spatial resolution for small objects (P2: 160×160) and coarse resolution with large receptive field for large objects (P6: 10×10). For underwater detection, objects range from small sea urchins to large starfish — FPN is essential for handling this scale variation.

---

## SECTION 9: Dataset & Task Context

**Q: What is the URPC dataset and what makes underwater detection challenging?**
> URPC (Underwater Robot Picking Contest) is a dataset of underwater images with 4 classes: holothurian (sea cucumber), echinus (sea urchin), scallop, and starfish. Challenges include: low contrast (similar colors to seafloor), turbid water causing blur and color distortion, varying illumination, occlusion between organisms, and extreme scale variation. The model achieved mAP@50 = 0.755 on the val set — strong performance given these conditions.

**Q: Why use a score threshold of 0.20 for the FPGA testbench?**
> Analysis of score distributions across 5 representative images showed a natural gap between 0.10 and 0.15 — very few predictions fell in this range, so it represents noise rather than real detections. 0.20 was chosen as it: (1) matches the PTQ evaluation threshold (consistent comparison), (2) eliminates the noise below 0.15, and (3) captures all meaningful detections that 0.30 would also capture plus a few more borderline cases that were verified as correct. The value is defined in `fpga_types.h` as `SCORE_THR 0.20f`.

---

## SECTION 10: Comparison Questions

**Q: How does your PTQ compare to the state of the art for underwater detection?**
> The float baseline achieves mAP = 0.408 (mAP@50 = 0.755) at only 198.9 GFLOPs and 12.4 M parameters — efficient for the accuracy level. The PTQ W8A32 model preserves 99.8% of this accuracy while reducing conv weight storage by 5.2×. This is competitive with quantization results reported for similar lightweight detectors on benchmark datasets (typical PTQ mAP drop for INT8 is 0.3–1.5%).

**Q: What would you improve if you had more time?**
> (1) W8A8 simulation — the `act_scale` values are already in `ptq_scales.json`, ready to implement. Expected additional mAP drop: 0.5–1.5%. (2) Mixed precision for the two depthwise outlier layers — keep `stages_0.0.conv2_kxk` and `stages_1.0.conv2_kxk` in FP32. (3) INT8 testbench mode — load `backbone_int8.bin` with per-channel scales and BN biases (now in `scales.json`) to run C-sim in W8A32 mode for end-to-end numerical validation before HLS synthesis. (4) Actual HLS synthesis and implementation in Vivado to get real latency and resource numbers rather than estimates.
