/*
 * htdet_top.cpp
 * Top-level HLS kernel for HTDet object detection on FPGA.
 *
 * THIS IS THE FILE THAT VITIS HLS SYNTHESIZES TO RTL.
 * Set top function: htdet_inference
 *
 * Pipeline: MobileViT-S Backbone → FPN Neck → RetinaNet Head
 *
 * AXI interface allocation (each bundle = separate AXI4 master port):
 *   gmem0 : input image
 *   gmem1 : backbone weights (~5MB)
 *   gmem2 : FPN weights (~5.2MB)
 *   gmem3 : cls head stacked conv weights (~4.7MB)
 *   gmem4 : reg head stacked conv weights (~4.7MB)
 *   gmem5 : cls/reg prediction head weights (small)
 *   gmem6 : detections output
 *   gmem7 : num_dets output
 *   control: s_axilite for start/done handshake
 *
 * On-chip BRAM/URAM budget (xczu9eg, 640×640 input):
 *   URAM: stem(800KB) + s0(1.6MB) + lat1(3.3MB) + up2(3.3MB)  ≈ 9MB  (280 × 288Kb = 10MB avail)
 *   BRAM: c1(820KB) + c2(307KB) + c3(102KB) + c4(128KB)
 *         + lat2-4(1.1MB) + p3-6(1.1MB) + stacked conv tmp(3.3MB) + misc ≈ 7MB  (4MB avail)
 *   → stacked conv tmp (p2 scale) also goes to URAM
 *
 * NOTE: For the first synthesis pass, use a smaller input (e.g., 224×224).
 *       Adjust INPUT_H/INPUT_W in fpga_types.h accordingly.
 */

#include "fpga_types.h"
#include "fpga_utils.h"
#include "mobilevit_backbone.h"
#include "fpn_neck.h"
#include "retina_head.h"

// ============================================================
// TOP-LEVEL KERNEL
// All weight arrays live in DDR and are streamed on demand.
// On-chip buffers hold the backbone outputs and FPN outputs.
// ============================================================
extern "C"
void htdet_inference(
    // Input image: 3 × INPUT_H × INPUT_W, CHW, float16 (ap_fixed<16,8>)
    const input_t   image      [INPUT_C * INPUT_H * INPUT_W],

    // Backbone weights (all stages, flat float16 stream)
    const weight_t  backbone_w [],   // ~5.7M params × 2B ≈ 11.4MB

    // FPN neck weights
    const weight_t  fpn_w      [],   // ~2.6M params × 2B ≈ 5.2MB

    // Head stacked conv weights (cls branch, 4 layers)
    const weight_t  cls_conv_w [],   // 4 × 256*256*9+512 ≈ 4.7MB

    // Head stacked conv weights (reg branch, 4 layers)
    const weight_t  reg_conv_w [],   // same ≈ 4.7MB

    // Head prediction conv weights + biases
    const weight_t  cls_pred_w [ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9],
    const bias_t    cls_pred_b [ANCHORS_PER_LOC * NUM_CLASSES],
    const weight_t  reg_pred_w [ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9],
    const bias_t    reg_pred_b [ANCHORS_PER_LOC * 4],

    // Outputs
    Detection       detections [MAX_DETS],
    int*            num_dets
) {
    // ---- AXI Memory-Mapped Interfaces ----
    #pragma HLS INTERFACE m_axi port=image       offset=slave bundle=gmem0  depth=IMAGE_ELEMS
	#pragma HLS INTERFACE m_axi port=backbone_w  offset=slave bundle=gmem1  depth=4938784
	#pragma HLS INTERFACE m_axi port=fpn_w       offset=slave bundle=gmem2  depth=2600960
    #pragma HLS INTERFACE m_axi port=cls_conv_w  offset=slave bundle=gmem3  depth=2361344
    #pragma HLS INTERFACE m_axi port=reg_conv_w  offset=slave bundle=gmem4  depth=2361344
    #pragma HLS INTERFACE m_axi port=cls_pred_w  offset=slave bundle=gmem5  depth=82944
    #pragma HLS INTERFACE m_axi port=cls_pred_b  offset=slave bundle=gmem5  depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_w  offset=slave bundle=gmem5  depth=82944
    #pragma HLS INTERFACE m_axi port=reg_pred_b  offset=slave bundle=gmem5  depth=36
    #pragma HLS INTERFACE m_axi port=detections  offset=slave bundle=gmem6  depth=100
    #pragma HLS INTERFACE m_axi port=num_dets    offset=slave bundle=gmem7  depth=1

    // s_axilite control registers (start/done handshake from PS)
    #pragma HLS INTERFACE s_axilite port=return      bundle=control
    #pragma HLS INTERFACE s_axilite port=image       bundle=control
    #pragma HLS INTERFACE s_axilite port=backbone_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=fpn_w       bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_conv_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_conv_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_pred_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_pred_b  bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_pred_w  bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_pred_b  bundle=control
    #pragma HLS INTERFACE s_axilite port=detections  bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dets    bundle=control

    // ---- On-chip output buffers for backbone (BRAM) ----
    static act_t c1[C1_CH * C1_H * C1_W];   // [64,  160, 160] = 6.4MB
    static act_t c2[C2_CH * C2_H * C2_W];   // [96,   80,  80] = 2.4MB
    static act_t c3[C3_CH * C3_H * C3_W];   // [128,  40,  40] = 820KB
    static act_t c4[C4_CH * C4_H * C4_W];   // [640,  20,  20] = 1MB
    #pragma HLS RESOURCE variable=c1 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c2 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c3 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c4 core=RAM_T2P_BRAM

    // ---- On-chip output buffers for FPN ----
    static act_t p2[FPN_OUT_CH * P2_H * P2_W];   // [256, 160, 160] = 26MB → URAM
    static act_t p3[FPN_OUT_CH * P3_H * P3_W];   // [256,  80,  80] = 6.5MB
    static act_t p4[FPN_OUT_CH * P4_H * P4_W];   // [256,  40,  40] = 1.6MB
    static act_t p5[FPN_OUT_CH * P5_H * P5_W];   // [256,  20,  20] = 410KB
    static act_t p6[FPN_OUT_CH * P6_H * P6_W];   // [256,  10,  10] =  102KB
    #pragma HLS RESOURCE variable=p2 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p3 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p4 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p5 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p6 core=RAM_T2P_BRAM

    // ===========================================================
    // STAGE 1: Backbone  (image → C1, C2, C3, C4)
    // ===========================================================
    mobilevit_backbone(image, backbone_w, c1, c2, c3, c4);

    // ===========================================================
    // STAGE 2: FPN Neck  (C1..C4 → P2..P6)
    // ===========================================================
    fpn_neck(c1, c2, c3, c4, fpn_w, p2, p3, p4, p5, p6);

    // ===========================================================
    // STAGE 3: RetinaNet Head  (P2..P6 → detections)
    // ===========================================================
    retina_head(
        p2, p3, p4, p5, p6,
        cls_conv_w, reg_conv_w,
        cls_pred_w, cls_pred_b,
        reg_pred_w, reg_pred_b,
        detections, num_dets
    );
}

// ============================================================
// OPTIONAL DEBUG KERNELS
// These allow layer-by-layer verification during co-simulation.
// Comment out for final synthesis if resource-constrained.
// ============================================================

extern "C"
void htdet_backbone_only(
    const input_t  image[INPUT_C * INPUT_H * INPUT_W],
    const weight_t backbone_w[],
    act_t c1[C1_CH * C1_H * C1_W],
    act_t c2[C2_CH * C2_H * C2_W],
    act_t c3[C3_CH * C3_H * C3_W],
    act_t c4[C4_CH * C4_H * C4_W]
) {
    #pragma HLS INTERFACE m_axi port=image      offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=backbone_w offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=c1         offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=c2         offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=c3         offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi port=c4         offset=slave bundle=gmem5
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    mobilevit_backbone(image, backbone_w, c1, c2, c3, c4);
}

extern "C"
void htdet_fpn_only(
    const act_t c1[C1_CH * C1_H * C1_W],
    const act_t c2[C2_CH * C2_H * C2_W],
    const act_t c3[C3_CH * C3_H * C3_W],
    const act_t c4[C4_CH * C4_H * C4_W],
    const weight_t fpn_w[],
    act_t p2[FPN_OUT_CH * P2_H * P2_W],
    act_t p3[FPN_OUT_CH * P3_H * P3_W],
    act_t p4[FPN_OUT_CH * P4_H * P4_W],
    act_t p5[FPN_OUT_CH * P5_H * P5_W],
    act_t p6[FPN_OUT_CH * P6_H * P6_W]
) {
	#pragma HLS bind_storage variable=c1 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=c2 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=c3 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=c4 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=p2 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=p3 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=p4 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=p5 type=RAM_T2P impl=BRAM
	#pragma HLS bind_storage variable=p6 type=RAM_T2P impl=BRAM
    #pragma HLS INTERFACE m_axi port=fpn_w offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    fpn_neck(c1, c2, c3, c4, fpn_w, p2, p3, p4, p5, p6);
}

// ============================================================
// DEBUG KERNEL: HEAD ONLY
// Feed pre-computed P2..P6 feature maps directly into the
// RetinaNet head for isolated debugging of box decoding,
// logit outputs, and NMS — without rerunning backbone/FPN.
//
// Typical usage:
//   1. Run htdet_fpn_only() and save p2..p6 to files.
//   2. Run htdet_head_only() with those files to inspect
//      decoded boxes and NMS output.
// ============================================================
extern "C"
void htdet_head_only(
    const act_t    p2[P2_ELEMS],   // [256, 160, 160]
    const act_t    p3[P3_ELEMS],   // [256,  80,  80]
    const act_t    p4[P4_ELEMS],   // [256,  40,  40]
    const act_t    p5[P5_ELEMS],   // [256,  20,  20]
    const act_t    p6[P6_ELEMS],   // [256,  10,  10]
    const weight_t cls_conv_w[],
    const weight_t reg_conv_w[],
    const weight_t cls_pred_w[ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9],
    const bias_t   cls_pred_b[ANCHORS_PER_LOC * NUM_CLASSES],
    const weight_t reg_pred_w[ANCHORS_PER_LOC * 4 * HEAD_FEAT_CH * 9],
    const bias_t   reg_pred_b[ANCHORS_PER_LOC * 4],
    Detection      detections[MAX_DETS],
    int*           num_dets
) {
    #pragma HLS INTERFACE m_axi port=p2        offset=slave bundle=gmem0  depth=P2_ELEMS
    #pragma HLS INTERFACE m_axi port=p3        offset=slave bundle=gmem1  depth=P3_ELEMS
    #pragma HLS INTERFACE m_axi port=p4        offset=slave bundle=gmem2  depth=P4_ELEMS
    #pragma HLS INTERFACE m_axi port=p5        offset=slave bundle=gmem3  depth=P5_ELEMS
    #pragma HLS INTERFACE m_axi port=p6        offset=slave bundle=gmem4  depth=P6_ELEMS
    #pragma HLS INTERFACE m_axi port=cls_conv_w offset=slave bundle=gmem5 depth=2361344
    #pragma HLS INTERFACE m_axi port=reg_conv_w offset=slave bundle=gmem6 depth=2361344
    #pragma HLS INTERFACE m_axi port=cls_pred_w offset=slave bundle=gmem7 depth=82944
    #pragma HLS INTERFACE m_axi port=cls_pred_b offset=slave bundle=gmem7 depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_w offset=slave bundle=gmem7 depth=82944
    #pragma HLS INTERFACE m_axi port=reg_pred_b offset=slave bundle=gmem7 depth=36
    #pragma HLS INTERFACE m_axi port=detections offset=slave bundle=gmem8 depth=100
    #pragma HLS INTERFACE m_axi port=num_dets   offset=slave bundle=gmem9 depth=1
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    retina_head(
        p2, p3, p4, p5, p6,
        cls_conv_w, reg_conv_w,
        cls_pred_w, cls_pred_b,
        reg_pred_w, reg_pred_b,
        detections, num_dets
    );
}
