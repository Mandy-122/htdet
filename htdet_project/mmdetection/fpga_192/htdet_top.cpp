/*
 * htdet_top.cpp
 * Top-level HLS kernel for HTDet — 192-channel FPN+Head variant.
 * MobileViT-S backbone (unchanged) + FPN out=192 + RetinaHead feat=192.
 *
 * THIS IS THE FILE THAT VITIS HLS SYNTHESIZES TO RTL.
 * Set top function: htdet_inference  (or htdet_backbone_only for backbone-only synthesis)
 *
 * AXI interface allocation:
 *   gmem0 : input image
 *   gmem1 : backbone weights (~19.8MB float32, same as 256-ch model)
 *   gmem2 : FPN weights  (~5.74MB float32  — reduced from 9.9MB in 256-ch)
 *   gmem3 : cls head stacked conv weights (~5.1MB float32 — reduced from 9.0MB)
 *   gmem4 : reg head stacked conv weights (~5.1MB float32)
 *   gmem5 : cls/reg prediction head weights (small)
 *   gmem6 : detections output
 *   gmem7 : num_dets output
 *   control: s_axilite
 *
 * Weight element counts (float32):
 *   backbone_w : 4,937,632  (mobilevit_s, unchanged)
 *   fpn_w      : 1,506,816  (lateral: (64+96+128+640)*192+4*192; output: 4*(192*192*9+192))
 *   cls_conv_w : 1,329,920  (4 * 192*(192*9+2))
 *   reg_conv_w : 1,329,920
 *   cls_pred_w :    62,208  (9*4*192*9)
 *   reg_pred_w :    62,208
 */

#include "fpga_types.h"
#include "fpga_utils.h"
#include "mobilevit_backbone.h"
#include "fpn_neck.h"
#include "retina_head.h"

// ============================================================
// TOP-LEVEL KERNEL
// ============================================================
extern "C"
void htdet_inference(
    const input_t   image      [INPUT_C * INPUT_H * INPUT_W],
    const weight_t  backbone_w [],
    const weight_t  fpn_w      [],
    const weight_t  cls_conv_w [],
    const weight_t  reg_conv_w [],
    const weight_t  cls_pred_w [ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9],
    const bias_t    cls_pred_b [ANCHORS_PER_LOC * NUM_CLASSES],
    const weight_t  reg_pred_w [ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9],
    const bias_t    reg_pred_b [ANCHORS_PER_LOC * 4],
    Detection       detections [MAX_DETS],
    int*            num_dets
) {
    #pragma HLS INTERFACE m_axi port=image       offset=slave bundle=gmem0  depth=IMAGE_ELEMS
    #pragma HLS INTERFACE m_axi port=backbone_w  offset=slave bundle=gmem1  depth=4937632
    #pragma HLS INTERFACE m_axi port=fpn_w       offset=slave bundle=gmem2  depth=1506816
    #pragma HLS INTERFACE m_axi port=cls_conv_w  offset=slave bundle=gmem3  depth=1328640
    #pragma HLS INTERFACE m_axi port=reg_conv_w  offset=slave bundle=gmem4  depth=1328640
    #pragma HLS INTERFACE m_axi port=cls_pred_w  offset=slave bundle=gmem5  depth=62208
    #pragma HLS INTERFACE m_axi port=cls_pred_b  offset=slave bundle=gmem5  depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_w  offset=slave bundle=gmem5  depth=62208
    #pragma HLS INTERFACE m_axi port=reg_pred_b  offset=slave bundle=gmem5  depth=36
    #pragma HLS INTERFACE m_axi port=detections  offset=slave bundle=gmem6  depth=100
    #pragma HLS INTERFACE m_axi port=num_dets    offset=slave bundle=gmem7  depth=1

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

    // ---- On-chip buffers for backbone outputs ----
    static act_t c1[C1_CH * C1_H * C1_W];   // [64,  160, 160]
    static act_t c2[C2_CH * C2_H * C2_W];   // [96,   80,  80]
    static act_t c3[C3_CH * C3_H * C3_W];   // [128,  40,  40]
    static act_t c4[C4_CH * C4_H * C4_W];   // [640,  20,  20]
    #pragma HLS RESOURCE variable=c1 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c2 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c3 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=c4 core=RAM_T2P_BRAM

    // ---- On-chip buffers for FPN outputs (192ch) ----
    static act_t p2[FPN_OUT_CH * P2_H * P2_W];   // [192, 160, 160] = ~18.8MB
    static act_t p3[FPN_OUT_CH * P3_H * P3_W];   // [192,  80,  80] = ~4.7MB
    static act_t p4[FPN_OUT_CH * P4_H * P4_W];   // [192,  40,  40] = ~1.2MB
    static act_t p5[FPN_OUT_CH * P5_H * P5_W];   // [192,  20,  20] = ~307KB
    static act_t p6[FPN_OUT_CH * P6_H * P6_W];   // [192,  10,  10] =  ~77KB
    #pragma HLS RESOURCE variable=p2 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p3 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p4 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p5 core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=p6 core=RAM_T2P_BRAM

    mobilevit_backbone(image, backbone_w, c1, c2, c3, c4);
    fpn_neck(c1, c2, c3, c4, fpn_w, p2, p3, p4, p5, p6);
    retina_head(
        p2, p3, p4, p5, p6,
        cls_conv_w, reg_conv_w,
        cls_pred_w, cls_pred_b,
        reg_pred_w, reg_pred_b,
        detections, num_dets
    );
}

// ============================================================
// DEBUG KERNELS (set as top function for isolated synthesis)
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

extern "C"
void htdet_head_only(
    const act_t    p2[P2_ELEMS],
    const act_t    p3[P3_ELEMS],
    const act_t    p4[P4_ELEMS],
    const act_t    p5[P5_ELEMS],
    const act_t    p6[P6_ELEMS],
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
    #pragma HLS INTERFACE m_axi port=cls_conv_w offset=slave bundle=gmem5 depth=1328640
    #pragma HLS INTERFACE m_axi port=reg_conv_w offset=slave bundle=gmem6 depth=1328640
    #pragma HLS INTERFACE m_axi port=cls_pred_w offset=slave bundle=gmem7 depth=62208
    #pragma HLS INTERFACE m_axi port=cls_pred_b offset=slave bundle=gmem7 depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_w offset=slave bundle=gmem7 depth=62208
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
