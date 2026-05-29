/*
 * htdet_top.cpp  (PTQ INT8 — fully corrected: FPN scale + backbone proj 1x1 scale)
 *
 * All fixes applied:
 *   - Includes mobilevit_backbone.h and calls mobilevit_backbone
 *   - backbone_meta AXI depth updated to 2927344 (+576 for proj 1x1 scales)
 *
 * Changes vs htdet_top.cpp (baseline):
 *   - FPN: fpn_neck with scale+bias (3072 meta elements, was 1536)
 *   - Backbone: mobilevit_backbone with proj 1x1 dequant scale fix
 */

#include "fpga_types.h"
#include "fpga_utils.h"
#include "mobilevit_backbone.h"
#include "fpn_neck.h"
#include "retina_head.h"

extern "C"
void htdet_inference(
    const input_t   image      [INPUT_C * INPUT_H * INPUT_W],
    const weight_t  backbone_conv [],
    const meta_t    backbone_meta [],
    const weight_t  fpn_conv      [],
    const meta_t    fpn_meta      [],
    const weight_t  cls_conv_int8 [],
    const meta_t    cls_conv_meta [],
    const weight_t  reg_conv_int8 [],
    const meta_t    reg_conv_meta [],
    const weight_t  cls_pred_w    [ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9],
    const meta_t    cls_pred_scale[ANCHORS_PER_LOC * NUM_CLASSES],
    const meta_t    cls_pred_b    [ANCHORS_PER_LOC * NUM_CLASSES],
    const weight_t  reg_pred_w    [ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9],
    const meta_t    reg_pred_scale[ANCHORS_PER_LOC * 4],
    const meta_t    reg_pred_b    [ANCHORS_PER_LOC * 4],
    Detection       detections    [MAX_DETS],
    int*            num_dets
) {
    #pragma HLS INTERFACE m_axi port=image          offset=slave bundle=gmem0  depth=IMAGE_ELEMS
    #pragma HLS INTERFACE m_axi port=backbone_conv  offset=slave bundle=gmem1  depth=2010864
    #pragma HLS INTERFACE m_axi port=backbone_meta  offset=slave bundle=gmem2  depth=2927344
    #pragma HLS INTERFACE m_axi port=fpn_conv       offset=slave bundle=gmem3  depth=1505280
    #pragma HLS INTERFACE m_axi port=fpn_meta       offset=slave bundle=gmem4  depth=3072
    #pragma HLS INTERFACE m_axi port=cls_conv_int8  offset=slave bundle=gmem5  depth=1327104
    #pragma HLS INTERFACE m_axi port=cls_conv_meta  offset=slave bundle=gmem6  depth=1536
    #pragma HLS INTERFACE m_axi port=reg_conv_int8  offset=slave bundle=gmem7  depth=1327104
    #pragma HLS INTERFACE m_axi port=reg_conv_meta  offset=slave bundle=gmem8  depth=1536
    #pragma HLS INTERFACE m_axi port=cls_pred_w     offset=slave bundle=gmem9  depth=62208
    #pragma HLS INTERFACE m_axi port=cls_pred_scale offset=slave bundle=gmem10 depth=36
    #pragma HLS INTERFACE m_axi port=cls_pred_b     offset=slave bundle=gmem11 depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_w     offset=slave bundle=gmem12 depth=62208
    #pragma HLS INTERFACE m_axi port=reg_pred_scale offset=slave bundle=gmem13 depth=36
    #pragma HLS INTERFACE m_axi port=reg_pred_b     offset=slave bundle=gmem14 depth=36
    #pragma HLS INTERFACE m_axi port=detections     offset=slave bundle=gmem15 depth=100
    #pragma HLS INTERFACE m_axi port=num_dets       offset=slave bundle=gmem16 depth=1

    #pragma HLS INTERFACE s_axilite port=return         bundle=control
    #pragma HLS INTERFACE s_axilite port=image          bundle=control
    #pragma HLS INTERFACE s_axilite port=backbone_conv  bundle=control
    #pragma HLS INTERFACE s_axilite port=backbone_meta  bundle=control
    #pragma HLS INTERFACE s_axilite port=fpn_conv       bundle=control
    #pragma HLS INTERFACE s_axilite port=fpn_meta       bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_conv_int8  bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_conv_meta  bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_conv_int8  bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_conv_meta  bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_pred_w     bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_pred_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=cls_pred_b     bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_pred_w     bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_pred_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=reg_pred_b     bundle=control
    #pragma HLS INTERFACE s_axilite port=detections     bundle=control
    #pragma HLS INTERFACE s_axilite port=num_dets       bundle=control

    static act_t c1[C1_CH * C1_H * C1_W];
    static act_t c2[C2_CH * C2_H * C2_W];
    static act_t c3[C3_CH * C3_H * C3_W];
    static act_t c4[C4_CH * C4_H * C4_W];
    #pragma HLS bind_storage variable=c1 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=c2 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=c3 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=c4 type=RAM_T2P impl=BRAM

    static act_t p2[FPN_OUT_CH * P2_H * P2_W];
    static act_t p3[FPN_OUT_CH * P3_H * P3_W];
    static act_t p4[FPN_OUT_CH * P4_H * P4_W];
    static act_t p5[FPN_OUT_CH * P5_H * P5_W];
    static act_t p6[FPN_OUT_CH * P6_H * P6_W];
    #pragma HLS bind_storage variable=p2 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=p3 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=p4 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=p5 type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=p6 type=RAM_T2P impl=BRAM

    mobilevit_backbone(image, backbone_conv, backbone_meta, c1, c2, c3, c4);
    fpn_neck(c1, c2, c3, c4, fpn_conv, fpn_meta, p2, p3, p4, p5, p6);
    retina_head(
        p2, p3, p4, p5, p6,
        cls_conv_int8, cls_conv_meta,
        reg_conv_int8, reg_conv_meta,
        cls_pred_w,    cls_pred_scale, cls_pred_b,
        reg_pred_w,    reg_pred_scale, reg_pred_b,
        detections, num_dets
    );
}
