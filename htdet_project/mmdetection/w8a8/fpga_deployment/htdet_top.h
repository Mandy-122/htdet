#pragma once

#include "fpga_types.h"

extern "C" {

void htdet_inference(
    const input_t image[],
    const weight_t backbone_w[],
    const weight_t fpn_w[],
    const weight_t cls_conv_w[],
    const weight_t reg_conv_w[],
    const weight_t cls_pred_w[],
    const bias_t cls_pred_b[],
    const weight_t reg_pred_w[],
    const bias_t reg_pred_b[],
    Detection detections[],
    int* num_dets
);

void htdet_backbone_only(
    const input_t  image[INPUT_C * INPUT_H * INPUT_W],
    const weight_t backbone_w[],
    act_t c1[C1_CH * C1_H * C1_W],
    act_t c2[C2_CH * C2_H * C2_W],
    act_t c3[C3_CH * C3_H * C3_W],
    act_t c4[C4_CH * C4_H * C4_W]
);

void htdet_fpn_only(
    const act_t    c1[C1_CH * C1_H * C1_W],
    const act_t    c2[C2_CH * C2_H * C2_W],
    const act_t    c3[C3_CH * C3_H * C3_W],
    const act_t    c4[C4_CH * C4_H * C4_W],
    const weight_t fpn_w[],
    act_t p2[FPN_OUT_CH * P2_H * P2_W],
    act_t p3[FPN_OUT_CH * P3_H * P3_W],
    act_t p4[FPN_OUT_CH * P4_H * P4_W],
    act_t p5[FPN_OUT_CH * P5_H * P5_W],
    act_t p6[FPN_OUT_CH * P6_H * P6_W]
);

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
);

}