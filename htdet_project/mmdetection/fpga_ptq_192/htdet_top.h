#pragma once

#include "fpga_types.h"

extern "C" {

void htdet_inference(
    const input_t   image[],
    const weight_t  backbone_conv[],    // int8: backbone conv weights (backbone_int8.bin)
    const meta_t    backbone_meta[],    // float: backbone BN params + transformer weights
    const weight_t  fpn_conv[],         // int8: FPN conv weights (fpn_int8.bin)
    const meta_t    fpn_meta[],         // float: FPN biases
    const weight_t  cls_conv_int8[],    // int8: cls stacked conv weights
    const meta_t    cls_conv_meta[],    // float: cls stacked conv scales + biases
    const weight_t  reg_conv_int8[],    // int8: reg stacked conv weights
    const meta_t    reg_conv_meta[],    // float: reg stacked conv scales + biases
    const weight_t  cls_pred_w[],       // int8: cls pred conv weights
    const meta_t    cls_pred_scale[],   // float: cls pred per-channel dequant scale
    const meta_t    cls_pred_b[],       // float: cls pred bias
    const weight_t  reg_pred_w[],       // int8: reg pred conv weights
    const meta_t    reg_pred_scale[],   // float: reg pred per-channel dequant scale
    const meta_t    reg_pred_b[],       // float: reg pred bias
    Detection       detections[],
    int*            num_dets
);

}
