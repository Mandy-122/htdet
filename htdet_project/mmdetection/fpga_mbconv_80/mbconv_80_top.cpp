/*
 * mbconv_80_top.cpp
 * Isolated C-synthesis target for mbconv_80 (PTQ INT8, W8A32).
 *
 * mbconv_80 has #pragma HLS INLINE, so HLS flattens it into mbconv_80_top.
 * The synthesis report for mbconv_80_top gives the resource/timing estimate
 * for the mbconv_80 building block.
 *
 * Fixed config: in_ch=64, out_ch=64, expand=4, stride=1  (Stage-1 residual call)
 * Spatial     : H=C1_H=80, W=C1_W=80  (320×320 input)
 */

#include "mbconv_80_top.h"
#include "fpga_utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// mbconv_80 — MobileNetV2-style inverted residual, spatial 160×160
// Identical to mobilevit_backbone.cpp (PTQ INT8 variant).
// ─────────────────────────────────────────────────────────────────────────────
static void mbconv_80(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C1_H, W = C1_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf [256 * C1_H * C1_W];
    static act_t dw_buf [256 * C1_H * C1_W];
    static act_t res_buf[C1_CH * C1_H * C1_W];
    #pragma HLS bind_storage variable=ex_buf  type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf  type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=res_buf type=RAM_T2P impl=URAM

    if (use_res) { buf_copy(res_buf, in, in_ch * H * W); }

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, res_buf, in_ch * H * W); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level HLS synthesis entry point
// Fixed: in_ch=64, out_ch=64, expand=4, stride=1 (Stage-1 residual path)
// ─────────────────────────────────────────────────────────────────────────────
void mbconv_80_top(
    const act_t    in     [MBCONV80_IN_ELEMS],
          act_t    out    [MBCONV80_OUT_ELEMS],
    const weight_t w_conv [MBCONV80_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV80_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // w_conv: 128 banks × 2 = 256 ports → covers proj IC=256 fully unrolled → II=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=128 dim=1
    // in: 32 banks × 2 = 64 ports → covers expand IC=64 fully unrolled → II=1
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=32  dim=1

    int coff = 0, moff = 0;
    mbconv_80(in, out, w_conv, coff, w_meta, moff,
              MBCONV80_IN_CH, MBCONV80_OUT_CH, MBCONV_EXPAND, /*stride=*/1);
}
