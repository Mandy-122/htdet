/*
 * mbconv_40_top.cpp
 * Isolated C-synthesis target for mbconv_40 (PTQ INT8, W8A32).
 *
 * Fixed config: in_ch=96, out_ch=128, expand=4, stride=2
 * Spatial     : H=C2_H=40, W=C2_W=40  (320×320 input)
 *
 * Lessons applied from mbconv_80 runs:
 *   - ex_buf / dw_buf → URAM (no manual cyclic partition; HLS infers as needed)
 *   - No res_buf: stride=2 means use_res=false
 *   - w_conv cyclic factor=192  → 384 ports, covers proj IC=384 fully unrolled
 *   - in     cyclic factor=48   → 96  ports, covers expand IC=96 fully unrolled
 *   - Clock 5ns (200 MHz) — sitofp is 3.24ns, fits in 4.46ns effective budget
 */

#include "mbconv_40_top.h"
#include "fpga_utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// mbconv_40 — MobileNetV2-style inverted residual, spatial 40×40
// Identical to mobilevit_backbone.cpp (PTQ INT8 variant).
// ─────────────────────────────────────────────────────────────────────────────
static void mbconv_40(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C2_H, W = C2_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[512 * C2_H * C2_W];
    static act_t dw_buf[512 * C2_H * C2_W];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=URAM

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

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level HLS synthesis entry point
// Fixed: in_ch=96, out_ch=128, expand=4, stride=2
// ─────────────────────────────────────────────────────────────────────────────
void mbconv_40_top(
    const act_t    in     [MBCONV40_IN_ELEMS],
          act_t    out    [MBCONV40_OUT_ELEMS],
    const weight_t w_conv [MBCONV40_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV40_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // w_conv: 192 banks × 2 = 384 ports → proj IC=384 fully unrolled → II=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=192 dim=1
    // in: 48 banks × 2 = 96 ports → expand IC=96 fully unrolled → II=1
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=48  dim=1

    int coff = 0, moff = 0;
    mbconv_40(in, out, w_conv, coff, w_meta, moff,
              MBCONV40_IN_CH, MBCONV40_OUT_CH, MBCONV_EXPAND, /*stride=*/2);
}
