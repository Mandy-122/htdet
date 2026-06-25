/*
 * mbconv_s2a_top.cpp  (v4 — HWC, OC_TILE=4, partial acc + 5-level tree)
 *
 * Stage-2a MBConv: in_ch=64, out_ch=96, expand=4, stride=2 (no residual)
 * Input : H=80, W=80  →  Output: H=40, W=40
 * hid = 256
 *
 * IC tiles: expand=4 (64/16), proj=16 (256/16)
 * OC groups: expand=64 (256/4), proj=24 (96/4)
 *
 * NOTE: Expand config identical to fpga_mbconv_80 (IC=64, hid=256, 80×80).
 * The only structural differences vs mbconv_80 are:
 *   - out_ch=96 instead of 64 (proj has 24 vs 16 OC groups)
 *   - stride=2 → dw outputs at 40×40 instead of 80×80
 *   - no residual add (in_ch ≠ out_ch)
 *
 * ⚠ URAM: ex_buf = 6.25 MB (overflows), dw_buf = 1.5625 MB (fits).
 */

#include "mbconv_s2a_top.h"
#include "fpga_utils.h"

static void mbconv_s2a(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff
) {
    #pragma HLS INLINE
    const int H = C1_H, W = C1_W;          // 80×80 input
    const int oH = C2_H, oW = C2_W;        // 40×40 output
    const int hid = MBCONVS2A_HID;         // 256

    // ex_buf: expand output — 80×80×256  (6.25 MB, exceeds URAM)
    static act_t ex_buf[C1_H * C1_W * MBCONVS2A_HID];
    // dw_buf: DW output  — 40×40×256  (1.5625 MB, fits in URAM)
    static act_t dw_buf[C2_H * C2_W * MBCONVS2A_HID];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=256 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=256 dim=1

    // expand: in_ch=64 → hid=256  (4 IC tiles)
    conv1x1_bn_silu(in, ex_buf,
                    w_conv + coff,
                    w_meta + moff,
                    w_meta + moff + hid,
                    MBCONVS2A_IN_CH, hid, H, W);
    coff += hid * MBCONVS2A_IN_CH;   // 256*64 = 16,384
    moff += hid + hid;               // 512

    // DW: ch=256, stride=2, input 80×80 → output 40×40, HWC
    dw_conv3x3_bn_silu(ex_buf, dw_buf,
                       w_conv + coff,
                       w_meta + moff,
                       w_meta + moff + hid,
                       hid, H, W, /*stride=*/2);
    coff += hid * 9;                 // 256*9 = 2,304
    moff += hid + hid;              // 512

    // proj: hid=256 → out_ch=96  (16 IC tiles), output at 40×40, HWC
    conv1x1_bn(dw_buf, out,
               w_conv + coff,
               w_meta + moff,
               w_meta + moff + MBCONVS2A_OUT_CH,
               hid, MBCONVS2A_OUT_CH, oH, oW);
    coff += MBCONVS2A_OUT_CH * hid;              // 96*256 = 24,576
    moff += MBCONVS2A_OUT_CH + MBCONVS2A_OUT_CH; // 192
    // stride=2, in_ch≠out_ch → no residual
}

void mbconv_s2a_top(
    const act_t    in    [MBCONVS2A_IN_ELEMS],
          act_t    out   [MBCONVS2A_OUT_ELEMS],
    const weight_t w_conv[MBCONVS2A_WCONV_ELEMS],
    const meta_t   w_meta[MBCONVS2A_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    int coff = 0, moff = 0;
    mbconv_s2a(in, out, w_conv, coff, w_meta, moff);
}
