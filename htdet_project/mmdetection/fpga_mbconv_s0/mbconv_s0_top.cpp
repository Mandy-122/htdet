/*
 * mbconv_s0_top.cpp  (v4 — HWC, OC_TILE=4, partial acc + 5-level tree)
 *
 * Stage-0 MBConv: in_ch=16, out_ch=32, expand=4, stride=1 (no residual)
 * Input spatial : H=160, W=160  (same as output)
 * hid = 64
 *
 * IC tiles: expand=1 (16/16), proj=4 (64/16)
 * OC groups: expand=16 (64/4), proj=8 (32/4)
 *
 * Note: expand has only 1 IC tile (IC=16, TILE=16) → pipeline depth ≈ iteration
 * latency. IC_T II may be 1 (single tile, no recurrence needed) or 2.
 * Observe from Run 1 report.
 *
 * ⚠ URAM: ex_buf (6.25 MB) + dw_buf (6.25 MB) = 12.5 MB >> 2.88 MB capacity.
 *   Synthesis completes; P&R blocked until spatial tiling implemented.
 *
 * Partition:
 *   in     cyclic factor=16: bank = (hw*16 + k)%16 = k ✓
 *   w_conv cyclic factor=16: bank = k ✓
 */

#include "mbconv_s0_top.h"
#include "fpga_utils.h"

static void mbconv_s0(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff
) {
    #pragma HLS INLINE
    const int H   = STEM_H, W = STEM_W;   // 160×160
    const int hid = MBCONVS0_HID;         // 64

    static act_t ex_buf[STEM_H * STEM_W * MBCONVS0_HID];  // 160×160×64 = 6.25 MB
    static act_t dw_buf[STEM_H * STEM_W * MBCONVS0_HID];  // 160×160×64 = 6.25 MB
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=64 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=64 dim=1

    // expand: in_ch=16 → hid=64  (1 IC tile)
    conv1x1_bn_silu(in, ex_buf,
                    w_conv + coff,
                    w_meta + moff,
                    w_meta + moff + hid,
                    MBCONVS0_IN_CH, hid, H, W);
    coff += hid * MBCONVS0_IN_CH;    // 64*16 = 1,024
    moff += hid + hid;               // 128

    // DW: ch=64, stride=1, HWC
    dw_conv3x3_bn_silu(ex_buf, dw_buf,
                       w_conv + coff,
                       w_meta + moff,
                       w_meta + moff + hid,
                       hid, H, W, /*stride=*/1);
    coff += hid * 9;                 // 64*9 = 576
    moff += hid + hid;              // 128

    // proj: hid=64 → out_ch=32  (4 IC tiles), HWC
    conv1x1_bn(dw_buf, out,
               w_conv + coff,
               w_meta + moff,
               w_meta + moff + MBCONVS0_OUT_CH,
               hid, MBCONVS0_OUT_CH, H, W);
    coff += MBCONVS0_OUT_CH * hid;  // 32*64 = 2,048
    moff += MBCONVS0_OUT_CH + MBCONVS0_OUT_CH;  // 64
    // stride=1 but in_ch(16)≠out_ch(32) → no residual add
}

void mbconv_s0_top(
    const act_t    in    [MBCONVS0_IN_ELEMS],
          act_t    out   [MBCONVS0_OUT_ELEMS],
    const weight_t w_conv[MBCONVS0_WCONV_ELEMS],
    const meta_t   w_meta[MBCONVS0_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    int coff = 0, moff = 0;
    mbconv_s0(in, out, w_conv, coff, w_meta, moff);
}
