/*
 * mbconv_160_top.cpp  (v5 — HWC, OC_TILE=8, partial acc + 5-level tree)
 * Run 2 changes vs Run 1 (OC_TILE=4, 5 ns):
 *   - OC_TILE 4→8: halves expand/proj outer iterations
 *   - ex_buf/dw_buf: added cyclic factor=128 partition → eliminates urem_64s in DW
 *   - Clock: 5 ns → 4.5 ns (222 MHz) — clears timing violation
 *
 * mbconv_160: in_ch=32, out_ch=64, expand=4, stride=2  (STAGE0→C1 downsample)
 * Input spatial : H=160, W=160  →  Output spatial: H=80, W=80
 * hid = 32 * 4 = 128
 *
 * IC tiles: expand=2 (32/16), proj=8 (128/16)
 * OC groups: expand=16 (128/8), proj=8 (64/8)
 *
 * Expected IC_T II=4 for OC_TILE=8 (8 weight reads / 2 BRAM ports → min II=4).
 * Outer trip savings (2×) dominate → expect ~2× speedup on expand vs Run 1.
 *
 * Expected latency:
 *   Expand: 16 OC-groups × 25,600 HW × ~295 cycles ≈ 118M = 531 ms
 *   DW    : 128×6400 trips × II=5 ≈ 4.1M = 18 ms  (same, urem fix is resource-only)
 *   Proj  : 8 OC-groups × 6,400 HW × ~295 cycles ≈ 15M = 68 ms
 *   Total ≈ 137M cycles ≈ 617 ms @ 222 MHz
 */

#include "mbconv_160_top.h"
#include "fpga_utils.h"

static void mbconv_160(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H   = STEM_H, W = STEM_W;          // 160×160
    int hid = in_ch * expand;              // 128
    int oH  = DIV_CEIL(H, stride);         // 80
    int oW  = DIV_CEIL(W, stride);         // 80

    // HWC layout buffers
    // ex_buf: DW input  — 160×160×128 (12.5 MB — exceeds device URAM)
    // dw_buf: DW output —  80× 80×128 ( 3.1 MB — also exceeds device URAM)
    static act_t ex_buf[STEM_H * STEM_W * 128];    // 3,276,800 elem
    static act_t dw_buf[C1_H   * C1_W   * 128];    //   819,200 elem
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=URAM
    // cyclic factor=128 (= ch, power-of-2) → bank = addr%128 = c → no urem_64s
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=128 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=128 dim=1

    // expand: in_ch=32 → hid=128  (2 IC tiles)
    conv1x1_bn_silu(in, ex_buf,
                    w_conv + coff,
                    w_meta + moff,
                    w_meta + moff + hid,
                    in_ch, hid, H, W);
    coff += hid * in_ch;        // 128*32 = 4,096
    moff += hid + hid;          // 256

    // DW: ch=128, stride=2, HWC in (ex_buf 160×160), HWC out (dw_buf 80×80)
    dw_conv3x3_bn_silu(ex_buf, dw_buf,
                       w_conv + coff,
                       w_meta + moff,
                       w_meta + moff + hid,
                       hid, H, W, stride);
    coff += hid * 9;            // 128*9 = 1,152
    moff += hid + hid;          // 256

    // proj: in_ch=128 → out_ch=64  (8 IC tiles), HWC in (dw_buf 80×80), HWC out
    conv1x1_bn(dw_buf, out,
               w_conv + coff,
               w_meta + moff,
               w_meta + moff + out_ch,
               hid, out_ch, oH, oW);
    coff += out_ch * hid;       // 64*128 = 8,192
    moff += out_ch + out_ch;    // 128
    // stride=2, in_ch≠out_ch → no residual
}

void mbconv_160_top(
    const act_t    in     [MBCONV160_IN_ELEMS],
          act_t    out    [MBCONV160_OUT_ELEMS],
    const weight_t w_conv [MBCONV160_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV160_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // HWC: bank=(hw*IC + t*16 + k)%16 = k → no urem, II=1/2 clean
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    int coff = 0, moff = 0;
    mbconv_160(in, out, w_conv, coff, w_meta, moff,
               MBCONV160_IN_CH, MBCONV160_OUT_CH, MBCONV_EXPAND, /*stride=*/2);
}
