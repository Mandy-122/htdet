/*
 * mbconv_20_top.cpp  (v4 — HWC, OC_TILE=4, partial acc + 5-level tree)
 *
 * mbconv_20: in_ch=128, out_ch=128, expand=4, stride=1  (C3 stage residual)
 * Spatial: H=20, W=20
 * hid = 128 * 4 = 512   →  IC tiles: expand=8, proj=32 (fits MAX_IC_TILES=32)
 *
 * Layout: HWC [H*W * ch] throughout.
 * Buffers (HWC):
 *   ex_buf : [C3_H*C3_W * 512] = 20*20*512 = 204,800 elem
 *   dw_buf : [C3_H*C3_W * 512] = 20*20*512 = 204,800 elem
 *   res_buf: [C3_H*C3_W * 128] = 20*20*128 =  51,200 elem  (stride=1 residual)
 *
 * Partition:
 *   in     cyclic factor=16: bank=(hw*128+ic)%16=ic%16=k ✓ (128%16=0)
 *   w_conv cyclic factor=16: bank=k ✓ (IC=128,512 both mult of 16)
 *
 * URAM estimate: ex+dw+res ≈ 1.78 MB < 2.88 MB device capacity (fits without partitioning)
 */

#include "mbconv_20_top.h"
#include "fpga_utils.h"

static void mbconv_20(
    const act_t*    in,  act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C3_H, W = C3_W;
    int hid = in_ch * expand;               // 512
    bool use_res = (stride == 1 && in_ch == out_ch);  // true

    static act_t ex_buf [C3_H * C3_W * 512];   // 204,800 elem
    static act_t dw_buf [C3_H * C3_W * 512];   // 204,800 elem
    static act_t res_buf[C3_H * C3_W * C3_CH]; //  51,200 elem
    #pragma HLS bind_storage variable=ex_buf  type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=dw_buf  type=RAM_T2P impl=URAM
    #pragma HLS bind_storage variable=res_buf type=RAM_T2P impl=URAM

    if (use_res) { buf_copy(res_buf, in, in_ch * H * W); }

    // expand: in_ch=128 → hid=512  (8 IC tiles)
    conv1x1_bn_silu(in, ex_buf,
                    w_conv + coff,
                    w_meta + moff,
                    w_meta + moff + hid,
                    in_ch, hid, H, W);
    coff += hid * in_ch;        // 512*128 = 65,536
    moff += hid + hid;          // 1,024

    // DW: ch=512, stride=1, HWC in/out
    dw_conv3x3_bn_silu(ex_buf, dw_buf,
                       w_conv + coff,
                       w_meta + moff,
                       w_meta + moff + hid,
                       hid, H, W, stride);
    coff += hid * 9;            // 512*9 = 4,608
    moff += hid + hid;          // 1,024

    // proj: in_ch=512 → out_ch=128  (32 IC tiles = MAX_IC_TILES)
    conv1x1_bn(dw_buf, out,
               w_conv + coff,
               w_meta + moff,
               w_meta + moff + out_ch,
               hid, out_ch, H, W);
    coff += out_ch * hid;       // 128*512 = 65,536
    moff += out_ch + out_ch;    // 256

    if (use_res) { elem_add_inplace(out, res_buf, in_ch * H * W); }
}

void mbconv_20_top(
    const act_t    in     [MBCONV20_IN_ELEMS],
          act_t    out    [MBCONV20_OUT_ELEMS],
    const weight_t w_conv [MBCONV20_WCONV_ELEMS],
    const meta_t   w_meta [MBCONV20_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // HWC: bank=(hw*IC + t*16 + k)%16 = k → clean, no urem, II=1 target
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    int coff = 0, moff = 0;
    mbconv_20(in, out, w_conv, coff, w_meta, moff,
              MBCONV20_IN_CH, MBCONV20_OUT_CH, MBCONV_EXPAND, /*stride=*/1);
}
