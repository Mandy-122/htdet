#include "conv1_1_top.h"
#include <stdint.h>

static inline float silu(float x) {
    return x / (1.0f + __builtin_expf(-x));
}

#define TILE      16
#define OC_TILE    2
#define NUM_TILES (TEST_IN_CH / TILE)   // 16 for IC=256

// OC_TILE=2: process 2 output channels per IC pass.
// The same 'in' tile (16 floats) is reused for both OCs — halves the in-reads per output.
// w_conv accessed at oc*IC and (oc+1)*IC; with factor=16 and IC=256 both hit the same
// 16 banks (256 % 16 = 0). HLS will use dual-port BRAM access (port A / port B) to
// serve both reads in the same cycle, keeping II=1.
static void conv1x1_bn_silu_tiled(
    const act_t*    input,
          act_t*    output,
    const weight_t* weights,
    const meta_t*   bn_scale,
    const meta_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    C1X1S_OC: for (int oc = 0; oc < out_ch; oc += OC_TILE) {
        C1X1S_HW: for (int hw = 0; hw < H * W; hw++) {

            acc_t partial0[NUM_TILES], partial1[NUM_TILES];
            #pragma HLS ARRAY_PARTITION variable=partial0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial1 complete dim=1

            // IC tile loop — II target=1, both OC tiles computed simultaneously
            C1X1S_IC_T: for (int t = 0; t < NUM_TILES; t++) {
                #pragma HLS PIPELINE II=1
                acc_t tile0 = 0.0f, tile1 = 0.0f;
                C1X1S_IC_K: for (int k = 0; k < TILE; k++) {
                    #pragma HLS UNROLL
                    acc_t x = (acc_t)input[hw * in_ch + t * TILE + k];
                    tile0 += x * (acc_t)(float)weights[oc       * in_ch + t * TILE + k];
                    tile1 += x * (acc_t)(float)weights[(oc + 1) * in_ch + t * TILE + k];
                }
                partial0[t] = tile0;
                partial1[t] = tile1;
            }

            // 4-level binary tree reduction — both OCs in parallel
            acc_t l0_0[8], l1_0[4], l2_0[2];
            acc_t l0_1[8], l1_1[4], l2_1[2];
            #pragma HLS ARRAY_PARTITION variable=l0_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_1 complete dim=1

            for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                l0_0[i] = partial0[2*i] + partial0[2*i+1];
                l0_1[i] = partial1[2*i] + partial1[2*i+1];
            }
            for (int i = 0; i < 4; i++) {
                #pragma HLS UNROLL
                l1_0[i] = l0_0[2*i] + l0_0[2*i+1];
                l1_1[i] = l0_1[2*i] + l0_1[2*i+1];
            }
            for (int i = 0; i < 2; i++) {
                #pragma HLS UNROLL
                l2_0[i] = l1_0[2*i] + l1_0[2*i+1];
                l2_1[i] = l1_1[2*i] + l1_1[2*i+1];
            }
            acc_t acc0 = l2_0[0] + l2_0[1];
            acc_t acc1 = l2_1[0] + l2_1[1];

            acc_t y0 = acc0 * (acc_t)bn_scale[oc]     + (acc_t)bn_bias[oc];
            acc_t y1 = acc1 * (acc_t)bn_scale[oc + 1] + (acc_t)bn_bias[oc + 1];
            output[oc       * H * W + hw] = (act_t)silu((float)y0);
            output[(oc + 1) * H * W + hw] = (act_t)silu((float)y1);
        }
    }
}

void conv1x1_top(
    const act_t    in     [TEST_IN_ELEMS],
          act_t    out    [TEST_OUT_ELEMS],
    const weight_t w_conv [TEST_W_ELEMS],
    const meta_t   w_meta [TEST_META_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // in:     factor=16 — bank = (hw*IC + t*16 + k) % 16 = k (constant per k)
    // w_conv: factor=16 — OC 0 and OC 1 hit same banks; HLS uses port A / port B
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1

    conv1x1_bn_silu_tiled(
        in, out, w_conv,
        w_meta,
        w_meta + TEST_OUT_CH,
        TEST_IN_CH, TEST_OUT_CH, TEST_H, TEST_W
    );
}
