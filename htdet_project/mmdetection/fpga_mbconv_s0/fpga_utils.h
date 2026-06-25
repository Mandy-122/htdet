/*
 * fpga_utils.h  (PTQ INT8 — v4, OC_TILE=4, MAX_IC_TILES=32)
 *
 * Covers mbconv_80 (IC up to 256, 16 tiles), mbconv_40 (IC up to 384, 24 tiles),
 * mbconv_20 (IC up to 512, 32 tiles). MAX_IC_TILES=32 handles all with zero padding.
 *
 * Layout: HWC [H*W * ch] for all 1x1 and DW conv activations.
 * OC_TILE=4: 4 output channels per IC pass; halves outer trip count vs OC_TILE=2.
 * Tree: 5-level binary tree (32→16→8→4→2→1) for partial sum reduction.
 */

#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include "fpga_types.h"
#include <cmath>

#define MAX_IC_TILES 32   // ceil(max_IC / 16) = 512/16 = 32 for mbconv_20 proj
#define IC_TILE      16
#define OC_TILE       4

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================
inline act_t relu(act_t x) {
    #pragma HLS INLINE
    return (x > (act_t)0) ? x : (act_t)0;
}

inline act_t silu(act_t x) {
    #pragma HLS INLINE
    #ifdef __SYNTHESIS__
    act_t sig = (x < (act_t)(-4.0f)) ? (act_t)0.0f :
                (x > (act_t)( 4.0f)) ? (act_t)1.0f :
                (act_t)0.5f + x * (act_t)0.125f;
    return x * sig;
    #else
    return (act_t)((float)x / (1.0f + expf(-(float)x)));
    #endif
}

inline score_t sigmoid(score_t x) {
    #pragma HLS INLINE
    #ifdef __SYNTHESIS__
    if (x < (score_t)(-4.0f)) return (score_t)0.0f;
    if (x > (score_t)( 4.0f)) return (score_t)1.0f;
    return (score_t)0.5f + x * (score_t)0.125f;
    #else
    return (score_t)(1.0f / (1.0f + expf(-(float)x)));
    #endif
}

// ============================================================
// FUSED BN / LAYER NORM / SOFTMAX / LINEAR (unchanged)
// ============================================================
inline act_t fused_bn(act_t x, meta_t scale, bias_t bias) {
    #pragma HLS INLINE
    return (act_t)((acc_t)x * (acc_t)scale + (acc_t)bias);
}

inline void layer_norm_seq(
    const act_t* input, act_t* output,
    const meta_t* weight, const meta_t* bias_ln,
    int seq_len, int dim
) {
    #pragma HLS INLINE
    for (int s = 0; s < seq_len; s++) {
        const act_t* row = input  + s * dim;
        act_t*       out = output + s * dim;
        acc_t sum = 0;
        LN_MEAN: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            sum += (acc_t)row[i];
        }
        act_t mean = (act_t)(sum / (acc_t)dim);
        acc_t var_sum = 0;
        LN_VAR: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            acc_t d = (acc_t)row[i] - (acc_t)mean; var_sum += d * d;
        }
        act_t inv_std = (act_t)(1.0f / sqrtf((float)(var_sum / (acc_t)dim) + 1e-5f));
        LN_NORM: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            out[i] = (act_t)(((acc_t)row[i]-(acc_t)mean)*(acc_t)inv_std*(acc_t)weight[i]+(acc_t)bias_ln[i]);
        }
    }
}

inline void softmax_row(const act_t* input, act_t* output, int N) {
    #pragma HLS INLINE
    act_t max_v = input[0];
    SM_MAX: for (int i = 1; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if (input[i] > max_v) max_v = input[i];
    }
    acc_t sum = 0;
    SM_EXP: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        float e = expf((float)(input[i]-max_v)); output[i] = (act_t)e; sum += (acc_t)e;
    }
    SM_NORM: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = (act_t)((acc_t)output[i]/sum);
    }
}

inline void linear_layer(
    const act_t* input, act_t* output,
    const meta_t* weights, const meta_t* bias,
    int batch, int in_dim, int out_dim
) {
    LIN_B: for (int b = 0; b < batch; b++) {
        LIN_OD: for (int od = 0; od < out_dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)bias[od];
            LIN_ID: for (int id = 0; id < in_dim; id++) {
                #pragma HLS UNROLL factor=16
                acc += (acc_t)input[b*in_dim+id]*(acc_t)weights[od*in_dim+id];
            }
            output[b*out_dim+od] = (act_t)acc;
        }
    }
}

// ============================================================
// 3x3 CONV (CHW layout — for stem/FPN, unchanged)
// ============================================================
inline void conv3x3_bn_silu(
    const act_t* input, act_t* output,
    const weight_t* weights, const meta_t* bn_scale, const meta_t* bn_bias,
    int in_ch, int out_ch, int H, int W, int stride
) {
    int out_H = DIV_CEIL(H,stride), out_W = DIV_CEIL(W,stride);
    C3X3_OC: for (int oc = 0; oc < out_ch; oc++) {
        C3X3_OH: for (int oh = 0; oh < out_H; oh++) {
            C3X3_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                C3X3_IC: for (int ic = 0; ic < in_ch; ic++) {
                    #pragma HLS UNROLL factor=16
                    C3X3_KH: for (int kh = 0; kh < 3; kh++) {
                        #pragma HLS UNROLL
                        C3X3_KW: for (int kw = 0; kw < 3; kw++) {
                            #pragma HLS UNROLL
                            int ih=oh*stride+kh-1, iw=ow*stride+kw-1;
                            if (ih>=0&&ih<H&&iw>=0&&iw<W)
                                acc += (acc_t)input[ic*H*W+ih*W+iw]*(acc_t)(float)weights[(oc*in_ch+ic)*9+kh*3+kw];
                        }
                    }
                }
                output[oc*out_H*out_W+oh*out_W+ow] = silu((act_t)((acc_t)acc*(acc_t)bn_scale[oc]+(acc_t)bn_bias[oc]));
            }
        }
    }
}

inline void conv3x3_bn(
    const act_t* input, act_t* output,
    const weight_t* weights, const meta_t* bn_scale, const meta_t* bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    C3X3B_OC: for (int oc = 0; oc < out_ch; oc++) {
        C3X3B_OH: for (int oh = 0; oh < H; oh++) {
            C3X3B_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                C3X3B_IC: for (int ic = 0; ic < in_ch; ic++) {
                    #pragma HLS UNROLL factor=16
                    C3X3B_KH: for (int kh = 0; kh < 3; kh++) {
                        #pragma HLS UNROLL
                        C3X3B_KW: for (int kw = 0; kw < 3; kw++) {
                            #pragma HLS UNROLL
                            int ih=oh+kh-1, iw=ow+kw-1;
                            if (ih>=0&&ih<H&&iw>=0&&iw<W)
                                acc += (acc_t)input[ic*H*W+ih*W+iw]*(acc_t)(float)weights[(oc*in_ch+ic)*9+kh*3+kw];
                        }
                    }
                }
                output[oc*H*W+oh*W+ow] = (act_t)((acc_t)acc*(acc_t)bn_scale[oc]+(acc_t)bn_bias[oc]);
            }
        }
    }
}

// ============================================================
// 1x1 CONV + BN + SiLU  (HWC, OC_TILE=4, partial acc, 5-level tree)
// input/output: HWC [H*W * ch]
// Caller partitions: in cyclic factor=16, w_conv cyclic factor=16
// ============================================================
inline void conv1x1_bn_silu(
    const act_t*    input,
          act_t*    output,
    const weight_t* weights,
    const meta_t*   bn_scale,
    const meta_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    #pragma HLS INLINE
    C1X1S_OC: for (int oc = 0; oc < out_ch; oc += OC_TILE) {
        C1X1S_HW: for (int hw = 0; hw < H * W; hw++) {

            // All local arrays declared first — ARRAY_PARTITION must precede any statements
            acc_t partial0[MAX_IC_TILES], partial1[MAX_IC_TILES];
            acc_t partial2[MAX_IC_TILES], partial3[MAX_IC_TILES];
            #pragma HLS ARRAY_PARTITION variable=partial0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial3 complete dim=1

            // 5-level binary tree buffers declared up-front (before any for-loops)
            acc_t l0_0[16], l0_1[16], l0_2[16], l0_3[16];
            acc_t l1_0[ 8], l1_1[ 8], l1_2[ 8], l1_3[ 8];
            acc_t l2_0[ 4], l2_1[ 4], l2_2[ 4], l2_3[ 4];
            acc_t l3_0[ 2], l3_1[ 2], l3_2[ 2], l3_3[ 2];
            #pragma HLS ARRAY_PARTITION variable=l0_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_3 complete dim=1

            // Initialise partials (handles in_ch/16 < MAX_IC_TILES with zero padding)
            C1X1S_INIT: for (int t = 0; t < MAX_IC_TILES; t++) {
                #pragma HLS UNROLL
                partial0[t] = 0.0f;
                partial1[t] = 0.0f;
                partial2[t] = 0.0f;
                partial3[t] = 0.0f;
            }

            C1X1S_IC_T: for (int t = 0; t < in_ch / IC_TILE; t++) {
                #pragma HLS PIPELINE II=1
                acc_t tile0=0.0f, tile1=0.0f, tile2=0.0f, tile3=0.0f;
                C1X1S_IC_K: for (int k = 0; k < IC_TILE; k++) {
                    #pragma HLS UNROLL
                    acc_t x = (acc_t)input[hw * in_ch + t * IC_TILE + k];
                    tile0 += x * (acc_t)(float)weights[ oc      * in_ch + t * IC_TILE + k];
                    tile1 += x * (acc_t)(float)weights[(oc + 1) * in_ch + t * IC_TILE + k];
                    tile2 += x * (acc_t)(float)weights[(oc + 2) * in_ch + t * IC_TILE + k];
                    tile3 += x * (acc_t)(float)weights[(oc + 3) * in_ch + t * IC_TILE + k];
                }
                partial0[t] = tile0;
                partial1[t] = tile1;
                partial2[t] = tile2;
                partial3[t] = tile3;
            }

            // 5-level binary tree reduction (32→16→8→4→2→1), all 4 OCs in parallel
            for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                l0_0[i] = partial0[2*i] + partial0[2*i+1];
                l0_1[i] = partial1[2*i] + partial1[2*i+1];
                l0_2[i] = partial2[2*i] + partial2[2*i+1];
                l0_3[i] = partial3[2*i] + partial3[2*i+1];
            }
            for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                l1_0[i] = l0_0[2*i] + l0_0[2*i+1];
                l1_1[i] = l0_1[2*i] + l0_1[2*i+1];
                l1_2[i] = l0_2[2*i] + l0_2[2*i+1];
                l1_3[i] = l0_3[2*i] + l0_3[2*i+1];
            }
            for (int i = 0; i < 4; i++) {
                #pragma HLS UNROLL
                l2_0[i] = l1_0[2*i] + l1_0[2*i+1];
                l2_1[i] = l1_1[2*i] + l1_1[2*i+1];
                l2_2[i] = l1_2[2*i] + l1_2[2*i+1];
                l2_3[i] = l1_3[2*i] + l1_3[2*i+1];
            }
            for (int i = 0; i < 2; i++) {
                #pragma HLS UNROLL
                l3_0[i] = l2_0[2*i] + l2_0[2*i+1];
                l3_1[i] = l2_1[2*i] + l2_1[2*i+1];
                l3_2[i] = l2_2[2*i] + l2_2[2*i+1];
                l3_3[i] = l2_3[2*i] + l2_3[2*i+1];
            }
            acc_t acc0 = l3_0[0] + l3_0[1];
            acc_t acc1 = l3_1[0] + l3_1[1];
            acc_t acc2 = l3_2[0] + l3_2[1];
            acc_t acc3 = l3_3[0] + l3_3[1];

            output[hw*out_ch + oc    ] = silu((act_t)((acc_t)acc0*(acc_t)bn_scale[oc    ]+(acc_t)bn_bias[oc    ]));
            output[hw*out_ch + oc + 1] = silu((act_t)((acc_t)acc1*(acc_t)bn_scale[oc + 1]+(acc_t)bn_bias[oc + 1]));
            output[hw*out_ch + oc + 2] = silu((act_t)((acc_t)acc2*(acc_t)bn_scale[oc + 2]+(acc_t)bn_bias[oc + 2]));
            output[hw*out_ch + oc + 3] = silu((act_t)((acc_t)acc3*(acc_t)bn_scale[oc + 3]+(acc_t)bn_bias[oc + 3]));
        }
    }
}

// ============================================================
// 1x1 CONV + BN  (HWC, OC_TILE=4, partial acc, 5-level tree, no activation)
// ============================================================
inline void conv1x1_bn(
    const act_t*    input,
          act_t*    output,
    const weight_t* weights,
    const meta_t*   bn_scale,
    const meta_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    #pragma HLS INLINE
    C1X1_OC: for (int oc = 0; oc < out_ch; oc += OC_TILE) {
        C1X1_HW: for (int hw = 0; hw < H * W; hw++) {

            // All local arrays declared first — before any statements
            acc_t partial0[MAX_IC_TILES], partial1[MAX_IC_TILES];
            acc_t partial2[MAX_IC_TILES], partial3[MAX_IC_TILES];
            #pragma HLS ARRAY_PARTITION variable=partial0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial3 complete dim=1

            acc_t l0_0[16], l0_1[16], l0_2[16], l0_3[16];
            acc_t l1_0[ 8], l1_1[ 8], l1_2[ 8], l1_3[ 8];
            acc_t l2_0[ 4], l2_1[ 4], l2_2[ 4], l2_3[ 4];
            acc_t l3_0[ 2], l3_1[ 2], l3_2[ 2], l3_3[ 2];
            #pragma HLS ARRAY_PARTITION variable=l0_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l0_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l1_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l2_3 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=l3_3 complete dim=1

            C1X1_INIT: for (int t = 0; t < MAX_IC_TILES; t++) {
                #pragma HLS UNROLL
                partial0[t] = 0.0f;
                partial1[t] = 0.0f;
                partial2[t] = 0.0f;
                partial3[t] = 0.0f;
            }

            C1X1_IC_T: for (int t = 0; t < in_ch / IC_TILE; t++) {
                #pragma HLS PIPELINE II=1
                acc_t tile0=0.0f, tile1=0.0f, tile2=0.0f, tile3=0.0f;
                C1X1_IC_K: for (int k = 0; k < IC_TILE; k++) {
                    #pragma HLS UNROLL
                    acc_t x = (acc_t)input[hw * in_ch + t * IC_TILE + k];
                    tile0 += x * (acc_t)(float)weights[ oc      * in_ch + t * IC_TILE + k];
                    tile1 += x * (acc_t)(float)weights[(oc + 1) * in_ch + t * IC_TILE + k];
                    tile2 += x * (acc_t)(float)weights[(oc + 2) * in_ch + t * IC_TILE + k];
                    tile3 += x * (acc_t)(float)weights[(oc + 3) * in_ch + t * IC_TILE + k];
                }
                partial0[t] = tile0;
                partial1[t] = tile1;
                partial2[t] = tile2;
                partial3[t] = tile3;
            }

            for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                l0_0[i] = partial0[2*i] + partial0[2*i+1];
                l0_1[i] = partial1[2*i] + partial1[2*i+1];
                l0_2[i] = partial2[2*i] + partial2[2*i+1];
                l0_3[i] = partial3[2*i] + partial3[2*i+1];
            }
            for (int i = 0; i < 8; i++) {
                #pragma HLS UNROLL
                l1_0[i] = l0_0[2*i] + l0_0[2*i+1];
                l1_1[i] = l0_1[2*i] + l0_1[2*i+1];
                l1_2[i] = l0_2[2*i] + l0_2[2*i+1];
                l1_3[i] = l0_3[2*i] + l0_3[2*i+1];
            }
            for (int i = 0; i < 4; i++) {
                #pragma HLS UNROLL
                l2_0[i] = l1_0[2*i] + l1_0[2*i+1];
                l2_1[i] = l1_1[2*i] + l1_1[2*i+1];
                l2_2[i] = l1_2[2*i] + l1_2[2*i+1];
                l2_3[i] = l1_3[2*i] + l1_3[2*i+1];
            }
            for (int i = 0; i < 2; i++) {
                #pragma HLS UNROLL
                l3_0[i] = l2_0[2*i] + l2_0[2*i+1];
                l3_1[i] = l2_1[2*i] + l2_1[2*i+1];
                l3_2[i] = l2_2[2*i] + l2_2[2*i+1];
                l3_3[i] = l2_3[2*i] + l2_3[2*i+1];
            }
            acc_t acc0 = l3_0[0]+l3_0[1];
            acc_t acc1 = l3_1[0]+l3_1[1];
            acc_t acc2 = l3_2[0]+l3_2[1];
            acc_t acc3 = l3_3[0]+l3_3[1];

            output[hw*out_ch + oc    ] = (act_t)((acc_t)acc0*(acc_t)bn_scale[oc    ]+(acc_t)bn_bias[oc    ]);
            output[hw*out_ch + oc + 1] = (act_t)((acc_t)acc1*(acc_t)bn_scale[oc + 1]+(acc_t)bn_bias[oc + 1]);
            output[hw*out_ch + oc + 2] = (act_t)((acc_t)acc2*(acc_t)bn_scale[oc + 2]+(acc_t)bn_bias[oc + 2]);
            output[hw*out_ch + oc + 3] = (act_t)((acc_t)acc3*(acc_t)bn_scale[oc + 3]+(acc_t)bn_bias[oc + 3]);
        }
    }
}

// ============================================================
// 1x1 CONV plain (CHW, no BN — FPN/head, unchanged)
// ============================================================
inline void conv1x1_plain(
    const act_t* input, act_t* output,
    const weight_t* weights, const meta_t* bias,
    int in_ch, int out_ch, int H, int W
) {
    C1P_OC: for (int oc = 0; oc < out_ch; oc++) {
        C1P_HW: for (int hw = 0; hw < H * W; hw++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)bias[oc];
            C1P_IC: for (int ic = 0; ic < in_ch; ic++) {
                #pragma HLS UNROLL factor=16
                acc += (acc_t)input[ic*H*W+hw]*(acc_t)(float)weights[oc*in_ch+ic];
            }
            output[oc*H*W+hw] = (act_t)acc;
        }
    }
}

// ============================================================
// DEPTHWISE 3x3 CONV + BN + SiLU  (HWC layout)
// 9 reads at distinct URAM banks: step = ch%inferred_factor, gcd=1 → II=1 target
// ============================================================
inline void dw_conv3x3_bn_silu(
    const act_t*    input,
          act_t*    output,
    const weight_t* weights,
    const meta_t*   bn_scale,
    const meta_t*   bn_bias,
    int ch, int H, int W, int stride
) {
    #pragma HLS INLINE
    int out_H = DIV_CEIL(H, stride), out_W = DIV_CEIL(W, stride);
    DW_C: for (int c = 0; c < ch; c++) {
        DW_OH: for (int oh = 0; oh < out_H; oh++) {
            DW_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DW_KH: for (int kh = 0; kh < 3; kh++) {
                    #pragma HLS UNROLL
                    DW_KW: for (int kw = 0; kw < 3; kw++) {
                        #pragma HLS UNROLL
                        int ih = oh * stride + kh - 1;
                        int iw = ow * stride + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            acc += (acc_t)input[(ih*W+iw)*ch + c]
                                 * (acc_t)(float)weights[c*9 + kh*3 + kw];
                    }
                }
                output[(oh*out_W+ow)*ch + c] =
                    silu((act_t)((acc_t)acc*(acc_t)bn_scale[c]+(acc_t)bn_bias[c]));
            }
        }
    }
}

inline void dw_conv3x3_bn(
    const act_t*    input,
          act_t*    output,
    const weight_t* weights,
    const meta_t*   bn_scale,
    const meta_t*   bn_bias,
    int ch, int H, int W, int stride
) {
    #pragma HLS INLINE
    int out_H = DIV_CEIL(H, stride), out_W = DIV_CEIL(W, stride);
    DWB_C: for (int c = 0; c < ch; c++) {
        DWB_OH: for (int oh = 0; oh < out_H; oh++) {
            DWB_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DWB_KH: for (int kh = 0; kh < 3; kh++) {
                    #pragma HLS UNROLL
                    DWB_KW: for (int kw = 0; kw < 3; kw++) {
                        #pragma HLS UNROLL
                        int ih = oh * stride + kh - 1;
                        int iw = ow * stride + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            acc += (acc_t)input[(ih*W+iw)*ch + c]
                                 * (acc_t)(float)weights[c*9 + kh*3 + kw];
                    }
                }
                output[(oh*out_W+ow)*ch + c] =
                    (act_t)((acc_t)acc*(acc_t)bn_scale[c]+(acc_t)bn_bias[c]);
            }
        }
    }
}

// ============================================================
// UTILITY FUNCTIONS (unchanged)
// ============================================================
inline void elem_add_inplace(act_t* a, const act_t* b, int n) {
    #pragma HLS INLINE
    EAI: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        a[i] = a[i] + b[i];
    }
}

inline void upsample_2x(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    UP_C: for (int c = 0; c < ch; c++) {
        UP_H: for (int h = 0; h < H; h++) {
            UP_W: for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                act_t v = src[c*H*W+h*W+w];
                dst[c*(2*H)*(2*W)+(2*h  )*(2*W)+(2*w  )] = v;
                dst[c*(2*H)*(2*W)+(2*h  )*(2*W)+(2*w+1)] = v;
                dst[c*(2*H)*(2*W)+(2*h+1)*(2*W)+(2*w  )] = v;
                dst[c*(2*H)*(2*W)+(2*h+1)*(2*W)+(2*w+1)] = v;
            }
        }
    }
}

inline void maxpool2x2(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    int oH=H/2, oW=W/2;
    MP_C: for (int c = 0; c < ch; c++) {
        MP_H: for (int oh = 0; oh < oH; oh++) {
            MP_W: for (int ow = 0; ow < oW; ow++) {
                #pragma HLS PIPELINE II=1
                dst[c*oH*oW+oh*oW+ow] = src[c*H*W+(oh*2)*W+(ow*2)];
            }
        }
    }
}

inline void buf_copy(act_t* dst, const act_t* src, int n) {
    #pragma HLS INLINE
    BC: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

#endif // FPGA_UTILS_H
