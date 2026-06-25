/*
 * fpga_utils.h  (PTQ INT8 — v6, FPN_OC_TILE=16, MBConv OC_TILE=4, MAX_IC_TILES=32)
 *
 * Shared utility functions for all HTDet FPGA modules:
 *   mbconv_80/40/20  — MBConv blocks (HWC layout, int8 weights)
 *   fpn_192          — FPN neck      (CHW layout, int8 weights)
 *
 * MBConv uses HWC layout; FPN uses CHW layout (matching backbone outputs).
 * conv1x1_bn_silu / conv1x1_bn / dw_conv3x3_bn_silu / dw_conv3x3_bn : HWC  (MBConv path, OC_TILE=4)
 * conv1x1_plain / fpn_conv3x3                                         : CHW  (FPN path, FPN_OC_TILE=16)
 */

#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include "fpga_types.h"
#include <cmath>

#define MAX_IC_TILES 32
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
// BN / LAYER NORM / SOFTMAX / LINEAR
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
// 3×3 CONV  (CHW layout — stem, head)
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
// 1×1 CONV + BN + SiLU  (HWC layout — MBConv path)
// OC_TILE=4 (MBConv), MAX_IC_TILES=32, 5-level binary reduction tree
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

            C1X1S_INIT: for (int t = 0; t < MAX_IC_TILES; t++) {
                #pragma HLS UNROLL
                partial0[t] = 0.0f; partial1[t] = 0.0f;
                partial2[t] = 0.0f; partial3[t] = 0.0f;
            }

            C1X1S_IC_T: for (int t = 0; t < in_ch / IC_TILE; t++) {
                #pragma HLS PIPELINE II=1
                acc_t tile0=0,tile1=0,tile2=0,tile3=0;
                C1X1S_IC_K: for (int k = 0; k < IC_TILE; k++) {
                    #pragma HLS UNROLL
                    acc_t x = (acc_t)input[hw*in_ch + t*IC_TILE + k];
                    tile0 += x * (acc_t)(float)weights[ oc      *in_ch + t*IC_TILE + k];
                    tile1 += x * (acc_t)(float)weights[(oc+1)*in_ch + t*IC_TILE + k];
                    tile2 += x * (acc_t)(float)weights[(oc+2)*in_ch + t*IC_TILE + k];
                    tile3 += x * (acc_t)(float)weights[(oc+3)*in_ch + t*IC_TILE + k];
                }
                partial0[t] = tile0; partial1[t] = tile1;
                partial2[t] = tile2; partial3[t] = tile3;
            }

            for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                l0_0[i]=partial0[2*i]+partial0[2*i+1]; l0_1[i]=partial1[2*i]+partial1[2*i+1];
                l0_2[i]=partial2[2*i]+partial2[2*i+1]; l0_3[i]=partial3[2*i]+partial3[2*i+1];
            }
            for (int i = 0; i < 8;  i++) {
                #pragma HLS UNROLL
                l1_0[i]=l0_0[2*i]+l0_0[2*i+1]; l1_1[i]=l0_1[2*i]+l0_1[2*i+1];
                l1_2[i]=l0_2[2*i]+l0_2[2*i+1]; l1_3[i]=l0_3[2*i]+l0_3[2*i+1];
            }
            for (int i = 0; i < 4;  i++) {
                #pragma HLS UNROLL
                l2_0[i]=l1_0[2*i]+l1_0[2*i+1]; l2_1[i]=l1_1[2*i]+l1_1[2*i+1];
                l2_2[i]=l1_2[2*i]+l1_2[2*i+1]; l2_3[i]=l1_3[2*i]+l1_3[2*i+1];
            }
            for (int i = 0; i < 2;  i++) {
                #pragma HLS UNROLL
                l3_0[i]=l2_0[2*i]+l2_0[2*i+1]; l3_1[i]=l2_1[2*i]+l2_1[2*i+1];
                l3_2[i]=l2_2[2*i]+l2_2[2*i+1]; l3_3[i]=l2_3[2*i]+l2_3[2*i+1];
            }
            acc_t a0=l3_0[0]+l3_0[1], a1=l3_1[0]+l3_1[1];
            acc_t a2=l3_2[0]+l3_2[1], a3=l3_3[0]+l3_3[1];

            output[hw*out_ch+oc  ] = silu((act_t)(a0*(acc_t)bn_scale[oc  ]+(acc_t)bn_bias[oc  ]));
            output[hw*out_ch+oc+1] = silu((act_t)(a1*(acc_t)bn_scale[oc+1]+(acc_t)bn_bias[oc+1]));
            output[hw*out_ch+oc+2] = silu((act_t)(a2*(acc_t)bn_scale[oc+2]+(acc_t)bn_bias[oc+2]));
            output[hw*out_ch+oc+3] = silu((act_t)(a3*(acc_t)bn_scale[oc+3]+(acc_t)bn_bias[oc+3]));
        }
    }
}

// ============================================================
// 1×1 CONV + BN  (HWC, no activation — MBConv proj)
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

            acc_t partial0[MAX_IC_TILES], partial1[MAX_IC_TILES];
            acc_t partial2[MAX_IC_TILES], partial3[MAX_IC_TILES];
            #pragma HLS ARRAY_PARTITION variable=partial0 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial1 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial2 complete dim=1
            #pragma HLS ARRAY_PARTITION variable=partial3 complete dim=1

            acc_t l0_0[16],l0_1[16],l0_2[16],l0_3[16];
            acc_t l1_0[ 8],l1_1[ 8],l1_2[ 8],l1_3[ 8];
            acc_t l2_0[ 4],l2_1[ 4],l2_2[ 4],l2_3[ 4];
            acc_t l3_0[ 2],l3_1[ 2],l3_2[ 2],l3_3[ 2];
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
                partial0[t]=0.0f; partial1[t]=0.0f;
                partial2[t]=0.0f; partial3[t]=0.0f;
            }

            C1X1_IC_T: for (int t = 0; t < in_ch / IC_TILE; t++) {
                #pragma HLS PIPELINE II=1
                acc_t tile0=0,tile1=0,tile2=0,tile3=0;
                C1X1_IC_K: for (int k = 0; k < IC_TILE; k++) {
                    #pragma HLS UNROLL
                    acc_t x = (acc_t)input[hw*in_ch + t*IC_TILE + k];
                    tile0 += x * (acc_t)(float)weights[ oc      *in_ch + t*IC_TILE + k];
                    tile1 += x * (acc_t)(float)weights[(oc+1)*in_ch + t*IC_TILE + k];
                    tile2 += x * (acc_t)(float)weights[(oc+2)*in_ch + t*IC_TILE + k];
                    tile3 += x * (acc_t)(float)weights[(oc+3)*in_ch + t*IC_TILE + k];
                }
                partial0[t]=tile0; partial1[t]=tile1;
                partial2[t]=tile2; partial3[t]=tile3;
            }

            for (int i = 0; i < 16; i++) {
                #pragma HLS UNROLL
                l0_0[i]=partial0[2*i]+partial0[2*i+1]; l0_1[i]=partial1[2*i]+partial1[2*i+1];
                l0_2[i]=partial2[2*i]+partial2[2*i+1]; l0_3[i]=partial3[2*i]+partial3[2*i+1];
            }
            for (int i = 0; i < 8;  i++) {
                #pragma HLS UNROLL
                l1_0[i]=l0_0[2*i]+l0_0[2*i+1]; l1_1[i]=l0_1[2*i]+l0_1[2*i+1];
                l1_2[i]=l0_2[2*i]+l0_2[2*i+1]; l1_3[i]=l0_3[2*i]+l0_3[2*i+1];
            }
            for (int i = 0; i < 4;  i++) {
                #pragma HLS UNROLL
                l2_0[i]=l1_0[2*i]+l1_0[2*i+1]; l2_1[i]=l1_1[2*i]+l1_1[2*i+1];
                l2_2[i]=l1_2[2*i]+l1_2[2*i+1]; l2_3[i]=l1_3[2*i]+l1_3[2*i+1];
            }
            for (int i = 0; i < 2;  i++) {
                #pragma HLS UNROLL
                l3_0[i]=l2_0[2*i]+l2_0[2*i+1]; l3_1[i]=l2_1[2*i]+l2_1[2*i+1];
                l3_2[i]=l2_2[2*i]+l2_2[2*i+1]; l3_3[i]=l2_3[2*i]+l2_3[2*i+1];
            }
            acc_t a0=l3_0[0]+l3_0[1], a1=l3_1[0]+l3_1[1];
            acc_t a2=l3_2[0]+l3_2[1], a3=l3_3[0]+l3_3[1];

            output[hw*out_ch+oc  ] = (act_t)(a0*(acc_t)bn_scale[oc  ]+(acc_t)bn_bias[oc  ]);
            output[hw*out_ch+oc+1] = (act_t)(a1*(acc_t)bn_scale[oc+1]+(acc_t)bn_bias[oc+1]);
            output[hw*out_ch+oc+2] = (act_t)(a2*(acc_t)bn_scale[oc+2]+(acc_t)bn_bias[oc+2]);
            output[hw*out_ch+oc+3] = (act_t)(a3*(acc_t)bn_scale[oc+3]+(acc_t)bn_bias[oc+3]);
        }
    }
}

// ============================================================
// 1×1 CONV plain (CHW layout — FPN lateral, no BN)
// Weight layout: [out_ch][in_ch]  (row-major OC×IC)
// Bias layout:   [out_ch]
//
// FPN_OC_TILE=16: compute 16 output channels per outer trip.
// Reduces C1P_OC outer iterations to 12 (vs 192 at OC_TILE=1).
// 16 independent acc chains → 4 DSPs, II=5 (mac_muladd latency=4 → II≥5).
// Requires w_conv partitioned cyclic factor=17 in the top-level TCL:
//   16 OC reads stride in_ch → in_ch mod 17 is coprime with 17 for all
//   FPN in_ch values (640→11, 128→9, 96→11, 64→13; all gcd=1)
//   → all 16 reads hit distinct banks → II_BRAM=1 < II_DSP=5. ✓
// Cyclic-16 would put all reads in the same bank (in_ch is always multiple of 16).
// out_ch must be divisible by 16 (FPN_OUT_CH=192/16=12 ✓).
// ============================================================
inline void conv1x1_plain(
    const act_t*    input,    // CHW: [in_ch * H * W]
          act_t*    output,   // CHW: [out_ch * H * W]
    const weight_t* weights,  // [out_ch * in_ch]   int8
    const meta_t*   bias,     // [out_ch]            float
    int in_ch, int out_ch, int H, int W
) {
    C1P_OC: for (int oc = 0; oc < out_ch; oc += 16) {
        C1P_HW: for (int hw = 0; hw < H * W; hw++) {
            acc_t acc0  = (acc_t)bias[oc   ], acc1  = (acc_t)bias[oc+ 1];
            acc_t acc2  = (acc_t)bias[oc+ 2], acc3  = (acc_t)bias[oc+ 3];
            acc_t acc4  = (acc_t)bias[oc+ 4], acc5  = (acc_t)bias[oc+ 5];
            acc_t acc6  = (acc_t)bias[oc+ 6], acc7  = (acc_t)bias[oc+ 7];
            acc_t acc8  = (acc_t)bias[oc+ 8], acc9  = (acc_t)bias[oc+ 9];
            acc_t acc10 = (acc_t)bias[oc+10], acc11 = (acc_t)bias[oc+11];
            acc_t acc12 = (acc_t)bias[oc+12], acc13 = (acc_t)bias[oc+13];
            acc_t acc14 = (acc_t)bias[oc+14], acc15 = (acc_t)bias[oc+15];
            C1P_IC: for (int ic = 0; ic < in_ch; ic++) {
                #pragma HLS PIPELINE II=5
                acc_t x = (acc_t)input[ic*H*W + hw];
                acc0  += x * (acc_t)(float)weights[(oc   )*in_ch + ic];
                acc1  += x * (acc_t)(float)weights[(oc+ 1)*in_ch + ic];
                acc2  += x * (acc_t)(float)weights[(oc+ 2)*in_ch + ic];
                acc3  += x * (acc_t)(float)weights[(oc+ 3)*in_ch + ic];
                acc4  += x * (acc_t)(float)weights[(oc+ 4)*in_ch + ic];
                acc5  += x * (acc_t)(float)weights[(oc+ 5)*in_ch + ic];
                acc6  += x * (acc_t)(float)weights[(oc+ 6)*in_ch + ic];
                acc7  += x * (acc_t)(float)weights[(oc+ 7)*in_ch + ic];
                acc8  += x * (acc_t)(float)weights[(oc+ 8)*in_ch + ic];
                acc9  += x * (acc_t)(float)weights[(oc+ 9)*in_ch + ic];
                acc10 += x * (acc_t)(float)weights[(oc+10)*in_ch + ic];
                acc11 += x * (acc_t)(float)weights[(oc+11)*in_ch + ic];
                acc12 += x * (acc_t)(float)weights[(oc+12)*in_ch + ic];
                acc13 += x * (acc_t)(float)weights[(oc+13)*in_ch + ic];
                acc14 += x * (acc_t)(float)weights[(oc+14)*in_ch + ic];
                acc15 += x * (acc_t)(float)weights[(oc+15)*in_ch + ic];
            }
            output[(oc   )*H*W + hw] = (act_t)acc0;
            output[(oc+ 1)*H*W + hw] = (act_t)acc1;
            output[(oc+ 2)*H*W + hw] = (act_t)acc2;
            output[(oc+ 3)*H*W + hw] = (act_t)acc3;
            output[(oc+ 4)*H*W + hw] = (act_t)acc4;
            output[(oc+ 5)*H*W + hw] = (act_t)acc5;
            output[(oc+ 6)*H*W + hw] = (act_t)acc6;
            output[(oc+ 7)*H*W + hw] = (act_t)acc7;
            output[(oc+ 8)*H*W + hw] = (act_t)acc8;
            output[(oc+ 9)*H*W + hw] = (act_t)acc9;
            output[(oc+10)*H*W + hw] = (act_t)acc10;
            output[(oc+11)*H*W + hw] = (act_t)acc11;
            output[(oc+12)*H*W + hw] = (act_t)acc12;
            output[(oc+13)*H*W + hw] = (act_t)acc13;
            output[(oc+14)*H*W + hw] = (act_t)acc14;
            output[(oc+15)*H*W + hw] = (act_t)acc15;
        }
    }
}

// ============================================================
// 3×3 CONV plain (CHW layout — FPN output conv, no BN, no activation)
// in_ch = out_ch = FPN_OUT_CH = 192
// Weight layout: [out_ch][in_ch][3][3]  int8
// Bias layout:   [out_ch]               float
//
// FPN_OC_TILE=16: compute 16 output channels per inner pipeline run.
// Reduces outer FPN3_OC×OH×OW trips by 16× vs OC_TILE=1 (P2: 307,200→19,200).
// 16 independent acc chains → 4 DSPs, II=5 (mac_muladd latency=4 → II≥5).
// Requires w_conv partitioned cyclic factor=17 in the top-level TCL:
//   16 OC reads stride FPN_OUT_CH×9=1728 → 1728 mod 17=11, gcd(11,17)=1
//   → all 16 reads hit distinct banks → II_BRAM=1 < II_DSP=5. ✓
// Without the partition: dual-port BRAM ≤2 reads/cycle → ceil(16/2)=8 cycles
//   → II degrades to 8 → only ~10% gain vs OC_TILE=8. ❌
// FPN_OUT_CH=192 must be divisible by 16 (192/16=12 ✓).
// ============================================================
inline void fpn_conv3x3(
    const act_t*    input,    // CHW: [FPN_OUT_CH * H * W]
          act_t*    output,   // CHW: [FPN_OUT_CH * H * W]
    const weight_t* weights,  // [FPN_OUT_CH * FPN_OUT_CH * 9]  int8
    const meta_t*   bias,     // [FPN_OUT_CH]                    float
    int H, int W
) {
    FPN3_OC: for (int oc = 0; oc < FPN_OUT_CH; oc += 16) {
        FPN3_OH: for (int oh = 0; oh < H; oh++) {
            FPN3_OW: for (int ow = 0; ow < W; ow++) {
                acc_t acc0 =0, acc1 =0, acc2 =0, acc3 =0;
                acc_t acc4 =0, acc5 =0, acc6 =0, acc7 =0;
                acc_t acc8 =0, acc9 =0, acc10=0, acc11=0;
                acc_t acc12=0, acc13=0, acc14=0, acc15=0;
                FPN3_IC: for (int ic = 0; ic < FPN_OUT_CH; ic++) {
                    FPN3_KH: for (int kh = 0; kh < 3; kh++) {
                        FPN3_KW: for (int kw = 0; kw < 3; kw++) {
                            #pragma HLS PIPELINE II=5
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                act_t v = input[ic*H*W + ih*W + iw];
                                acc0  += (acc_t)v*(acc_t)(float)weights[(( oc   )*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc1  += (acc_t)v*(acc_t)(float)weights[((oc+ 1)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc2  += (acc_t)v*(acc_t)(float)weights[((oc+ 2)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc3  += (acc_t)v*(acc_t)(float)weights[((oc+ 3)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc4  += (acc_t)v*(acc_t)(float)weights[((oc+ 4)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc5  += (acc_t)v*(acc_t)(float)weights[((oc+ 5)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc6  += (acc_t)v*(acc_t)(float)weights[((oc+ 6)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc7  += (acc_t)v*(acc_t)(float)weights[((oc+ 7)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc8  += (acc_t)v*(acc_t)(float)weights[((oc+ 8)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc9  += (acc_t)v*(acc_t)(float)weights[((oc+ 9)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc10 += (acc_t)v*(acc_t)(float)weights[((oc+10)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc11 += (acc_t)v*(acc_t)(float)weights[((oc+11)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc12 += (acc_t)v*(acc_t)(float)weights[((oc+12)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc13 += (acc_t)v*(acc_t)(float)weights[((oc+13)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc14 += (acc_t)v*(acc_t)(float)weights[((oc+14)*FPN_OUT_CH+ic)*9+kh*3+kw];
                                acc15 += (acc_t)v*(acc_t)(float)weights[((oc+15)*FPN_OUT_CH+ic)*9+kh*3+kw];
                            }
                        }
                    }
                }
                output[( oc   )*H*W+oh*W+ow] = (act_t)(acc0  + (acc_t)bias[ oc   ]);
                output[(oc+ 1)*H*W+oh*W+ow] = (act_t)(acc1  + (acc_t)bias[oc+ 1]);
                output[(oc+ 2)*H*W+oh*W+ow] = (act_t)(acc2  + (acc_t)bias[oc+ 2]);
                output[(oc+ 3)*H*W+oh*W+ow] = (act_t)(acc3  + (acc_t)bias[oc+ 3]);
                output[(oc+ 4)*H*W+oh*W+ow] = (act_t)(acc4  + (acc_t)bias[oc+ 4]);
                output[(oc+ 5)*H*W+oh*W+ow] = (act_t)(acc5  + (acc_t)bias[oc+ 5]);
                output[(oc+ 6)*H*W+oh*W+ow] = (act_t)(acc6  + (acc_t)bias[oc+ 6]);
                output[(oc+ 7)*H*W+oh*W+ow] = (act_t)(acc7  + (acc_t)bias[oc+ 7]);
                output[(oc+ 8)*H*W+oh*W+ow] = (act_t)(acc8  + (acc_t)bias[oc+ 8]);
                output[(oc+ 9)*H*W+oh*W+ow] = (act_t)(acc9  + (acc_t)bias[oc+ 9]);
                output[(oc+10)*H*W+oh*W+ow] = (act_t)(acc10 + (acc_t)bias[oc+10]);
                output[(oc+11)*H*W+oh*W+ow] = (act_t)(acc11 + (acc_t)bias[oc+11]);
                output[(oc+12)*H*W+oh*W+ow] = (act_t)(acc12 + (acc_t)bias[oc+12]);
                output[(oc+13)*H*W+oh*W+ow] = (act_t)(acc13 + (acc_t)bias[oc+13]);
                output[(oc+14)*H*W+oh*W+ow] = (act_t)(acc14 + (acc_t)bias[oc+14]);
                output[(oc+15)*H*W+oh*W+ow] = (act_t)(acc15 + (acc_t)bias[oc+15]);
            }
        }
    }
}

// ============================================================
// DEPTHWISE 3×3 CONV + BN + SiLU  (HWC layout — MBConv)
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
    int out_H = DIV_CEIL(H,stride), out_W = DIV_CEIL(W,stride);
    DW_C: for (int c = 0; c < ch; c++) {
        DW_OH: for (int oh = 0; oh < out_H; oh++) {
            DW_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DW_KH: for (int kh = 0; kh < 3; kh++) {
                    #pragma HLS UNROLL
                    DW_KW: for (int kw = 0; kw < 3; kw++) {
                        #pragma HLS UNROLL
                        int ih=oh*stride+kh-1, iw=ow*stride+kw-1;
                        if (ih>=0&&ih<H&&iw>=0&&iw<W)
                            acc += (acc_t)input[(ih*W+iw)*ch+c]*(acc_t)(float)weights[c*9+kh*3+kw];
                    }
                }
                output[(oh*out_W+ow)*ch+c] =
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
    int out_H = DIV_CEIL(H,stride), out_W = DIV_CEIL(W,stride);
    DWB_C: for (int c = 0; c < ch; c++) {
        DWB_OH: for (int oh = 0; oh < out_H; oh++) {
            DWB_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DWB_KH: for (int kh = 0; kh < 3; kh++) {
                    #pragma HLS UNROLL
                    DWB_KW: for (int kw = 0; kw < 3; kw++) {
                        #pragma HLS UNROLL
                        int ih=oh*stride+kh-1, iw=ow*stride+kw-1;
                        if (ih>=0&&ih<H&&iw>=0&&iw<W)
                            acc += (acc_t)input[(ih*W+iw)*ch+c]*(acc_t)(float)weights[c*9+kh*3+kw];
                    }
                }
                output[(oh*out_W+ow)*ch+c] =
                    (act_t)((acc_t)acc*(acc_t)bn_scale[c]+(acc_t)bn_bias[c]);
            }
        }
    }
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
inline void elem_add_inplace(act_t* a, const act_t* b, int n) {
    #pragma HLS INLINE
    EAI: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        a[i] = a[i] + b[i];
    }
}

// Nearest-neighbour 2× upsample (CHW layout)
inline void upsample_2x(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    UP_C: for (int c = 0; c < ch; c++) {
        UP_H: for (int h = 0; h < H; h++) {
            UP_W: for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                act_t v = src[c*H*W + h*W + w];
                dst[c*(2*H)*(2*W) + (2*h  )*(2*W) + (2*w  )] = v;
                dst[c*(2*H)*(2*W) + (2*h  )*(2*W) + (2*w+1)] = v;
                dst[c*(2*H)*(2*W) + (2*h+1)*(2*W) + (2*w  )] = v;
                dst[c*(2*H)*(2*W) + (2*h+1)*(2*W) + (2*w+1)] = v;
            }
        }
    }
}

// Nearest-neighbour 2× upsample + add in-place (CHW layout).
// dst[c*(2H)*(2W) + oh*(2W) + ow] += src[c*H*W + (oh/2)*W + (ow/2)]
// Eliminates the temporary up-buffer needed by upsample_2x + elem_add_inplace.
// Caller must declare: #pragma HLS DEPENDENCE variable=<dst_var> inter false
// so HLS confirms no inter-iteration hazard (each dst address written exactly once).
inline void add_upsampled_2x_inplace(
    const act_t* src, act_t* dst,
    int ch, int H, int W
) {
    #pragma HLS INLINE
    int oH = 2*H, oW = 2*W;
    ADD_UP_C: for (int c = 0; c < ch; c++) {
        ADD_UP_OH: for (int oh = 0; oh < oH; oh++) {
            ADD_UP_OW: for (int ow = 0; ow < oW; ow++) {
                #pragma HLS PIPELINE II=1
                dst[c*oH*oW + oh*oW + ow] +=
                    src[c*H*W + (oh/2)*W + (ow/2)];
            }
        }
    }
}

// Stride-2 subsampling (CHW layout) — P5→P6.
// MMDetection FPN extra level uses MaxPool2d(kernel_size=1, stride=2),
// which is equivalent to taking every other pixel (top-left of each 2×2 block).
inline void maxpool2x2(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    int oH = H/2, oW = W/2;
    MP_C: for (int c = 0; c < ch; c++) {
        MP_H: for (int oh = 0; oh < oH; oh++) {
            MP_W: for (int ow = 0; ow < oW; ow++) {
                #pragma HLS PIPELINE II=1
                dst[c*oH*oW + oh*oW + ow] = src[c*H*W + (2*oh)*W + (2*ow)];
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
