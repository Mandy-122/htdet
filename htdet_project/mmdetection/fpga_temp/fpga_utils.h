/*
 * fpga_utils.h
 * Activation functions, normalization, and convolution primitives.
 * All functions are HLS-inlined; used by backbone, FPN, and head.
 */

#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include "fpga_types.h"
#include <cmath>

// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================

// ReLU
inline act_t relu(act_t x) {
    #pragma HLS INLINE
    return (x > (act_t)0) ? x : (act_t)0;
}

// SiLU (Swish): x * sigmoid(x)
// Exact implementation — used for C-sim.
// For FPGA synthesis swap to the piecewise-linear approximation below.
inline act_t silu(act_t x) {
    #pragma HLS INLINE
    return (act_t)((float)x / (1.0f + expf(-(float)x)));
}

// Sigmoid for classification output — exact implementation for C-sim.
// For FPGA synthesis swap to the piecewise-linear approximation below.
inline score_t sigmoid(score_t x) {
    #pragma HLS INLINE
    return (score_t)(1.0f / (1.0f + expf(-(float)x)));
}

// --- FPGA piecewise-linear approximations (uncomment for synthesis) ---
// inline act_t silu(act_t x) {
//     act_t sig;
//     if      (x < (act_t)(-4.0f)) sig = (act_t)0.0f;
//     else if (x > (act_t)( 4.0f)) sig = (act_t)1.0f;
//     else { sig = (act_t)0.5f + x * (act_t)0.125f; }
//     return x * sig;
// }
// inline score_t sigmoid(score_t x) {
//     if (x < (score_t)(-4.0f)) return (score_t)0.0f;
//     if (x > (score_t)( 4.0f)) return (score_t)1.0f;
//     return (score_t)0.5f + x * (score_t)0.125f;
// }

// ============================================================
// FUSED BATCH NORMALIZATION (inference-time)
// At inference BN fuses into conv: out = conv_out * scale + bias
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   bias[c]  = beta[c]  - mean[c] * scale[c]
// Both scale and bias are pre-computed in export_weights.py.
// ============================================================
inline act_t fused_bn(act_t x, weight_t scale, bias_t bias) {
    #pragma HLS INLINE
    return (act_t)((acc_t)x * (acc_t)scale + (acc_t)bias);
}

// ============================================================
// LAYER NORMALIZATION (for transformer tokens)
// input/output: [seq_len * dim] (row-major, each row is one token)
// weight/bias:  [dim]
// ============================================================
inline void layer_norm_seq(
    const act_t* input,
    act_t*       output,
    const weight_t* weight,
    const bias_t*   bias_ln,
    int seq_len, int dim
) {
    #pragma HLS INLINE
    for (int s = 0; s < seq_len; s++) {
        const act_t* row = input  + s * dim;
        act_t*       out = output + s * dim;

        // Mean
        acc_t sum = 0;
        LN_MEAN: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            sum += (acc_t)row[i];
        }
        act_t mean = (act_t)(sum / (acc_t)dim);

        // Variance
        acc_t var_sum = 0;
        LN_VAR: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            acc_t d = (acc_t)row[i] - (acc_t)mean;
            var_sum += d * d;
        }
        act_t inv_std = (act_t)(1.0f / sqrtf((float)(var_sum / (acc_t)dim) + 1e-5f));

        // Normalize + affine
        LN_NORM: for (int i = 0; i < dim; i++) {
            #pragma HLS PIPELINE II=1
            acc_t n = ((acc_t)row[i] - (acc_t)mean) * (acc_t)inv_std;
            out[i] = (act_t)(n * (acc_t)weight[i] + (acc_t)bias_ln[i]);
        }
    }
}

// ============================================================
// SOFTMAX over last dimension (for attention scores)
// input/output: [seq_len] for one query token
// ============================================================
inline void softmax_row(const act_t* input, act_t* output, int N) {
    #pragma HLS INLINE
    // Find max for numerical stability
    act_t max_v = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > max_v) max_v = input[i];
    }
    acc_t sum = 0;
    SM_EXP: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        float e = expf((float)(input[i] - max_v));
        output[i] = (act_t)e;
        sum += (acc_t)e;
    }
    SM_NORM: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = (act_t)((acc_t)output[i] / sum);
    }
}

// ============================================================
// CONVOLUTION PRIMITIVES
// All functions use flattened arrays [ch][H][W] = ch*H*W.
// Weight layout matches PyTorch: [out_ch][in_ch][kH][kW].
// For depthwise: [ch][kH][kW] = ch*9.
// BN fused into conv: weight pointer provides (bn_scale, bn_bias)
// right after the conv weights in the flat weight stream.
// ============================================================

// 3x3 conv + fused BN + SiLU activation, arbitrary stride
// weights:  [out_ch * in_ch * 9]
// bn_scale: [out_ch], bn_bias: [out_ch]
inline void conv3x3_bn_silu(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int in_ch, int out_ch, int H, int W, int stride
) {
    int out_H = DIV_CEIL(H, stride);
    int out_W = DIV_CEIL(W, stride);
    C3X3_OC: for (int oc = 0; oc < out_ch; oc++) {
        C3X3_OH: for (int oh = 0; oh < out_H; oh++) {
            C3X3_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                C3X3_IC: for (int ic = 0; ic < in_ch; ic++) {
                    C3X3_KH: for (int kh = 0; kh < 3; kh++) {
                        C3X3_KW: for (int kw = 0; kw < 3; kw++) {
                            int ih = oh * stride + kh - 1;
                            int iw = ow * stride + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += (acc_t)input[ic * H * W + ih * W + iw]
                                     * (acc_t)weights[(oc * in_ch + ic) * 9 + kh * 3 + kw];
                            }
                        }
                    }
                }
                act_t y = (act_t)((acc_t)acc * (acc_t)bn_scale[oc] + (acc_t)bn_bias[oc]);
                output[oc * out_H * out_W + oh * out_W + ow] = silu(y);
            }
        }
    }
}

// 3x3 conv + fused BN (no activation) – used for FPN output convs
inline void conv3x3_bn(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    C3X3B_OC: for (int oc = 0; oc < out_ch; oc++) {
        C3X3B_OH: for (int oh = 0; oh < H; oh++) {
            C3X3B_OW: for (int ow = 0; ow < W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                C3X3B_IC: for (int ic = 0; ic < in_ch; ic++) {
                    C3X3B_KH: for (int kh = 0; kh < 3; kh++) {
                        C3X3B_KW: for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += (acc_t)input[ic * H * W + ih * W + iw]
                                     * (acc_t)weights[(oc * in_ch + ic) * 9 + kh * 3 + kw];
                            }
                        }
                    }
                }
                output[oc * H * W + oh * W + ow] = (act_t)((acc_t)acc * (acc_t)bn_scale[oc] + (acc_t)bn_bias[oc]);
            }
        }
    }
}

// 1x1 conv + fused BN + SiLU
inline void conv1x1_bn_silu(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    C1X1S_OC: for (int oc = 0; oc < out_ch; oc++) {
        C1X1S_HW: for (int hw = 0; hw < H * W; hw++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = 0;
            C1X1S_IC: for (int ic = 0; ic < in_ch; ic++) {
                acc += (acc_t)input[ic * H * W + hw]
                     * (acc_t)weights[oc * in_ch + ic];
            }
            act_t y = (act_t)((acc_t)acc * (acc_t)bn_scale[oc] + (acc_t)bn_bias[oc]);
            output[oc * H * W + hw] = silu(y);
        }
    }
}

// 1x1 conv + fused BN (no activation) – used for FPN lateral convs, proj layers
inline void conv1x1_bn(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int in_ch, int out_ch, int H, int W
) {
    C1X1_OC: for (int oc = 0; oc < out_ch; oc++) {
        C1X1_HW: for (int hw = 0; hw < H * W; hw++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = 0;
            C1X1_IC: for (int ic = 0; ic < in_ch; ic++) {
                acc += (acc_t)input[ic * H * W + hw]
                     * (acc_t)weights[oc * in_ch + ic];
            }
            output[oc * H * W + hw] = (act_t)((acc_t)acc * (acc_t)bn_scale[oc] + (acc_t)bn_bias[oc]);
        }
    }
}

// 1x1 conv, no BN, no activation (RetinaHead prediction layers)
inline void conv1x1_plain(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const bias_t*   bias,
    int in_ch, int out_ch, int H, int W
) {
    C1P_OC: for (int oc = 0; oc < out_ch; oc++) {
        C1P_HW: for (int hw = 0; hw < H * W; hw++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)bias[oc];
            C1P_IC: for (int ic = 0; ic < in_ch; ic++) {
                acc += (acc_t)input[ic * H * W + hw]
                     * (acc_t)weights[oc * in_ch + ic];
            }
            output[oc * H * W + hw] = (act_t)acc;
        }
    }
}

// Depthwise 3x3 conv + fused BN + SiLU, arbitrary stride
// weights: [ch * 9]
inline void dw_conv3x3_bn_silu(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int ch, int H, int W, int stride
) {
    int out_H = DIV_CEIL(H, stride);
    int out_W = DIV_CEIL(W, stride);
    DW_C: for (int c = 0; c < ch; c++) {
        DW_OH: for (int oh = 0; oh < out_H; oh++) {
            DW_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DW_KH: for (int kh = 0; kh < 3; kh++) {
                    DW_KW: for (int kw = 0; kw < 3; kw++) {
                        int ih = oh * stride + kh - 1;
                        int iw = ow * stride + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            acc += (acc_t)input[c * H * W + ih * W + iw]
                                 * (acc_t)weights[c * 9 + kh * 3 + kw];
                        }
                    }
                }
                act_t y = (act_t)((acc_t)acc * (acc_t)bn_scale[c] + (acc_t)bn_bias[c]);
                output[c * out_H * out_W + oh * out_W + ow] = silu(y);
            }
        }
    }
}

// Depthwise 3x3 conv + fused BN (no activation) – project layer in MBConv
inline void dw_conv3x3_bn(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const weight_t* bn_scale,
    const bias_t*   bn_bias,
    int ch, int H, int W, int stride
) {
    int out_H = DIV_CEIL(H, stride);
    int out_W = DIV_CEIL(W, stride);
    DWB_C: for (int c = 0; c < ch; c++) {
        DWB_OH: for (int oh = 0; oh < out_H; oh++) {
            DWB_OW: for (int ow = 0; ow < out_W; ow++) {
                #pragma HLS PIPELINE II=1
                acc_t acc = 0;
                DWB_KH: for (int kh = 0; kh < 3; kh++) {
                    DWB_KW: for (int kw = 0; kw < 3; kw++) {
                        int ih = oh * stride + kh - 1;
                        int iw = ow * stride + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            acc += (acc_t)input[c * H * W + ih * W + iw]
                                 * (acc_t)weights[c * 9 + kh * 3 + kw];
                        }
                    }
                }
                output[c * out_H * out_W + oh * out_W + ow] =
                    (act_t)((acc_t)acc * (acc_t)bn_scale[c] + (acc_t)bn_bias[c]);
            }
        }
    }
}

// Linear layer: [batch * in_dim] → [batch * out_dim]
// weights: [out_dim * in_dim], bias: [out_dim]
inline void linear_layer(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const bias_t*   bias,
    int batch, int in_dim, int out_dim
) {
    LIN_B: for (int b = 0; b < batch; b++) {
        LIN_OD: for (int od = 0; od < out_dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)bias[od];
            LIN_ID: for (int id = 0; id < in_dim; id++) {
                acc += (acc_t)input[b * in_dim + id]
                     * (acc_t)weights[od * in_dim + id];
            }
            output[b * out_dim + od] = (act_t)acc;
        }
    }
}

// Element-wise add in-place: a[i] += b[i]
inline void elem_add_inplace(act_t* a, const act_t* b, int n) {
    #pragma HLS INLINE
    EAI: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        a[i] = a[i] + b[i];
    }
}

// 2× nearest-neighbor upsample: [ch][H][W] → [ch][2H][2W]
inline void upsample_2x(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    UP_C: for (int c = 0; c < ch; c++) {
        UP_H: for (int h = 0; h < H; h++) {
            UP_W: for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                act_t v = src[c * H * W + h * W + w];
                dst[c*(2*H)*(2*W) + (2*h  )*(2*W) + (2*w  )] = v;
                dst[c*(2*H)*(2*W) + (2*h  )*(2*W) + (2*w+1)] = v;
                dst[c*(2*H)*(2*W) + (2*h+1)*(2*W) + (2*w  )] = v;
                dst[c*(2*H)*(2*W) + (2*h+1)*(2*W) + (2*w+1)] = v;
            }
        }
    }
}

// Stride-2 subsampling: [ch][H][W] → [ch][H/2][W/2]
// MMDetection P6 = F.max_pool2d(p5, kernel_size=1, stride=2) = stride-2 pick
inline void maxpool2x2(const act_t* src, act_t* dst, int ch, int H, int W) {
    #pragma HLS INLINE
    int oH = H / 2, oW = W / 2;
    MP_C: for (int c = 0; c < ch; c++) {
        MP_H: for (int oh = 0; oh < oH; oh++) {
            MP_W: for (int ow = 0; ow < oW; ow++) {
                #pragma HLS PIPELINE II=1
                dst[c * oH * oW + oh * oW + ow] = src[c*H*W + (oh*2)*W + (ow*2)];
            }
        }
    }
}

// Copy buffer
inline void buf_copy(act_t* dst, const act_t* src, int n) {
    #pragma HLS INLINE
    BC: for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

#endif // FPGA_UTILS_H
