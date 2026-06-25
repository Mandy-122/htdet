/*
 * mvit_blk_s4_top.cpp
 * Full MobileViT Block Stage-4. Self-contained.
 * in_ch=160, d=240, H=W=10, depth=3, p=2, N=25 tokens per view
 */

#include "mvit_blk_s4_top.h"
#include "fpga_utils.h"
#include <cmath>

static void transformer_block_s4(act_t* tokens, const meta_t* w) {
    #pragma HLS INLINE
    static act_t s4t_norm_buf   [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_q_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_k_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_v_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_attn_scores[MVIT_N_S4 * MVIT_N_S4];
    static act_t s4t_attn_out   [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_mlp_hid    [MVIT_N_S4 * (2*MVIT_S4_DIM)];
    static act_t s4t_proj_out   [MVIT_N_S4 * MVIT_S4_DIM];
    #pragma HLS bind_storage variable=s4t_norm_buf    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_q_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_k_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_v_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_attn_scores type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_attn_out    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_mlp_hid     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4t_proj_out    type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=s4t_norm_buf    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_q_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_k_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_v_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_attn_scores cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_attn_out    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_mlp_hid     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_proj_out    cyclic factor=16 dim=1

    const int N = MVIT_N_S4, dim = MVIT_S4_DIM, head_dim = dim / MVIT_HEADS;
    int off = 0;
    const meta_t* ln1_w = w+off; off += dim;
    const meta_t* ln1_b = w+off; off += dim;
    layer_norm_seq(tokens, s4t_norm_buf, ln1_w, ln1_b, N, dim);
    const meta_t* Wq = w+off; const meta_t* bq = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wk = w+off; const meta_t* bk = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wv = w+off; const meta_t* bv = w+off+dim*dim; off += dim*dim+dim;
    linear_layer(s4t_norm_buf, s4t_q_buf, Wq, bq, N, dim, dim);
    linear_layer(s4t_norm_buf, s4t_k_buf, Wk, bk, N, dim, dim);
    linear_layer(s4t_norm_buf, s4t_v_buf, Wv, bv, N, dim, dim);
    float scale = 1.0f / sqrtf((float)head_dim);
    S4T_MHA: for (int h = 0; h < MVIT_HEADS; h++) {
        int hoff = h * head_dim;
        S4T_AQ: for (int q = 0; q < N; q++) {
            S4T_AK: for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0;
                S4T_AD: for (int d = 0; d < head_dim; d++) { #pragma HLS UNROLL factor=4
                    dot += (acc_t)s4t_q_buf[q*dim+hoff+d] * (acc_t)s4t_k_buf[k*dim+hoff+d]; }
                s4t_attn_scores[q*N+k] = (act_t)((float)dot * scale);
            }
        }
        S4T_SM: for (int q = 0; q < N; q++) softmax_row(s4t_attn_scores+q*N, s4t_attn_scores+q*N, N);
        S4T_VQ: for (int q = 0; q < N; q++) {
            S4T_VD: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                S4T_VK: for (int k = 0; k < N; k++) { #pragma HLS UNROLL factor=4
                    sum += (acc_t)s4t_attn_scores[q*N+k] * (acc_t)s4t_v_buf[k*dim+hoff+d]; }
                s4t_attn_out[q*dim+hoff+d] = (act_t)sum;
            }
        }
    }
    const meta_t* Wo = w+off; const meta_t* bo = w+off+dim*dim; off += dim*dim+dim;
    linear_layer(s4t_attn_out, s4t_proj_out, Wo, bo, N, dim, dim);
    S4T_R1: for (int i = 0; i < N*dim; i++) { #pragma HLS PIPELINE II=1 tokens[i] += s4t_proj_out[i]; }
    const meta_t* ln2_w = w+off; off += dim;
    const meta_t* ln2_b = w+off; off += dim;
    layer_norm_seq(tokens, s4t_norm_buf, ln2_w, ln2_b, N, dim);
    const int hidden = dim * 2;
    const meta_t* W1 = w+off; const meta_t* b1 = w+off+hidden*dim; off += hidden*dim+hidden;
    const meta_t* W2 = w+off; const meta_t* b2 = w+off+dim*hidden; off += dim*hidden+dim;
    S4T_M1B: for (int b = 0; b < N; b++) {
        S4T_M1O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            S4T_M1I: for (int id = 0; id < dim; id++) { #pragma HLS UNROLL factor=8
                acc += (acc_t)s4t_norm_buf[b*dim+id] * (acc_t)W1[od*dim+id]; }
            s4t_mlp_hid[b*hidden+od] = silu((act_t)acc);
        }
    }
    S4T_M2B: for (int b = 0; b < N; b++) {
        S4T_M2O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            S4T_M2I: for (int id = 0; id < hidden; id++) { #pragma HLS UNROLL factor=8
                acc += (acc_t)s4t_mlp_hid[b*hidden+id] * (acc_t)W2[od*hidden+id]; }
            s4t_proj_out[b*dim+od] = (act_t)acc;
        }
    }
    S4T_R2: for (int i = 0; i < N*dim; i++) { #pragma HLS PIPELINE II=1 tokens[i] += s4t_proj_out[i]; }
}

static void mobilevit_block_s4(
    const act_t* input, act_t* output,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff
) {
    #pragma HLS INLINE
    const int in_ch = STAGE4_PRE_CH, d = MVIT_S4_DIM, H = C4_H, W = C4_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);

    static act_t local_feat[STAGE4_PRE_CH * C4_H * C4_W];
    static act_t proj_feat [MVIT_S4_DIM * C4_H * C4_W];
    static act_t tokens    [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t fold_feat [MVIT_S4_DIM * C4_H * C4_W];
    static act_t proj_back [STAGE4_PRE_CH * C4_H * C4_W];
    static act_t concat_buf[2 * STAGE4_PRE_CH * C4_H * C4_W];
    #pragma HLS bind_storage variable=local_feat  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_feat   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=tokens      type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=fold_feat   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_back   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=concat_buf  type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    conv3x3_bn_silu(input, local_feat, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, in_ch, in_ch, H, W, 1);
    coff += in_ch*in_ch*9; moff += in_ch + in_ch;
    static const meta_t _zero_bias4[MVIT_S4_DIM] = {};
    conv1x1_bn(local_feat, proj_feat, w_conv+coff, w_meta+moff, _zero_bias4, in_ch, d, H, W);
    coff += d*in_ch; moff += d;
    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_moff = moff;
    moff += MVIT_S4_DEPTH * tb_sz;
    const meta_t* norm_w = w_meta + moff; moff += d;
    const meta_t* norm_b = w_meta + moff; moff += d;

    S4_VIEW: for (int view = 0; view < p2; view++) {
        int pi = view / p, pj = view % p;
        S4_UNF: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) { #pragma HLS UNROLL
                    tokens[n*d+c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)]; }
            }
        }
        for (int lyr = 0; lyr < MVIT_S4_DEPTH; lyr++)
            transformer_block_s4(tokens, w_meta+transformer_moff+lyr*tb_sz);
        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);
        S4_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) { #pragma HLS UNROLL
                    fold_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)] = tokens[n*d+c]; }
            }
        }
    }
    conv1x1_bn_silu(fold_feat, proj_back, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, d, in_ch, H, W);
    coff += in_ch*d; moff += in_ch + in_ch;
    S4_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W+hw]         = input[c*H*W+hw];
            concat_buf[(in_ch+c)*H*W+hw] = proj_back[c*H*W+hw];
        }
    }
    conv3x3_bn_silu(concat_buf, output, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, 2*in_ch, in_ch, H, W, 1);
    coff += in_ch*2*in_ch*9; moff += in_ch + in_ch;
}

void mvit_blk_s4_top(
    const act_t    in    [MVIT_S4_IN_ELEMS],
          act_t    out   [MVIT_S4_OUT_ELEMS],
    const weight_t w_conv[MVIT_S4_WCONV_ELEMS],
    const meta_t   w_meta[MVIT_S4_WMETA_ELEMS]
) {
    #pragma HLS INTERFACE bram port=in
    #pragma HLS INTERFACE bram port=out
    #pragma HLS INTERFACE bram port=w_conv
    #pragma HLS INTERFACE bram port=w_meta
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS ARRAY_PARTITION variable=in     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_conv cyclic factor=16 dim=1
    int coff = 0, moff = 0;
    mobilevit_block_s4(in, out, w_conv, coff, w_meta, moff);
}
