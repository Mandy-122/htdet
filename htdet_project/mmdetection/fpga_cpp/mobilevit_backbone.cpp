/*
 * mobilevit_backbone.cpp
 * MobileViT-S backbone — function definitions (Vitis HLS).
 * Add this file to the Vitis HLS project alongside mobilevit_backbone.h.
 */

#include "mobilevit_backbone.h"

void mbconv_160(
    const act_t* in, act_t* out,
    const weight_t* w,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = STEM_H, W = STEM_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[256 * STEM_H * STEM_W];
    static act_t dw_buf[256 * STEM_H * STEM_W];
    #pragma HLS RESOURCE variable=ex_buf core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=8 dim=1

    int off = 0;
    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w+off, w+off+hid*in_ch, (const bias_t*)(w+off+hid*in_ch+hid), in_ch, hid, H, W);
        off += hid * in_ch + hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w+off, w+off+hid*9, (const bias_t*)(w+off+hid*9+hid), hid, H, W, stride);
    off += hid * 9 + hid + hid;
    conv1x1_bn(dw_buf, out, w+off, w+off+out_ch*hid, (const bias_t*)(w+off+out_ch*hid+out_ch), hid, out_ch, oH, oW);
    off += out_ch * hid + out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_80(
    const act_t* in, act_t* out,
    const weight_t* w,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C1_H, W = C1_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[256 * C1_H * C1_W];
    static act_t dw_buf[256 * C1_H * C1_W];
    static act_t res_buf[C1_CH * C1_H * C1_W];
    #pragma HLS RESOURCE variable=ex_buf  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=res_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf  cyclic factor=8 dim=1

    if (use_res) { buf_copy(res_buf, in, in_ch * H * W); }

    int off = 0;
    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w+off, w+off+hid*in_ch, (const bias_t*)(w+off+hid*in_ch+hid), in_ch, hid, H, W);
        off += hid * in_ch + hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w+off, w+off+hid*9, (const bias_t*)(w+off+hid*9+hid), hid, H, W, stride);
    off += hid * 9 + hid + hid;
    conv1x1_bn(dw_buf, out, w+off, w+off+out_ch*hid, (const bias_t*)(w+off+out_ch*hid+out_ch), hid, out_ch, oH, oW);
    off += out_ch * hid + out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, res_buf, in_ch * H * W); }
}

void mbconv_40(
    const act_t* in, act_t* out,
    const weight_t* w,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C2_H, W = C2_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[512 * C2_H * C2_W];
    static act_t dw_buf[512 * C2_H * C2_W];
    #pragma HLS RESOURCE variable=ex_buf core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=8 dim=1

    int off = 0;
    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w+off, w+off+hid*in_ch, (const bias_t*)(w+off+hid*in_ch+hid), in_ch, hid, H, W);
        off += hid * in_ch + hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w+off, w+off+hid*9, (const bias_t*)(w+off+hid*9+hid), hid, H, W, stride);
    off += hid * 9 + hid + hid;
    conv1x1_bn(dw_buf, out, w+off, w+off+out_ch*hid, (const bias_t*)(w+off+out_ch*hid+out_ch), hid, out_ch, oH, oW);
    off += out_ch * hid + out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_20(
    const act_t* in, act_t* out,
    const weight_t* w,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C3_H, W = C3_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[640 * C3_H * C3_W];
    static act_t dw_buf[640 * C3_H * C3_W];
    #pragma HLS RESOURCE variable=ex_buf core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=8 dim=1

    int off = 0;
    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w+off, w+off+hid*in_ch, (const bias_t*)(w+off+hid*in_ch+hid), in_ch, hid, H, W);
        off += hid * in_ch + hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w+off, w+off+hid*9, (const bias_t*)(w+off+hid*9+hid), hid, H, W, stride);
    off += hid * 9 + hid + hid;
    conv1x1_bn(dw_buf, out, w+off, w+off+out_ch*hid, (const bias_t*)(w+off+out_ch*hid+out_ch), hid, out_ch, oH, oW);
    off += out_ch * hid + out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_10(
    const act_t* in, act_t* out,
    const weight_t* w,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C4_H, W = C4_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[640 * C4_H * C4_W];
    static act_t dw_buf[640 * C4_H * C4_W];
    #pragma HLS RESOURCE variable=ex_buf core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=16 dim=1

    int off = 0;
    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w+off, w+off+hid*in_ch, (const bias_t*)(w+off+hid*in_ch+hid), in_ch, hid, H, W);
        off += hid * in_ch + hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w+off, w+off+hid*9, (const bias_t*)(w+off+hid*9+hid), hid, H, W, stride);
    off += hid * 9 + hid + hid;
    conv1x1_bn(dw_buf, out, w+off, w+off+out_ch*hid, (const bias_t*)(w+off+out_ch*hid+out_ch), hid, out_ch, oH, oW);
    off += out_ch * hid + out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void transformer_block(
    act_t* tokens,
    const weight_t* w,
    int N, int dim
) {
    #pragma HLS INLINE

    static act_t norm_buf[MVIT_N_MAX * MVIT_S4_DIM];
    static act_t q_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t k_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t v_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t attn_scores[MVIT_N_MAX * MVIT_N_MAX];
    static act_t attn_out[MVIT_N_MAX * MVIT_S4_DIM];
    static act_t mlp_hidden[MVIT_N_MAX * (2 * MVIT_S4_DIM)];
    #pragma HLS RESOURCE variable=norm_buf    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=q_buf       core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=k_buf       core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=v_buf       core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=attn_scores core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=attn_out    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=mlp_hidden  core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=q_buf       cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=k_buf       cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=v_buf       cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=attn_scores cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=attn_out    cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=norm_buf    cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=mlp_hidden  cyclic factor=8 dim=1

    int off = 0;

    const weight_t* ln1_w = w + off; off += dim;
    const bias_t*   ln1_b = (const bias_t*)(w + off); off += dim;
    layer_norm_seq(tokens, norm_buf, ln1_w, ln1_b, N, dim);

    const weight_t* Wq = w + off; const bias_t* bq = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    const weight_t* Wk = w + off; const bias_t* bk = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    const weight_t* Wv = w + off; const bias_t* bv = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    linear_layer(norm_buf, q_buf, Wq, bq, N, dim, dim);
    linear_layer(norm_buf, k_buf, Wk, bk, N, dim, dim);
    linear_layer(norm_buf, v_buf, Wv, bv, N, dim, dim);

    int head_dim = dim / MVIT_HEADS;
    float scale = 1.0f / sqrtf((float)head_dim);
    MHA_HEAD: for (int h = 0; h < MVIT_HEADS; h++) {
        #pragma HLS UNROLL
        int hoff = h * head_dim;
        ATTN_Q: for (int q = 0; q < N; q++) {
            ATTN_K: for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0;
                ATTN_D: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS UNROLL factor=4
                    dot += (acc_t)q_buf[q*dim + hoff + d] * (acc_t)k_buf[k*dim + hoff + d];
                }
                attn_scores[q*N+k] = (act_t)((float)dot * scale);
            }
        }
        SM_ROW: for (int q = 0; q < N; q++) {
            softmax_row(attn_scores + q*N, attn_scores + q*N, N);
        }
        AV_Q: for (int q = 0; q < N; q++) {
            AV_D: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                AV_K: for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL factor=4
                    sum += (acc_t)attn_scores[q*N+k] * (acc_t)v_buf[k*dim + hoff + d];
                }
                attn_out[q*dim + hoff + d] = (act_t)sum;
            }
        }
    }

    const weight_t* Wo = w + off; const bias_t* bo = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    static act_t proj_out[MVIT_N_MAX * MVIT_S4_DIM];
    #pragma HLS RESOURCE variable=proj_out core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=proj_out cyclic factor=8 dim=1
    linear_layer(attn_out, proj_out, Wo, bo, N, dim, dim);

    TRES1: for (int i = 0; i < N * dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] = tokens[i] + proj_out[i];
    }

    const weight_t* ln2_w = w + off; off += dim;
    const bias_t*   ln2_b = (const bias_t*)(w + off); off += dim;
    layer_norm_seq(tokens, norm_buf, ln2_w, ln2_b, N, dim);

    int hidden = dim * 2;
    const weight_t* W1 = w + off; const bias_t* b1 = (const bias_t*)(w+off+hidden*dim); off += hidden*dim + hidden;
    const weight_t* W2 = w + off; const bias_t* b2 = (const bias_t*)(w+off+dim*hidden); off += dim*hidden + dim;

    MLP1_B: for (int b = 0; b < N; b++) {
        MLP1_O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            MLP1_I: for (int id = 0; id < dim; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)norm_buf[b*dim+id] * (acc_t)W1[od*dim+id];
            }
            mlp_hidden[b*hidden+od] = silu((act_t)acc);
        }
    }
    MLP2_B: for (int b = 0; b < N; b++) {
        MLP2_O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            MLP2_I: for (int id = 0; id < hidden; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)mlp_hidden[b*hidden+id] * (acc_t)W2[od*hidden+id];
            }
            proj_out[b*dim+od] = (act_t)acc;
        }
    }

    TRES2: for (int i = 0; i < N * dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] = tokens[i] + proj_out[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage-specific transformer blocks — right-sized BRAM, no dynamic dispatch.
// S2: streaming attention (score_row[N] not N×N matrix) → BRAM 10 MB → 6.4 KB.
// S3/S4: standard attention with correctly-sized attn_scores buffers.
// Sequential MHA heads (no UNROLL) — avoids write conflicts in synthesis.
// ─────────────────────────────────────────────────────────────────────────────

void transformer_block_s2(act_t* tokens, const weight_t* w) {
    #pragma HLS INLINE

    static act_t s2t_norm_buf [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_q_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_k_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_v_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_score_row[MVIT_N_S2];
    static act_t s2t_attn_out [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_mlp_hid  [MVIT_N_S2 * (2*MVIT_S2_DIM)];
    static act_t s2t_proj_out [MVIT_N_S2 * MVIT_S2_DIM];
    #pragma HLS RESOURCE variable=s2t_norm_buf  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_q_buf     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_k_buf     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_v_buf     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_score_row core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_attn_out  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_mlp_hid   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2t_proj_out  core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=s2t_norm_buf  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_q_buf     cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_k_buf     cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_v_buf     cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_score_row cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_attn_out  cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_mlp_hid   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_proj_out  cyclic factor=8 dim=1

    const int N        = MVIT_N_S2;
    const int dim      = MVIT_S2_DIM;
    const int head_dim = dim / MVIT_HEADS;
    int off = 0;

    const weight_t* ln1_w = w+off; off += dim;
    const bias_t*   ln1_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s2t_norm_buf, ln1_w, ln1_b, N, dim);

    const weight_t* Wq = w+off; const bias_t* bq = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wk = w+off; const bias_t* bk = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wv = w+off; const bias_t* bv = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    linear_layer(s2t_norm_buf, s2t_q_buf, Wq, bq, N, dim, dim);
    linear_layer(s2t_norm_buf, s2t_k_buf, Wk, bk, N, dim, dim);
    linear_layer(s2t_norm_buf, s2t_v_buf, Wv, bv, N, dim, dim);

    float scale = 1.0f / sqrtf((float)head_dim);
    S2T_MHA: for (int h = 0; h < MVIT_HEADS; h++) {
        int hoff = h * head_dim;
        S2T_AQ: for (int q = 0; q < N; q++) {
            S2T_AK: for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0;
                S2T_AD: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS UNROLL factor=4
                    dot += (acc_t)s2t_q_buf[q*dim+hoff+d] * (acc_t)s2t_k_buf[k*dim+hoff+d];
                }
                s2t_score_row[k] = (act_t)((float)dot * scale);
            }
            softmax_row(s2t_score_row, s2t_score_row, N);
            S2T_VD: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                S2T_VK: for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL factor=4
                    sum += (acc_t)s2t_score_row[k] * (acc_t)s2t_v_buf[k*dim+hoff+d];
                }
                s2t_attn_out[q*dim+hoff+d] = (act_t)sum;
            }
        }
    }

    const weight_t* Wo = w+off; const bias_t* bo = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    linear_layer(s2t_attn_out, s2t_proj_out, Wo, bo, N, dim, dim);

    S2T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s2t_proj_out[i];
    }

    const weight_t* ln2_w = w+off; off += dim;
    const bias_t*   ln2_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s2t_norm_buf, ln2_w, ln2_b, N, dim);

    const int hidden = dim * 2;
    const weight_t* W1 = w+off; const bias_t* b1 = (const bias_t*)(w+off+hidden*dim); off += hidden*dim+hidden;
    const weight_t* W2 = w+off; const bias_t* b2 = (const bias_t*)(w+off+dim*hidden); off += dim*hidden+dim;

    S2T_M1B: for (int b = 0; b < N; b++) {
        S2T_M1O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            S2T_M1I: for (int id = 0; id < dim; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s2t_norm_buf[b*dim+id] * (acc_t)W1[od*dim+id];
            }
            s2t_mlp_hid[b*hidden+od] = silu((act_t)acc);
        }
    }
    S2T_M2B: for (int b = 0; b < N; b++) {
        S2T_M2O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            S2T_M2I: for (int id = 0; id < hidden; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s2t_mlp_hid[b*hidden+id] * (acc_t)W2[od*hidden+id];
            }
            s2t_proj_out[b*dim+od] = (act_t)acc;
        }
    }

    S2T_R2: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s2t_proj_out[i];
    }
}

void transformer_block_s3(act_t* tokens, const weight_t* w) {
    #pragma HLS INLINE

    static act_t s3t_norm_buf   [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_q_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_k_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_v_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_attn_scores[MVIT_N_S3 * MVIT_N_S3];
    static act_t s3t_attn_out   [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_mlp_hid    [MVIT_N_S3 * (2*MVIT_S3_DIM)];
    static act_t s3t_proj_out   [MVIT_N_S3 * MVIT_S3_DIM];
    #pragma HLS RESOURCE variable=s3t_norm_buf   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_q_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_k_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_v_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_attn_scores core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_attn_out   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_mlp_hid    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3t_proj_out   core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=s3t_norm_buf   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_q_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_k_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_v_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_attn_scores cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_attn_out   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_mlp_hid    cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_proj_out   cyclic factor=8 dim=1

    const int N        = MVIT_N_S3;
    const int dim      = MVIT_S3_DIM;
    const int head_dim = dim / MVIT_HEADS;
    int off = 0;

    const weight_t* ln1_w = w+off; off += dim;
    const bias_t*   ln1_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s3t_norm_buf, ln1_w, ln1_b, N, dim);

    const weight_t* Wq = w+off; const bias_t* bq = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wk = w+off; const bias_t* bk = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wv = w+off; const bias_t* bv = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    linear_layer(s3t_norm_buf, s3t_q_buf, Wq, bq, N, dim, dim);
    linear_layer(s3t_norm_buf, s3t_k_buf, Wk, bk, N, dim, dim);
    linear_layer(s3t_norm_buf, s3t_v_buf, Wv, bv, N, dim, dim);

    float scale = 1.0f / sqrtf((float)head_dim);
    S3T_MHA: for (int h = 0; h < MVIT_HEADS; h++) {
        int hoff = h * head_dim;
        S3T_AQ: for (int q = 0; q < N; q++) {
            S3T_AK: for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0;
                S3T_AD: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS UNROLL factor=4
                    dot += (acc_t)s3t_q_buf[q*dim+hoff+d] * (acc_t)s3t_k_buf[k*dim+hoff+d];
                }
                s3t_attn_scores[q*N+k] = (act_t)((float)dot * scale);
            }
        }
        S3T_SM: for (int q = 0; q < N; q++) {
            softmax_row(s3t_attn_scores+q*N, s3t_attn_scores+q*N, N);
        }
        S3T_VQ: for (int q = 0; q < N; q++) {
            S3T_VD: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                S3T_VK: for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL factor=4
                    sum += (acc_t)s3t_attn_scores[q*N+k] * (acc_t)s3t_v_buf[k*dim+hoff+d];
                }
                s3t_attn_out[q*dim+hoff+d] = (act_t)sum;
            }
        }
    }

    const weight_t* Wo = w+off; const bias_t* bo = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    linear_layer(s3t_attn_out, s3t_proj_out, Wo, bo, N, dim, dim);

    S3T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s3t_proj_out[i];
    }

    const weight_t* ln2_w = w+off; off += dim;
    const bias_t*   ln2_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s3t_norm_buf, ln2_w, ln2_b, N, dim);

    const int hidden = dim * 2;
    const weight_t* W1 = w+off; const bias_t* b1 = (const bias_t*)(w+off+hidden*dim); off += hidden*dim+hidden;
    const weight_t* W2 = w+off; const bias_t* b2 = (const bias_t*)(w+off+dim*hidden); off += dim*hidden+dim;

    S3T_M1B: for (int b = 0; b < N; b++) {
        S3T_M1O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            S3T_M1I: for (int id = 0; id < dim; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s3t_norm_buf[b*dim+id] * (acc_t)W1[od*dim+id];
            }
            s3t_mlp_hid[b*hidden+od] = silu((act_t)acc);
        }
    }
    S3T_M2B: for (int b = 0; b < N; b++) {
        S3T_M2O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            S3T_M2I: for (int id = 0; id < hidden; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s3t_mlp_hid[b*hidden+id] * (acc_t)W2[od*hidden+id];
            }
            s3t_proj_out[b*dim+od] = (act_t)acc;
        }
    }

    S3T_R2: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s3t_proj_out[i];
    }
}

void transformer_block_s4(act_t* tokens, const weight_t* w) {
    #pragma HLS INLINE

    static act_t s4t_norm_buf   [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_q_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_k_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_v_buf      [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_attn_scores[MVIT_N_S4 * MVIT_N_S4];
    static act_t s4t_attn_out   [MVIT_N_S4 * MVIT_S4_DIM];
    static act_t s4t_mlp_hid    [MVIT_N_S4 * (2*MVIT_S4_DIM)];
    static act_t s4t_proj_out   [MVIT_N_S4 * MVIT_S4_DIM];
    #pragma HLS RESOURCE variable=s4t_norm_buf   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_q_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_k_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_v_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_attn_scores core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_attn_out   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_mlp_hid    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4t_proj_out   core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=s4t_norm_buf   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_q_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_k_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_v_buf      cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_attn_scores cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_attn_out   cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_mlp_hid    cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=s4t_proj_out   cyclic factor=8 dim=1

    const int N        = MVIT_N_S4;
    const int dim      = MVIT_S4_DIM;
    const int head_dim = dim / MVIT_HEADS;
    int off = 0;

    const weight_t* ln1_w = w+off; off += dim;
    const bias_t*   ln1_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s4t_norm_buf, ln1_w, ln1_b, N, dim);

    const weight_t* Wq = w+off; const bias_t* bq = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wk = w+off; const bias_t* bk = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    const weight_t* Wv = w+off; const bias_t* bv = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
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
                S4T_AD: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS UNROLL factor=4
                    dot += (acc_t)s4t_q_buf[q*dim+hoff+d] * (acc_t)s4t_k_buf[k*dim+hoff+d];
                }
                s4t_attn_scores[q*N+k] = (act_t)((float)dot * scale);
            }
        }
        S4T_SM: for (int q = 0; q < N; q++) {
            softmax_row(s4t_attn_scores+q*N, s4t_attn_scores+q*N, N);
        }
        S4T_VQ: for (int q = 0; q < N; q++) {
            S4T_VD: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                S4T_VK: for (int k = 0; k < N; k++) {
                    #pragma HLS UNROLL factor=4
                    sum += (acc_t)s4t_attn_scores[q*N+k] * (acc_t)s4t_v_buf[k*dim+hoff+d];
                }
                s4t_attn_out[q*dim+hoff+d] = (act_t)sum;
            }
        }
    }

    const weight_t* Wo = w+off; const bias_t* bo = (const bias_t*)(w+off+dim*dim); off += dim*dim+dim;
    linear_layer(s4t_attn_out, s4t_proj_out, Wo, bo, N, dim, dim);

    S4T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s4t_proj_out[i];
    }

    const weight_t* ln2_w = w+off; off += dim;
    const bias_t*   ln2_b = (const bias_t*)(w+off); off += dim;
    layer_norm_seq(tokens, s4t_norm_buf, ln2_w, ln2_b, N, dim);

    const int hidden = dim * 2;
    const weight_t* W1 = w+off; const bias_t* b1 = (const bias_t*)(w+off+hidden*dim); off += hidden*dim+hidden;
    const weight_t* W2 = w+off; const bias_t* b2 = (const bias_t*)(w+off+dim*hidden); off += dim*hidden+dim;

    S4T_M1B: for (int b = 0; b < N; b++) {
        S4T_M1O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            S4T_M1I: for (int id = 0; id < dim; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s4t_norm_buf[b*dim+id] * (acc_t)W1[od*dim+id];
            }
            s4t_mlp_hid[b*hidden+od] = silu((act_t)acc);
        }
    }
    S4T_M2B: for (int b = 0; b < N; b++) {
        S4T_M2O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            S4T_M2I: for (int id = 0; id < hidden; id++) {
                #pragma HLS UNROLL factor=8
                acc += (acc_t)s4t_mlp_hid[b*hidden+id] * (acc_t)W2[od*hidden+id];
            }
            s4t_proj_out[b*dim+od] = (act_t)acc;
        }
    }

    S4T_R2: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s4t_proj_out[i];
    }
}

void mobilevit_block_s2(
    const act_t* input,
    act_t*       output,
    const weight_t* w
) {
    #pragma HLS INLINE
    const int in_ch = C2_CH, d = MVIT_S2_DIM, H = C2_H, W = C2_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);

    static act_t local_feat[C2_CH * C2_H * C2_W];
    static act_t proj_feat [MVIT_S2_DIM * C2_H * C2_W];
    static act_t tokens    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t fold_feat [MVIT_S2_DIM * C2_H * C2_W];
    static act_t proj_back [C2_CH * C2_H * C2_W];
    static act_t concat_buf[2 * C2_CH * C2_H * C2_W];
    #pragma HLS RESOURCE variable=local_feat core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    int off = 0;

    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    static const bias_t _zero_proj2[MVIT_S2_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj2, in_ch, d, H, W);
    off += d*in_ch;

    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_off = off;
    off += MVIT_S2_DEPTH * tb_sz;
    const weight_t* norm_w = w + off; off += d;
    const bias_t*   norm_b = (const bias_t*)(w + off); off += d;

    S2_VIEW: for (int view = 0; view < p2; view++) {
        int pi = view / p, pj = view % p;

        S2_UNF: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        for (int lyr = 0; lyr < MVIT_S2_DEPTH; lyr++) {
            transformer_block_s2(tokens, w+transformer_off+lyr*tb_sz);
        }

        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        S2_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    fold_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)] = tokens[n*d + c];
                }
            }
        }
    }

    conv1x1_bn_silu(fold_feat, proj_back, w+off, w+off+in_ch*d,
                    (const bias_t*)(w+off+in_ch*d+in_ch), d, in_ch, H, W);
    off += in_ch*d + in_ch + in_ch;

    S2_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W + hw]         = input[c*H*W + hw];
            concat_buf[(in_ch+c)*H*W + hw] = proj_back[c*H*W + hw];
        }
    }
    conv3x3_bn_silu(concat_buf, output, w+off, w+off+(in_ch*2*in_ch*9),
                    (const bias_t*)(w+off+(in_ch*2*in_ch*9)+in_ch), 2*in_ch, in_ch, H, W, 1);
    off += in_ch*2*in_ch*9 + in_ch + in_ch;
}

void mobilevit_block_s3(
    const act_t* input,
    act_t*       output,
    const weight_t* w
) {
    #pragma HLS INLINE
    const int in_ch = C3_CH, d = MVIT_S3_DIM, H = C3_H, W = C3_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);

    static act_t local_feat[C3_CH * C3_H * C3_W];
    static act_t proj_feat [MVIT_S3_DIM * C3_H * C3_W];
    static act_t tokens    [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t fold_feat [MVIT_S3_DIM * C3_H * C3_W];
    static act_t proj_back [C3_CH * C3_H * C3_W];
    static act_t concat_buf[2 * C3_CH * C3_H * C3_W];
    #pragma HLS RESOURCE variable=local_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf  core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    int off = 0;

    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    static const bias_t _zero_proj3[MVIT_S3_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj3, in_ch, d, H, W);
    off += d*in_ch;

    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_off = off;
    off += MVIT_S3_DEPTH * tb_sz;
    const weight_t* norm_w = w + off; off += d;
    const bias_t*   norm_b = (const bias_t*)(w + off); off += d;

    S3_VIEW: for (int view = 0; view < p2; view++) {
        int pi = view / p, pj = view % p;

        S3_UNF: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        for (int lyr = 0; lyr < MVIT_S3_DEPTH; lyr++) {
            transformer_block_s3(tokens, w+transformer_off+lyr*tb_sz);
        }

        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        S3_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    fold_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)] = tokens[n*d + c];
                }
            }
        }
    }

    conv1x1_bn_silu(fold_feat, proj_back, w+off, w+off+in_ch*d,
                    (const bias_t*)(w+off+in_ch*d+in_ch), d, in_ch, H, W);
    off += in_ch*d + in_ch + in_ch;

    S3_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W+hw]         = input[c*H*W+hw];
            concat_buf[(in_ch+c)*H*W+hw] = proj_back[c*H*W+hw];
        }
    }
    conv3x3_bn_silu(concat_buf, output, w+off, w+off+(in_ch*2*in_ch*9),
                    (const bias_t*)(w+off+(in_ch*2*in_ch*9)+in_ch), 2*in_ch, in_ch, H, W, 1);
    off += in_ch*2*in_ch*9 + in_ch + in_ch;
}

void mobilevit_block_s4(
    const act_t* input,
    act_t*       output,
    const weight_t* w
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
    #pragma HLS RESOURCE variable=local_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf  core=RAM_T2P_BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    int off = 0;

    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    static const bias_t _zero_proj4[MVIT_S4_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj4, in_ch, d, H, W);
    off += d*in_ch;

    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_off = off;
    off += MVIT_S4_DEPTH * tb_sz;
    const weight_t* norm_w = w + off; off += d;
    const bias_t*   norm_b = (const bias_t*)(w + off); off += d;

    S4_VIEW: for (int view = 0; view < p2; view++) {
        int pi = view / p, pj = view % p;

        S4_UNF: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        for (int lyr = 0; lyr < MVIT_S4_DEPTH; lyr++) {
            transformer_block_s4(tokens, w+transformer_off+lyr*tb_sz);
        }

        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        S4_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    #pragma HLS UNROLL
                    fold_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)] = tokens[n*d + c];
                }
            }
        }
    }

    conv1x1_bn_silu(fold_feat, proj_back, w+off, w+off+in_ch*d,
                    (const bias_t*)(w+off+in_ch*d+in_ch), d, in_ch, H, W);
    off += in_ch*d + in_ch + in_ch;

    S4_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W+hw]         = input[c*H*W+hw];
            concat_buf[(in_ch+c)*H*W+hw] = proj_back[c*H*W+hw];
        }
    }
    conv3x3_bn_silu(concat_buf, output, w+off, w+off+(in_ch*2*in_ch*9),
                    (const bias_t*)(w+off+(in_ch*2*in_ch*9)+in_ch), 2*in_ch, in_ch, H, W, 1);
    off += in_ch*2*in_ch*9 + in_ch + in_ch;
}

void mobilevit_backbone(
    const input_t*  image,
    const weight_t* weights,
    act_t* c1,
    act_t* c2,
    act_t* c3,
    act_t* c4
) {
    static act_t stem_out[STEM_CH   * STEM_H * STEM_W];
    static act_t s0_out  [STAGE0_CH * STEM_H * STEM_W];
    static act_t s1_a    [C1_CH     * C1_H   * C1_W  ];
    static act_t s2_mb   [C2_CH     * C2_H   * C2_W  ];
    static act_t s3_mb   [C3_CH     * C3_H   * C3_W  ];
    static act_t s4_mb   [STAGE4_PRE_CH * C4_H * C4_W];
    static act_t s4_mvit [STAGE4_PRE_CH * C4_H * C4_W];
    #pragma HLS RESOURCE variable=stem_out core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s0_out   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s1_a     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4_mvit  core=RAM_T2P_BRAM

    CAST_IN: for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++) {
        #pragma HLS PIPELINE II=1
        (void)i;
    }

    int off = 0;

    const act_t* img_act = reinterpret_cast<const act_t*>(image);
    conv3x3_bn_silu(img_act, stem_out,
                    weights+off, weights+off+STEM_CH*INPUT_C*9,
                    (const bias_t*)(weights+off+STEM_CH*INPUT_C*9+STEM_CH),
                    INPUT_C, STEM_CH, INPUT_H, INPUT_W, 2);
    off += STEM_CH*INPUT_C*9 + STEM_CH + STEM_CH;

    mbconv_160(stem_out, s0_out, weights+off, STEM_CH, STAGE0_CH, MBCONV_EXPAND, 1);
    {
        int hid0 = STEM_CH*MBCONV_EXPAND;
        off += hid0*STEM_CH+hid0+hid0 + hid0*9+hid0+hid0 + STAGE0_CH*hid0+STAGE0_CH+STAGE0_CH;
    }

    mbconv_160(s0_out, s1_a, weights+off, STAGE0_CH, C1_CH, MBCONV_EXPAND, 2);
    {
        int hid1 = STAGE0_CH*MBCONV_EXPAND;
        off += hid1*STAGE0_CH+hid1+hid1 + hid1*9+hid1+hid1 + C1_CH*hid1+C1_CH+C1_CH;
    }
    mbconv_80(s1_a, c1, weights+off, C1_CH, C1_CH, MBCONV_EXPAND, 1);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C1_CH*hid+C1_CH+C1_CH;
    }
    mbconv_80(c1, c1, weights+off, C1_CH, C1_CH, MBCONV_EXPAND, 1);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C1_CH*hid+C1_CH+C1_CH;
    }

    mbconv_80(c1, s2_mb, weights+off, C1_CH, C2_CH, MBCONV_EXPAND, 2);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C2_CH*hid+C2_CH+C2_CH;
    }
    mobilevit_block_s2(s2_mb, c2, weights+off);
    off += 612288;

    static act_t s2_out[C2_CH * C2_H * C2_W];
    buf_copy(s2_out, c2, C2_CH * C2_H * C2_W);
    mbconv_40(s2_out, s3_mb, weights+off, C2_CH, C3_CH, MBCONV_EXPAND, 2);
    {
        int hid = C2_CH*MBCONV_EXPAND;
        off += hid*C2_CH+hid+hid + hid*9+hid+hid + C3_CH*hid+C3_CH+C3_CH;
    }
    mobilevit_block_s3(s3_mb, c3, weights+off);
    off += 1680768;

    static act_t s3_out[C3_CH * C3_H * C3_W];
    buf_copy(s3_out, c3, C3_CH * C3_H * C3_W);
    mbconv_20(s3_out, s4_mb, weights+off, C3_CH, STAGE4_PRE_CH, MBCONV_EXPAND, 2);
    {
        int hid = C3_CH*MBCONV_EXPAND;
        off += hid*C3_CH+hid+hid + hid*9+hid+hid + STAGE4_PRE_CH*hid+STAGE4_PRE_CH+STAGE4_PRE_CH;
    }
    mobilevit_block_s4(s4_mb, s4_mvit, weights+off);
    off += 2159760;

    const weight_t* exp_conv_w = weights + off;
    const weight_t* exp_bn_s   = exp_conv_w + C4_CH * STAGE4_PRE_CH;
    const bias_t*   exp_bn_b   = (const bias_t*)(exp_bn_s + C4_CH);
    conv1x1_bn_silu(s4_mvit, c4, exp_conv_w, exp_bn_s, exp_bn_b, STAGE4_PRE_CH, C4_CH, C4_H, C4_W);
}
