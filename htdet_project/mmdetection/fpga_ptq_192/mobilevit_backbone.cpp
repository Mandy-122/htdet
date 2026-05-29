/*
 * mobilevit_backbone.cpp  (PTQ INT8 — backbone proj 1x1 dequant scale fix applied)
 *
 * Changes vs mobilevit_backbone.cpp:
 *   - mobilevit_block_s2: conv1x1_plain → conv1x1_bn(w_meta+moff, zero_bias); moff += d
 *   - mobilevit_block_s3: same fix for S3 (d=192)
 *   - mobilevit_block_s4: same fix for S4 (d=240)
 *   - mobilevit_backbone → mobilevit_backbone
 *
 * backbone_meta layout insert (per MViT block, after local conv3x3 BN):
 *   proj_1x1_scale[d]  (144 / 192 / 240 floats for S2/S3/S4)
 *
 * Compile with mobilevit_backbone.h (NOT mobilevit_backbone.h).
 */

#include "mobilevit_backbone.h"

void mbconv_160(
    const act_t* in, act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t* w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = STEM_H, W = STEM_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[256 * STEM_H * STEM_W];
    static act_t dw_buf[256 * STEM_H * STEM_W];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=16 dim=1

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_80(
    const act_t* in, act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t* w_meta, int& moff,
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
    #pragma HLS bind_storage variable=ex_buf  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=dw_buf  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=res_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf  cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf  cyclic factor=16 dim=1

    if (use_res) { buf_copy(res_buf, in, in_ch * H * W); }

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, res_buf, in_ch * H * W); }
}

void mbconv_40(
    const act_t* in, act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t* w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C2_H, W = C2_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[512 * C2_H * C2_W];
    static act_t dw_buf[512 * C2_H * C2_W];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=16 dim=1

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_20(
    const act_t* in, act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t* w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C3_H, W = C3_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[640 * C3_H * C3_W];
    static act_t dw_buf[640 * C3_H * C3_W];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=16 dim=1

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

void mbconv_10(
    const act_t* in, act_t* out,
    const weight_t* w_conv, int& coff,
    const meta_t* w_meta, int& moff,
    int in_ch, int out_ch, int expand, int stride
) {
    #pragma HLS INLINE
    int H = C4_H, W = C4_W;
    int hid = in_ch * expand;
    int oH = DIV_CEIL(H, stride), oW = DIV_CEIL(W, stride);
    bool use_res = (stride == 1 && in_ch == out_ch);

    static act_t ex_buf[640 * C4_H * C4_W];
    static act_t dw_buf[640 * C4_H * C4_W];
    #pragma HLS bind_storage variable=ex_buf type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=dw_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=ex_buf cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=dw_buf cyclic factor=16 dim=1

    const act_t* dw_in = in;

    if (expand > 1) {
        conv1x1_bn_silu(in, ex_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, in_ch, hid, H, W);
        coff += hid * in_ch;
        moff += hid + hid;
        dw_in = ex_buf;
    }
    dw_conv3x3_bn_silu(dw_in, dw_buf, w_conv+coff, w_meta+moff, w_meta+moff+hid, hid, H, W, stride);
    coff += hid * 9;
    moff += hid + hid;
    conv1x1_bn(dw_buf, out, w_conv+coff, w_meta+moff, w_meta+moff+out_ch, hid, out_ch, oH, oW);
    coff += out_ch * hid;
    moff += out_ch + out_ch;

    if (use_res) { elem_add_inplace(out, in, in_ch * H * W); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformer blocks — identical to mobilevit_backbone.cpp (all-float meta)
// ─────────────────────────────────────────────────────────────────────────────

void transformer_block_s2(act_t* tokens, const meta_t* w) {
    #pragma HLS INLINE

    static act_t s2t_norm_buf [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_q_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_k_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_v_buf    [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_score_row[MVIT_N_S2];
    static act_t s2t_attn_out [MVIT_N_S2 * MVIT_S2_DIM];
    static act_t s2t_mlp_hid  [MVIT_N_S2 * (2*MVIT_S2_DIM)];
    static act_t s2t_proj_out [MVIT_N_S2 * MVIT_S2_DIM];
    #pragma HLS bind_storage variable=s2t_norm_buf  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_q_buf     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_k_buf     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_v_buf     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_score_row type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_attn_out  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_mlp_hid   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2t_proj_out  type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=s2t_norm_buf  cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_q_buf     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_k_buf     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_v_buf     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_score_row cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_attn_out  cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_mlp_hid   cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s2t_proj_out  cyclic factor=16 dim=1

    const int N        = MVIT_N_S2;
    const int dim      = MVIT_S2_DIM;
    const int head_dim = dim / MVIT_HEADS;
    int off = 0;

    const meta_t* ln1_w = w+off; off += dim;
    const meta_t* ln1_b = w+off; off += dim;
    layer_norm_seq(tokens, s2t_norm_buf, ln1_w, ln1_b, N, dim);

    const meta_t* Wq = w+off; const meta_t* bq = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wk = w+off; const meta_t* bk = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wv = w+off; const meta_t* bv = w+off+dim*dim; off += dim*dim+dim;
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

    const meta_t* Wo = w+off; const meta_t* bo = w+off+dim*dim; off += dim*dim+dim;
    linear_layer(s2t_attn_out, s2t_proj_out, Wo, bo, N, dim, dim);

    S2T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s2t_proj_out[i];
    }

    const meta_t* ln2_w = w+off; off += dim;
    const meta_t* ln2_b = w+off; off += dim;
    layer_norm_seq(tokens, s2t_norm_buf, ln2_w, ln2_b, N, dim);

    const int hidden = dim * 2;
    const meta_t* W1 = w+off; const meta_t* b1 = w+off+hidden*dim; off += hidden*dim+hidden;
    const meta_t* W2 = w+off; const meta_t* b2 = w+off+dim*hidden; off += dim*hidden+dim;

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

void transformer_block_s3(act_t* tokens, const meta_t* w) {
    #pragma HLS INLINE

    static act_t s3t_norm_buf   [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_q_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_k_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_v_buf      [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_attn_scores[MVIT_N_S3 * MVIT_N_S3];
    static act_t s3t_attn_out   [MVIT_N_S3 * MVIT_S3_DIM];
    static act_t s3t_mlp_hid    [MVIT_N_S3 * (2*MVIT_S3_DIM)];
    static act_t s3t_proj_out   [MVIT_N_S3 * MVIT_S3_DIM];
    #pragma HLS bind_storage variable=s3t_norm_buf    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_q_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_k_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_v_buf       type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_attn_scores type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_attn_out    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_mlp_hid     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3t_proj_out    type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=s3t_norm_buf    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_q_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_k_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_v_buf       cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_attn_scores cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_attn_out    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_mlp_hid     cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=s3t_proj_out    cyclic factor=16 dim=1

    const int N        = MVIT_N_S3;
    const int dim      = MVIT_S3_DIM;
    const int head_dim = dim / MVIT_HEADS;
    int off = 0;

    const meta_t* ln1_w = w+off; off += dim;
    const meta_t* ln1_b = w+off; off += dim;
    layer_norm_seq(tokens, s3t_norm_buf, ln1_w, ln1_b, N, dim);

    const meta_t* Wq = w+off; const meta_t* bq = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wk = w+off; const meta_t* bk = w+off+dim*dim; off += dim*dim+dim;
    const meta_t* Wv = w+off; const meta_t* bv = w+off+dim*dim; off += dim*dim+dim;
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

    const meta_t* Wo = w+off; const meta_t* bo = w+off+dim*dim; off += dim*dim+dim;
    linear_layer(s3t_attn_out, s3t_proj_out, Wo, bo, N, dim, dim);

    S3T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s3t_proj_out[i];
    }

    const meta_t* ln2_w = w+off; off += dim;
    const meta_t* ln2_b = w+off; off += dim;
    layer_norm_seq(tokens, s3t_norm_buf, ln2_w, ln2_b, N, dim);

    const int hidden = dim * 2;
    const meta_t* W1 = w+off; const meta_t* b1 = w+off+hidden*dim; off += hidden*dim+hidden;
    const meta_t* W2 = w+off; const meta_t* b2 = w+off+dim*hidden; off += dim*hidden+dim;

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

void transformer_block_s4(act_t* tokens, const meta_t* w) {
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

    const int N        = MVIT_N_S4;
    const int dim      = MVIT_S4_DIM;
    const int head_dim = dim / MVIT_HEADS;
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

    const meta_t* Wo = w+off; const meta_t* bo = w+off+dim*dim; off += dim*dim+dim;
    linear_layer(s4t_attn_out, s4t_proj_out, Wo, bo, N, dim, dim);

    S4T_R1: for (int i = 0; i < N*dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] += s4t_proj_out[i];
    }

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

// ─────────────────────────────────────────────────────────────────────────────
// MobileViT blocks — proj 1x1 NOW uses conv1x1_bn with dequant scale from meta
// ─────────────────────────────────────────────────────────────────────────────

void mobilevit_block_s2(
    const act_t* input,
    act_t*       output,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff
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
    #pragma HLS bind_storage variable=local_feat type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_feat  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=tokens     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=fold_feat  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_back  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=concat_buf type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    // local 3x3 conv + BN + SiLU
    conv3x3_bn_silu(input, local_feat, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, in_ch, in_ch, H, W, 1);
    coff += in_ch*in_ch*9;
    moff += in_ch + in_ch;

    // proj 1x1 + dequant scale from meta (no BN bias)
    static const meta_t _zero_bias2[MVIT_S2_DIM] = {};
    conv1x1_bn(local_feat, proj_feat, w_conv+coff, w_meta+moff, _zero_bias2, in_ch, d, H, W);
    coff += d*in_ch;
    moff += d;  // proj_1x1_scale[d]

    // transformer blocks (all float, from w_meta)
    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_moff = moff;
    moff += MVIT_S2_DEPTH * tb_sz;

    // final LN
    const meta_t* norm_w = w_meta + moff; moff += d;
    const meta_t* norm_b = w_meta + moff; moff += d;

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
            transformer_block_s2(tokens, w_meta+transformer_moff+lyr*tb_sz);
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

    // back proj (1x1 BN + SiLU)
    conv1x1_bn_silu(fold_feat, proj_back, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, d, in_ch, H, W);
    coff += in_ch*d;
    moff += in_ch + in_ch;

    // concat input + proj_back
    S2_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W + hw]         = input[c*H*W + hw];
            concat_buf[(in_ch+c)*H*W + hw] = proj_back[c*H*W + hw];
        }
    }

    // fusion conv (3x3 BN + SiLU)
    conv3x3_bn_silu(concat_buf, output, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, 2*in_ch, in_ch, H, W, 1);
    coff += in_ch*2*in_ch*9;
    moff += in_ch + in_ch;
}

void mobilevit_block_s3(
    const act_t* input,
    act_t*       output,
    const weight_t* w_conv, int& coff,
    const meta_t*   w_meta, int& moff
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
    #pragma HLS bind_storage variable=local_feat  type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_feat   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=tokens      type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=fold_feat   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=proj_back   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=concat_buf  type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=tokens    cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=proj_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=fold_feat cyclic factor=16 dim=1

    // local 3x3 conv + BN + SiLU
    conv3x3_bn_silu(input, local_feat, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, in_ch, in_ch, H, W, 1);
    coff += in_ch*in_ch*9;
    moff += in_ch + in_ch;

    // proj 1x1 + dequant scale from meta (no BN bias)
    static const meta_t _zero_bias3[MVIT_S3_DIM] = {};
    conv1x1_bn(local_feat, proj_feat, w_conv+coff, w_meta+moff, _zero_bias3, in_ch, d, H, W);
    coff += d*in_ch;
    moff += d;  // proj_1x1_scale[d]

    // transformer blocks (all float, from w_meta)
    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_moff = moff;
    moff += MVIT_S3_DEPTH * tb_sz;

    // final LN
    const meta_t* norm_w = w_meta + moff; moff += d;
    const meta_t* norm_b = w_meta + moff; moff += d;

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
            transformer_block_s3(tokens, w_meta+transformer_moff+lyr*tb_sz);
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

    // back proj (1x1 BN + SiLU)
    conv1x1_bn_silu(fold_feat, proj_back, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, d, in_ch, H, W);
    coff += in_ch*d;
    moff += in_ch + in_ch;

    // concat input + proj_back
    S3_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W+hw]         = input[c*H*W+hw];
            concat_buf[(in_ch+c)*H*W+hw] = proj_back[c*H*W+hw];
        }
    }

    // fusion conv (3x3 BN + SiLU)
    conv3x3_bn_silu(concat_buf, output, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, 2*in_ch, in_ch, H, W, 1);
    coff += in_ch*2*in_ch*9;
    moff += in_ch + in_ch;
}

void mobilevit_block_s4(
    const act_t* input,
    act_t*       output,
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

    // local 3x3 conv + BN + SiLU
    conv3x3_bn_silu(input, local_feat, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, in_ch, in_ch, H, W, 1);
    coff += in_ch*in_ch*9;
    moff += in_ch + in_ch;

    // proj 1x1 + dequant scale from meta (no BN bias)
    static const meta_t _zero_bias4[MVIT_S4_DIM] = {};
    conv1x1_bn(local_feat, proj_feat, w_conv+coff, w_meta+moff, _zero_bias4, in_ch, d, H, W);
    coff += d*in_ch;
    moff += d;  // proj_1x1_scale[d]

    // transformer blocks (all float, from w_meta)
    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_moff = moff;
    moff += MVIT_S4_DEPTH * tb_sz;

    // final LN
    const meta_t* norm_w = w_meta + moff; moff += d;
    const meta_t* norm_b = w_meta + moff; moff += d;

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
            transformer_block_s4(tokens, w_meta+transformer_moff+lyr*tb_sz);
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

    // back proj (1x1 BN + SiLU)
    conv1x1_bn_silu(fold_feat, proj_back, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, d, in_ch, H, W);
    coff += in_ch*d;
    moff += in_ch + in_ch;

    // concat input + proj_back
    S4_CAT: for (int c = 0; c < in_ch; c++) {
        for (int hw = 0; hw < H*W; hw++) {
            #pragma HLS PIPELINE II=1
            concat_buf[c*H*W+hw]         = input[c*H*W+hw];
            concat_buf[(in_ch+c)*H*W+hw] = proj_back[c*H*W+hw];
        }
    }

    // fusion conv (3x3 BN + SiLU)
    conv3x3_bn_silu(concat_buf, output, w_conv+coff, w_meta+moff, w_meta+moff+in_ch, 2*in_ch, in_ch, H, W, 1);
    coff += in_ch*2*in_ch*9;
    moff += in_ch + in_ch;
}

void mobilevit_backbone(
    const input_t*  image,
    const weight_t* backbone_conv,
    const meta_t*   backbone_meta,
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
    #pragma HLS bind_storage variable=stem_out type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s0_out   type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s1_a     type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s2_mb    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s3_mb    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4_mb    type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=s4_mvit  type=RAM_T2P impl=BRAM

    int coff = 0;
    int moff = 0;

    const act_t* img_act = reinterpret_cast<const act_t*>(image);

    // Stem: Conv3x3(3→16, s=2) + BN + SiLU
    conv3x3_bn_silu(img_act, stem_out,
                    backbone_conv+coff, backbone_meta+moff, backbone_meta+moff+STEM_CH,
                    INPUT_C, STEM_CH, INPUT_H, INPUT_W, 2);
    coff += STEM_CH * INPUT_C * 9;
    moff += STEM_CH + STEM_CH;

    // Stage 0: MBConv(16→32, expand=4, s=1)
    mbconv_160(stem_out, s0_out, backbone_conv, coff, backbone_meta, moff, STEM_CH, STAGE0_CH, MBCONV_EXPAND, 1);

    // Stage 1: MBConv(32→64, s=2) + 2×MBConv(64→64, s=1) → C1
    mbconv_160(s0_out, s1_a, backbone_conv, coff, backbone_meta, moff, STAGE0_CH, C1_CH, MBCONV_EXPAND, 2);
    mbconv_80(s1_a, c1, backbone_conv, coff, backbone_meta, moff, C1_CH, C1_CH, MBCONV_EXPAND, 1);
    mbconv_80(c1, c1, backbone_conv, coff, backbone_meta, moff, C1_CH, C1_CH, MBCONV_EXPAND, 1);

    // Stage 2: MBConv(64→96, s=2) + MobileViTBlock → C2
    mbconv_80(c1, s2_mb, backbone_conv, coff, backbone_meta, moff, C1_CH, C2_CH, MBCONV_EXPAND, 2);
    mobilevit_block_s2(s2_mb, c2, backbone_conv, coff, backbone_meta, moff);

    static act_t s2_out[C2_CH * C2_H * C2_W];
    #pragma HLS bind_storage variable=s2_out type=RAM_T2P impl=BRAM
    buf_copy(s2_out, c2, C2_CH * C2_H * C2_W);

    // Stage 3: MBConv(96→128, s=2) + MobileViTBlock → C3
    mbconv_40(s2_out, s3_mb, backbone_conv, coff, backbone_meta, moff, C2_CH, C3_CH, MBCONV_EXPAND, 2);
    mobilevit_block_s3(s3_mb, c3, backbone_conv, coff, backbone_meta, moff);

    static act_t s3_out[C3_CH * C3_H * C3_W];
    #pragma HLS bind_storage variable=s3_out type=RAM_T2P impl=BRAM
    buf_copy(s3_out, c3, C3_CH * C3_H * C3_W);

    // Stage 4: MBConv(128→160, s=2) + MobileViTBlock → expansion → C4
    mbconv_20(s3_out, s4_mb, backbone_conv, coff, backbone_meta, moff, C3_CH, STAGE4_PRE_CH, MBCONV_EXPAND, 2);
    mobilevit_block_s4(s4_mb, s4_mvit, backbone_conv, coff, backbone_meta, moff);

    // Final expansion Conv1x1(160→640) + BN + SiLU
    conv1x1_bn_silu(s4_mvit, c4, backbone_conv+coff, backbone_meta+moff, backbone_meta+moff+C4_CH,
                    STAGE4_PRE_CH, C4_CH, C4_H, C4_W);
    coff += C4_CH * STAGE4_PRE_CH;
    moff += C4_CH + C4_CH;

    (void)coff;
    (void)moff;
}
