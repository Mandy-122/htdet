/*
 * mobilevit_backbone.h
 * MobileViT-S backbone for HTDet FPGA (Vitis HLS)
 *
 * Architecture (TIMM mobilevit_s, out_indices=(1,2,3,4)):
 *
 *   Input  [3, 640, 640]  (INPUT_H=640, INPUT_W=640 — change in fpga_types.h)
 *   Stem   Conv3x3(3→16, s=2) + BN + SiLU            → [16, 320, 320]
 *   Stg0   MBConv(16→32, expand=4, s=1)               → [32, 320, 320]  (not exported)
 *   Stg1   MBConv(32→64, expand=4, s=2)               → [64, 160, 160]
 *          MBConv(64→64, expand=4, s=1) × 2            → C1 = [64, 160, 160]  ← FPN input 0
 *   Stg2   MBConv(64→96, expand=4, s=2)               → [96,  80,  80]
 *          MobileViTBlock(96, d=144, depth=2)           → C2 = [96,  80,  80]  ← FPN input 1
 *   Stg3   MBConv(96→128, expand=4, s=2)              → [128, 40,  40]
 *          MobileViTBlock(128, d=192, depth=4)          → C3 = [128, 40,  40]  ← FPN input 2
 *   Stg4   MBConv(128→160, expand=4, s=2)             → [160, 20,  20]
 *          MobileViTBlock(160, d=240, depth=3)          → [160, 20,  20]
 *          Conv1x1(160→640) + BN + SiLU               → C4 = [640, 20,  20]  ← FPN input 3
 *
 * Weight layout (flat float32 / weight_t array, passed from DDR):
 *   See WEIGHT_OFFSETS section at the bottom of this file.
 *   All BN parameters are pre-fused into conv weights at export time:
 *     - bn_scale[oc] = gamma[oc] / sqrt(var[oc] + eps)
 *     - bn_bias[oc]  = beta[oc]  - mean[oc] * bn_scale[oc]
 *
 * MobileViT patch/transformer convention:
 *   patch_size = 2×2.  For an H×W feature map, we extract (H/2)×(W/2) = N patches.
 *   Each patch is proj_dim = d × patch_size² values → linearly projected to d.
 *   TransformerBlock: LayerNorm → MHSA (4 heads) → residual → LayerNorm → MLP → residual.
 */

#ifndef MOBILEVIT_BACKBONE_H
#define MOBILEVIT_BACKBONE_H

#include "fpga_types.h"
#include "fpga_utils.h"

// ============================================================
// MBCONV BLOCK (MobileNetV2 Inverted Residual)
// Expand(1×1) → DW-Conv(3×3) → Project(1×1)
// When stride==1 and in_ch==out_ch: add residual.
//
// Weight layout consumed (offset returned):
//   [if expand>1] expand_w[hid*in*1]:  hid*in weights
//                 expand_bn_s[hid], expand_bn_b[hid]
//   dw_w[hid*9]:  hid*9 weights
//   dw_bn_s[hid], dw_bn_b[hid]
//   proj_w[out*hid]: out*hid weights
//   proj_bn_s[out], proj_bn_b[out]
// ============================================================

// MBConv for stem-scale layers (160×160)
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

    static act_t ex_buf[256 * STEM_H * STEM_W];   // expand buffer, max hid=256
    static act_t dw_buf[256 * STEM_H * STEM_W];   // act_t=float now, no saturation
    #pragma HLS RESOURCE variable=ex_buf core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf core=RAM_T2P_BRAM

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

// MBConv for C1-scale layers (80×80)
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
    static act_t dw_buf[256 * C1_H * C1_W];   // act_t=float now, no saturation
    static act_t res_buf[C1_CH * C1_H * C1_W];
    #pragma HLS RESOURCE variable=ex_buf  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=dw_buf  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=res_buf core=RAM_T2P_BRAM

    // Must save input BEFORE project conv overwrites out (handles in==out aliasing)
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

// MBConv for C2-scale layers (40×40) – only stride 1 or 2 variants used
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

// MBConv for C3-scale layers (20×20)
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

// MBConv for C4-scale input (10×10)
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

// ============================================================
// TRANSFORMER BLOCK (pre-norm variant used in MobileViT)
//
// Forward:
//   x1 = x + MHSA(LN1(x))
//   x2 = x1 + MLP(LN2(x1))
//
// Weight layout:
//   ln1_w[dim], ln1_b[dim]
//   Wq[dim*dim], bq[dim], Wk[dim*dim], bk[dim], Wv[dim*dim], bv[dim]
//   Wo[dim*dim], bo[dim]
//   ln2_w[dim], ln2_b[dim]
//   W1[2*dim*dim], b1[2*dim], W2[dim*2*dim], b2[dim]
//
// tokens: [N * dim]  (N = num_patches)
// ============================================================
void transformer_block(
    act_t* tokens,           // in-place update, [N * dim]
    const weight_t* w,
    int N, int dim
) {
    #pragma HLS INLINE

    // --- Allocate working buffers ---
    // Max N = MVIT_N_MAX (S2, 1600 for 640 input), max dim = MVIT_S4_DIM (240).
    // They don't coexist (S2: N=1600,d=144; S4: N=100,d=240) but we size for
    // the worst case of each dimension independently to stay resolution-safe.
    static act_t norm_buf[MVIT_N_MAX * MVIT_S4_DIM];
    static act_t q_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t k_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t v_buf   [MVIT_N_MAX * MVIT_S4_DIM];
    static act_t attn_scores[MVIT_N_MAX * MVIT_N_MAX];
    static act_t attn_out[MVIT_N_MAX * MVIT_S4_DIM];
    static act_t mlp_hidden[MVIT_N_MAX * (2 * MVIT_S4_DIM)];
    #pragma HLS RESOURCE variable=norm_buf   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=q_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=k_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=v_buf      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=attn_scores core=RAM_T2P_BRAM  // 400*400*2B=320KB
    #pragma HLS RESOURCE variable=attn_out   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=mlp_hidden core=RAM_T2P_BRAM

    int off = 0;

    // --- LN1 ---
    const weight_t* ln1_w = w + off; off += dim;
    const bias_t*   ln1_b = (const bias_t*)(w + off); off += dim;
    layer_norm_seq(tokens, norm_buf, ln1_w, ln1_b, N, dim);

    // --- QKV projections ---
    const weight_t* Wq = w + off; const bias_t* bq = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    const weight_t* Wk = w + off; const bias_t* bk = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    const weight_t* Wv = w + off; const bias_t* bv = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    linear_layer(norm_buf, q_buf, Wq, bq, N, dim, dim);
    linear_layer(norm_buf, k_buf, Wk, bk, N, dim, dim);
    linear_layer(norm_buf, v_buf, Wv, bv, N, dim, dim);

    // --- Multi-head scaled dot-product attention ---
    // TIMM mobilevit_s uses MVIT_HEADS=4; per-head scale = 1/sqrt(head_dim)
    int head_dim = dim / MVIT_HEADS;
    float scale = 1.0f / sqrtf((float)head_dim);
    MHA_HEAD: for (int h = 0; h < MVIT_HEADS; h++) {
        int hoff = h * head_dim;
        // Attention scores [N, N] using only the head_dim slice of Q and K
        ATTN_Q: for (int q = 0; q < N; q++) {
            ATTN_K: for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                acc_t dot = 0;
                ATTN_D: for (int d = 0; d < head_dim; d++) {
                    dot += (acc_t)q_buf[q*dim + hoff + d] * (acc_t)k_buf[k*dim + hoff + d];
                }
                attn_scores[q*N+k] = (act_t)((float)dot * scale);
            }
        }
        // Softmax each row
        SM_ROW: for (int q = 0; q < N; q++) {
            softmax_row(attn_scores + q*N, attn_scores + q*N, N);
        }
        // Weighted sum of V_h → attn_out[:, hoff : hoff+head_dim]
        AV_Q: for (int q = 0; q < N; q++) {
            AV_D: for (int d = 0; d < head_dim; d++) {
                #pragma HLS PIPELINE II=1
                acc_t sum = 0;
                AV_K: for (int k = 0; k < N; k++) {
                    sum += (acc_t)attn_scores[q*N+k] * (acc_t)v_buf[k*dim + hoff + d];
                }
                attn_out[q*dim + hoff + d] = (act_t)sum;
            }
        }
    }

    // --- Output projection ---
    const weight_t* Wo = w + off; const bias_t* bo = (const bias_t*)(w+off+dim*dim); off += dim*dim + dim;
    static act_t proj_out[MVIT_N_MAX * MVIT_S4_DIM];
    #pragma HLS RESOURCE variable=proj_out core=RAM_T2P_BRAM
    linear_layer(attn_out, proj_out, Wo, bo, N, dim, dim);

    // Residual 1: tokens += proj_out
    TRES1: for (int i = 0; i < N * dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] = tokens[i] + proj_out[i];
    }

    // --- LN2 ---
    const weight_t* ln2_w = w + off; off += dim;
    const bias_t*   ln2_b = (const bias_t*)(w + off); off += dim;
    layer_norm_seq(tokens, norm_buf, ln2_w, ln2_b, N, dim);

    // --- MLP: Linear(dim→2*dim) + SiLU + Linear(2*dim→dim) ---
    int hidden = dim * 2;
    const weight_t* W1 = w + off; const bias_t* b1 = (const bias_t*)(w+off+hidden*dim); off += hidden*dim + hidden;
    const weight_t* W2 = w + off; const bias_t* b2 = (const bias_t*)(w+off+dim*hidden); off += dim*hidden + dim;

    // First linear + SiLU
    MLP1_B: for (int b = 0; b < N; b++) {
        MLP1_O: for (int od = 0; od < hidden; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b1[od];
            MLP1_I: for (int id = 0; id < dim; id++) {
                acc += (acc_t)norm_buf[b*dim+id] * (acc_t)W1[od*dim+id];
            }
            mlp_hidden[b*hidden+od] = silu((act_t)acc);
        }
    }
    // Second linear
    MLP2_B: for (int b = 0; b < N; b++) {
        MLP2_O: for (int od = 0; od < dim; od++) {
            #pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)b2[od];
            MLP2_I: for (int id = 0; id < hidden; id++) {
                acc += (acc_t)mlp_hidden[b*hidden+id] * (acc_t)W2[od*hidden+id];
            }
            proj_out[b*dim+od] = (act_t)acc;
        }
    }

    // Residual 2: tokens += proj_out
    TRES2: for (int i = 0; i < N * dim; i++) {
        #pragma HLS PIPELINE II=1
        tokens[i] = tokens[i] + proj_out[i];
    }
}

// ============================================================
// MOBILEVIT BLOCK  (TIMM 1.0.x — no token_proj / token_unproj)
//
// Input:  feat [in_ch, H, W]
// Output: feat [in_ch, H, W]  (same shape, in-place conceptually)
//
// Steps:
//   1. conv1: 3x3(in_ch→in_ch) + BN + SiLU      → local_feat
//   2. conv2: 1x1(in_ch→d)  (scale=1,bias=0)     → proj_feat [d,H,W]
//   3. For each of P=p²=4 pixel positions in the patch:
//        a. Extract view_tokens[N*d]  (direct unfold, no projection)
//        b. depth × transformer_block(view_tokens, N, d)  [shared weights]
//        c. layer_norm(view_tokens)                         [shared norm]
//        d. Fold view_tokens back into fold_feat[d,H,W]
//   4. conv3: 1x1(d→in_ch) + BN + SiLU           → proj_back
//   5. Concat [local_feat, proj_back], conv4: 3x3(2*in_ch→in_ch)+BN+SiLU → out
//
// Weight layout (matches export_weights.py export_mobilevit_block):
//   conv1_w[in_ch*in_ch*9], conv1_bn_s[in_ch], conv1_bn_b[in_ch]
//   conv2_w[d*in_ch], conv2_scale=1[d], conv2_bias=0[d]   (no BN)
//   depth × transformer_block weights                       (shared across 4 views)
//   norm_w[d], norm_b[d]                                   (final LayerNorm)
//   conv3_w[in_ch*d], conv3_bn_s[in_ch], conv3_bn_b[in_ch]
//   conv4_w[in_ch*2*in_ch*9], conv4_bn_s[in_ch], conv4_bn_b[in_ch]
// ============================================================

// Stage 2: in_ch=96, d=144 (TIMM MVIT_S2_DIM), H=40, W=40, depth=2, N=400 patches
void mobilevit_block_s2(
    const act_t* input,    // [96 * 40 * 40]
    act_t*       output,   // [96 * 40 * 40]
    const weight_t* w
) {
    #pragma HLS INLINE
    const int in_ch = C2_CH, d = MVIT_S2_DIM, H = C2_H, W = C2_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);  // 400

    static act_t local_feat[C2_CH * C2_H * C2_W];           // [96, 40, 40]
    static act_t proj_feat [MVIT_S2_DIM * C2_H * C2_W];     // [144, 40, 40]
    static act_t tokens    [MVIT_N_S2 * MVIT_S2_DIM];        // [N=MVIT_N_S2, d=144]
    static act_t fold_feat [MVIT_S2_DIM * C2_H * C2_W];     // [144, 40, 40]
    static act_t proj_back [C2_CH * C2_H * C2_W];           // [96, 40, 40]
    static act_t concat_buf[2 * C2_CH * C2_H * C2_W];       // [192, 40, 40]
    #pragma HLS RESOURCE variable=local_feat core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf core=RAM_T2P_BRAM

    int off = 0;

    // 1. conv1: 3x3(in_ch→in_ch) + BN + SiLU
    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    // 2. conv2: 1x1(in_ch→d), plain Conv2d — weights only, no BN
    static const bias_t _zero_proj2[MVIT_S2_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj2, in_ch, d, H, W);
    off += d*in_ch;

    // 3–5. For each of P=4 pixel positions: unfold → transformer × depth → fold
    //      Transformer weights are shared across all 4 views.
    int tb_sz = d+d + (d*d+d)*4 + d+d + (2*d)*d+(2*d) + d*(2*d)+d;
    int transformer_off = off;  // same offset reused for each view
    off += MVIT_S2_DEPTH * tb_sz;
    // norm_w / norm_b follow the transformer weights in the stream
    const weight_t* norm_w = w + off; off += d;
    const bias_t*   norm_b = (const bias_t*)(w + off); off += d;

    S2_VIEW: for (int view = 0; view < p2; view++) {
        int pi = view / p, pj = view % p;

        // Unfold: extract this pixel position from proj_feat into tokens
        S2_UNF: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        // Transformer: depth × transformer_block (same weights, all views)
        for (int lyr = 0; lyr < MVIT_S2_DEPTH; lyr++) {
            transformer_block(tokens, w+transformer_off+lyr*tb_sz, N, d);
        }

        // Final LayerNorm (shared weights)
        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        // Fold: write tokens back to fold_feat at this pixel position
        S2_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
                    fold_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)] = tokens[n*d + c];
                }
            }
        }
    }

    // 6. conv3: 1x1(d→in_ch) + BN + SiLU
    conv1x1_bn_silu(fold_feat, proj_back, w+off, w+off+in_ch*d,
                    (const bias_t*)(w+off+in_ch*d+in_ch), d, in_ch, H, W);
    off += in_ch*d + in_ch + in_ch;

    // 7. Concat [input, proj_back] and conv4: 3x3(2*in_ch→in_ch) + BN + SiLU
    //    TIMM shortcut is the original block input, NOT conv_kxk output.
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

// Stage 3: in_ch=128, d=192 (TIMM MVIT_S3_DIM), H=20, W=20, depth=4, N=100 patches
void mobilevit_block_s3(
    const act_t* input,    // [128 * 20 * 20]
    act_t*       output,   // [128 * 20 * 20]
    const weight_t* w
) {
    #pragma HLS INLINE
    const int in_ch = C3_CH, d = MVIT_S3_DIM, H = C3_H, W = C3_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);  // 100

    static act_t local_feat[C3_CH * C3_H * C3_W];           // [128, 20, 20]
    static act_t proj_feat [MVIT_S3_DIM * C3_H * C3_W];     // [192, 20, 20]
    static act_t tokens    [MVIT_N_S3 * MVIT_S3_DIM];        // [N=MVIT_N_S3, d=192]
    static act_t fold_feat [MVIT_S3_DIM * C3_H * C3_W];     // [192, 20, 20]
    static act_t proj_back [C3_CH * C3_H * C3_W];           // [128, 20, 20]
    static act_t concat_buf[2 * C3_CH * C3_H * C3_W];       // [256, 20, 20]
    #pragma HLS RESOURCE variable=local_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf  core=RAM_T2P_BRAM

    int off = 0;

    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    static const bias_t _zero_proj2[MVIT_S3_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj2, in_ch, d, H, W);
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
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        for (int lyr = 0; lyr < MVIT_S3_DEPTH; lyr++) {
            transformer_block(tokens, w+transformer_off+lyr*tb_sz, N, d);
        }

        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        S3_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
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

// Stage 4: in_ch=160, d=240 (TIMM MVIT_S4_DIM), H=10, W=10, depth=3, N=25 patches
void mobilevit_block_s4(
    const act_t* input,    // [160 * 10 * 10]
    act_t*       output,   // [160 * 10 * 10]
    const weight_t* w
) {
    #pragma HLS INLINE
    const int in_ch = STAGE4_PRE_CH, d = MVIT_S4_DIM, H = C4_H, W = C4_W;
    const int p = MVIT_PATCH, p2 = p * p;
    const int N = (H/p) * (W/p);  // 25

    static act_t local_feat[STAGE4_PRE_CH * C4_H * C4_W];   // [160, 10, 10]
    static act_t proj_feat [MVIT_S4_DIM * C4_H * C4_W];     // [240, 10, 10]
    static act_t tokens    [MVIT_N_S4 * MVIT_S4_DIM];        // [N=MVIT_N_S4, d=240]
    static act_t fold_feat [MVIT_S4_DIM * C4_H * C4_W];     // [240, 10, 10]
    static act_t proj_back [STAGE4_PRE_CH * C4_H * C4_W];   // [160, 10, 10]
    static act_t concat_buf[2 * STAGE4_PRE_CH * C4_H * C4_W]; // [320, 10, 10]
    #pragma HLS RESOURCE variable=local_feat  core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=tokens      core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=fold_feat   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=proj_back   core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=concat_buf  core=RAM_T2P_BRAM

    int off = 0;

    conv3x3_bn_silu(input, local_feat, w+off, w+off+in_ch*in_ch*9,
                    (const bias_t*)(w+off+in_ch*in_ch*9+in_ch), in_ch, in_ch, H, W, 1);
    off += in_ch*in_ch*9 + in_ch + in_ch;

    static const bias_t _zero_proj2[MVIT_S4_DIM] = {};
    conv1x1_plain(local_feat, proj_feat, w+off, _zero_proj2, in_ch, d, H, W);
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
                    tokens[n*d + c] = proj_feat[c*H*W + (ph*p+pi)*W + (pw*p+pj)];
                }
            }
        }

        for (int lyr = 0; lyr < MVIT_S4_DEPTH; lyr++) {
            transformer_block(tokens, w+transformer_off+lyr*tb_sz, N, d);
        }

        layer_norm_seq(tokens, tokens, norm_w, norm_b, N, d);

        S4_FLD: for (int ph = 0; ph < H/p; ph++) {
            for (int pw = 0; pw < W/p; pw++) {
                #pragma HLS PIPELINE II=1
                int n = ph*(W/p) + pw;
                for (int c = 0; c < d; c++) {
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

// ============================================================
// TOP-LEVEL BACKBONE FUNCTION
// Reads image from DDR, writes C1..C4 to on-chip BRAM buffers.
// ============================================================
void mobilevit_backbone(
    const input_t*  image,       // [INPUT_C * INPUT_H * INPUT_W]
    const weight_t* weights,     // all backbone weights, flat, from DDR
    act_t* c1,                   // [C1_CH * C1_H * C1_W]
    act_t* c2,                   // [C2_CH * C2_H * C2_W]
    act_t* c3,                   // [C3_CH * C3_H * C3_W]
    act_t* c4                    // [C4_CH * C4_H * C4_W]
) {
    // On-chip intermediate feature buffers
    static act_t stem_out[STEM_CH   * STEM_H * STEM_W];   // 16*160*160 = 409.6K × 2B
    static act_t s0_out  [STAGE0_CH * STEM_H * STEM_W];   // 32*160*160 = 819.2K × 2B
    static act_t s1_a    [C1_CH     * C1_H   * C1_W  ];   // 64*80*80 = 409.6K × 2B
    static act_t s2_mb   [C2_CH     * C2_H   * C2_W  ];   // 96*40*40 = 153.6K × 2B
    static act_t s3_mb   [C3_CH     * C3_H   * C3_W  ];   // 128*20*20 = 51.2K × 2B
    static act_t s4_mb   [STAGE4_PRE_CH * C4_H * C4_W];   // 160*10*10 = 16K × 2B
    static act_t s4_mvit [STAGE4_PRE_CH * C4_H * C4_W];   // 160*10*10
    #pragma HLS RESOURCE variable=stem_out core=RAM_T2P_BRAM  // 800KB – use BRAM
    #pragma HLS RESOURCE variable=s0_out   core=RAM_T2P_BRAM  // 1.6MB – use BRAM
    #pragma HLS RESOURCE variable=s1_a     core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s2_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s3_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4_mb    core=RAM_T2P_BRAM
    #pragma HLS RESOURCE variable=s4_mvit  core=RAM_T2P_BRAM

    // Convert input_t → act_t
    CAST_IN: for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++) {
        #pragma HLS PIPELINE II=1
        // Use as temporary – reuse s0_out top region as image cast buffer
        // Actually allocate image inline during stem conv
        (void)i;  // silence unused warning; image used directly below
    }

    // weight offset tracker
    int off = 0;

    // -- Stem: Conv3x3(3→16, s=2) + BN + SiLU --
    // input_t=float=act_t, so reinterpret_cast is a no-op
    const act_t* img_act = reinterpret_cast<const act_t*>(image);
    conv3x3_bn_silu(img_act, stem_out,
                    weights+off, weights+off+STEM_CH*INPUT_C*9, (const bias_t*)(weights+off+STEM_CH*INPUT_C*9+STEM_CH),
                    INPUT_C, STEM_CH, INPUT_H, INPUT_W, 2);
    off += STEM_CH*INPUT_C*9 + STEM_CH + STEM_CH;
    fake_quant_buf(stem_out, STEM_CH*STEM_H*STEM_W, 0.09562f); // backbone.model.stem.conv

    // -- Stage 0: MBConv(16→32, expand=4, s=1) --
    mbconv_160(stem_out, s0_out, weights+off, STEM_CH, STAGE0_CH, MBCONV_EXPAND, 1);
    {   // advance offset: expand(16→64): 64*16+64+64; dw(64*9): 64*9+64+64; proj(32*64): 32*64+32+32
        int hid0 = STEM_CH*MBCONV_EXPAND;
        off += hid0*STEM_CH+hid0+hid0 + hid0*9+hid0+hid0 + STAGE0_CH*hid0+STAGE0_CH+STAGE0_CH;
    }
    fake_quant_buf(s0_out, STAGE0_CH*STEM_H*STEM_W, 0.39651f); // stages_0.0.conv3_1x1

    // -- Stage 1: MBConv(32→64, s=2) + 2×MBConv(64→64, s=1) --
    mbconv_160(s0_out, s1_a, weights+off, STAGE0_CH, C1_CH, MBCONV_EXPAND, 2);
    {
        int hid1 = STAGE0_CH*MBCONV_EXPAND;
        off += hid1*STAGE0_CH+hid1+hid1 + hid1*9+hid1+hid1 + C1_CH*hid1+C1_CH+C1_CH;
    }
    fake_quant_buf(s1_a, C1_CH*C1_H*C1_W, 0.11798f); // stages_1.0.conv3_1x1

    // Two MBConv(64→64, s=1) blocks – result goes to c1
    mbconv_80(s1_a, c1, weights+off, C1_CH, C1_CH, MBCONV_EXPAND, 1);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C1_CH*hid+C1_CH+C1_CH;
    }
    fake_quant_buf(c1, C1_CH*C1_H*C1_W, 0.12537f); // stages_1.1.conv3_1x1

    mbconv_80(c1, c1, weights+off, C1_CH, C1_CH, MBCONV_EXPAND, 1);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C1_CH*hid+C1_CH+C1_CH;
    }
    fake_quant_buf(c1, C1_CH*C1_H*C1_W, 0.06396f); // stages_1.2.conv3_1x1
    // c1 = [64, 160, 160]  ← FPN input at stride 4

    // -- Stage 2: MBConv(64→96, s=2) + MobileViTBlock --
    mbconv_80(c1, s2_mb, weights+off, C1_CH, C2_CH, MBCONV_EXPAND, 2);
    {
        int hid = C1_CH*MBCONV_EXPAND;
        off += hid*C1_CH+hid+hid + hid*9+hid+hid + C2_CH*hid+C2_CH+C2_CH;
    }
    fake_quant_buf(s2_mb, C2_CH*C2_H*C2_W, 0.06428f); // stages_2.0.conv3_1x1
    mobilevit_block_s2(s2_mb, c2, weights+off);
    // MobileViT S2 size (in_ch=96, d=144, depth=2):
    //   conv_kxk: 96*96*9+96+96 = 83136
    //   conv_1x1: 144*96 = 13824 (weights only, no scale/bias)
    //   2 × transformer_block(d=144): 2*167472 = 334944
    //   norm: 144+144 = 288
    //   conv_proj: 96*144+96+96 = 14016
    //   conv_fusion: 96*192*9+96+96 = 166080
    //   TOTAL = 612288
    off += 612288;
    fake_quant_buf(c2, C2_CH*C2_H*C2_W, 0.20161f); // stages_2.1.conv_fusion
    // c2 = [96, 80, 80]  ← FPN input at stride 8

    // -- Stage 3: MBConv(96→128, s=2) + MobileViTBlock --
    static act_t s2_out[C2_CH * C2_H * C2_W];
    buf_copy(s2_out, c2, C2_CH * C2_H * C2_W);
    mbconv_40(s2_out, s3_mb, weights+off, C2_CH, C3_CH, MBCONV_EXPAND, 2);
    {
        int hid = C2_CH*MBCONV_EXPAND;
        off += hid*C2_CH+hid+hid + hid*9+hid+hid + C3_CH*hid+C3_CH+C3_CH;
    }
    fake_quant_buf(s3_mb, C3_CH*C3_H*C3_W, 0.07822f); // stages_3.0.conv3_1x1
    mobilevit_block_s3(s3_mb, c3, weights+off);
    // MobileViT S3 size (in_ch=128, d=192, depth=4):
    //   conv_kxk: 128*128*9+128+128 = 147712
    //   conv_1x1: 192*128 = 24576 (weights only, no scale/bias)
    //   4 × transformer_block(d=192): 4*297024 = 1188096
    //   norm: 192+192 = 384
    //   conv_proj: 128*192+128+128 = 24832
    //   conv_fusion: 128*256*9+128+128 = 295168
    //   TOTAL = 1680768
    off += 1680768;
    fake_quant_buf(c3, C3_CH*C3_H*C3_W, 0.21355f); // stages_3.1.conv_fusion
    // c3 = [128, 40, 40]  ← FPN input at stride 16

    // -- Stage 4: MBConv(128→160, s=2) + MobileViTBlock + Conv1x1(160→640) --
    static act_t s3_out[C3_CH * C3_H * C3_W];
    buf_copy(s3_out, c3, C3_CH * C3_H * C3_W);
    mbconv_20(s3_out, s4_mb, weights+off, C3_CH, STAGE4_PRE_CH, MBCONV_EXPAND, 2);
    {
        int hid = C3_CH*MBCONV_EXPAND;
        off += hid*C3_CH+hid+hid + hid*9+hid+hid + STAGE4_PRE_CH*hid+STAGE4_PRE_CH+STAGE4_PRE_CH;
    }
    fake_quant_buf(s4_mb, STAGE4_PRE_CH*C4_H*C4_W, 0.06868f); // stages_4.0.conv3_1x1
    mobilevit_block_s4(s4_mb, s4_mvit, weights+off);
    // MobileViT S4 size (in_ch=160, d=240, depth=3):
    //   conv_kxk: 160*160*9+160+160 = 230720
    //   conv_1x1: 240*160 = 38400 (weights only, no scale/bias)
    //   3 × transformer_block(d=240): 3*463440 = 1390320
    //   norm: 240+240 = 480
    //   conv_proj: 160*240+160+160 = 38720
    //   conv_fusion: 160*320*9+160+160 = 461120
    //   TOTAL = 2159760
    off += 2159760;
    fake_quant_buf(s4_mvit, STAGE4_PRE_CH*C4_H*C4_W, 0.16822f); // stages_4.1.conv_fusion

    // Final 1x1 expansion: Conv1x1(160→640) + BN + SiLU
    // off now points exactly to the final_conv weights
    const weight_t* exp_conv_w  = weights + off;
    const weight_t* exp_bn_s    = exp_conv_w + C4_CH * STAGE4_PRE_CH;
    const bias_t*   exp_bn_b    = (const bias_t*)(exp_bn_s + C4_CH);
    conv1x1_bn_silu(s4_mvit, c4, exp_conv_w, exp_bn_s, exp_bn_b, STAGE4_PRE_CH, C4_CH, C4_H, C4_W);
    fake_quant_buf(c4, C4_CH*C4_H*C4_W, 0.02812f); // backbone.model.final_conv
    // c4 = [640, 20, 20]  ← FPN input at stride 32
}

#endif // MOBILEVIT_BACKBONE_H
