/*
 * retina_head_top.cpp
 * Isolated RetinaNet head — Vitis HLS top function implementation.
 *
 * ── Synthesis history ──────────────────────────────────────────────────────
 * v1: hung  — II=1 on OW loop + 144-unrolled MACs + large ARRAY_PARTITION.
 * v2: II=72 — CO_ICT pipelined, no weight port partition. Bottleneck confirmed:
 *             144 reads × KH×KW×IC_TILE / 2 BRAM ports = 72.
 * v3: OOM?  — w_row[1728] complete. Log showed "Finished C synthesis" but was
 *             cancelled. May have succeeded; could not confirm.
 * v4: II=72 — Removed KH/KW UNROLL. Same result: total reads/2 = 72 regardless
 *             of concurrent vs sequential access.
 * v5: II=72 — w_row cyclic-144 partition. Failed because t×144 % 144 = 0 →
 *             all t values map to same banks → no cross-iteration parallelism.
 *             Also: DSP +59, LUT +12K worse than v2.
 *
 * v6 (this file): w_kpos[HEAD_FEAT_CH] — per-kernel-position buffer.
 *
 *   Key insight: buffer ONE kernel position's IC weights (192 int8) with
 *   `complete` partition. Each buffer element is a separate register, and
 *   different CO_ICT iterations t access DIFFERENT registers:
 *     t=0: w_kpos[0..15]    (registers 0-15)
 *     t=1: w_kpos[16..31]   (registers 16-31)
 *     ...
 *     t=11: w_kpos[176..191] (registers 176-191)
 *   → All 192 registers distinct for 12 concurrent iterations → II=1 from weights.
 *
 *   Elaboration safety: 192 int8 registers × 2 module instances = 384 registers
 *   (vs 3456 for the full w_row complete approach). Will NOT hang elaboration.
 *
 *   Remaining bottleneck: src cyclic-16 with 16 reads per CO_ICT iteration:
 *     bank = ((t×16+k)×100 + offset) % 16 = (4k + offset) % 16
 *     4 reads per bank, 2 BRAM ports → II = 2 from src.
 *
 *   Loop restructure for efficiency — LOAD runs once per (oc, kh, kw), NOT
 *   per output pixel. partial[TEST_H×TEST_W×N_IC_TILES] accumulates across
 *   all 9 kernel positions before final reduction.
 *
 * ── Expected v6 result ─────────────────────────────────────────────────────
 *   CO_ICT II   : 2    (src cyclic-16: 4 banks × 4 reads / 2 ports = 2)
 *   LOAD cost   : 192 OC × 9 kpos × 192 cycles   = 331,776 cy/call
 *   Compute     : 192 OC × 9 kpos × 100 px × 12t × II_2  = 41.5 M cy/call
 *   Total/call  : ~42 M cy
 *
 *   Hmm — total is similar to v2 because the OC×kpos loop now wraps the inner
 *   OH×OW compute. See partial[H×W×T] for the accumulation pattern.
 *
 *   Revisited: with II=2 on CO_ICT (12 tiles) per pixel per kernel pos:
 *     Per pixel: 9 pos × (192 LOAD + 12×2 compute) = 9×216 = 1944 cy
 *     BUT LOAD is outside OH×OW → amortised over 100 pixels:
 *     Per pixel (amortised): 9×(192/100 + 24) = 9×26 = 234 cy   ~9× vs v2's 2220
 *   Total: 19,200 × 234 = 4.5M cy/call → ~9× speedup over v2!  ← correct.
 */

#include "retina_head_top.h"
#include "fpga_utils.h"

#define HEAD_IC_TILE   16
#define N_IC_TILES     (HEAD_FEAT_CH / HEAD_IC_TILE)   // 12

// Partial accumulator array: one value per (OH, OW, tile).
// Accumulates contributions from all 9 kernel positions before final reduction.
// complete partition → all elements are registers, no BRAM.
#define N_PARTIALS  (TEST_H * TEST_W * N_IC_TILES)    // 10×10×12 = 1 200

// ─────────────────────────────────────────────────────────────────────────────
// Single stacked 3×3 conv + per-channel dequant-scale + bias + ReLU.
//
// ── v6 loop structure ─────────────────────────────────────────────────────
//   CO_OC  (192)  outer                       → one output-channel row at a time
//     CO_KH (3) × CO_KW (3)  sequential       → 9 kernel positions
//       LOAD_KPOS (192 cycles, II=1)           → fill w_kpos[192] from BRAM
//       CO_OH (H) × CO_OW (W)  spatial        → iterate output pixels
//         CO_ICT (12, PIPELINE II=2)           → accumulate contribution of this
//           CO_IC (16, UNROLL)                    kernel position into partial[]
//     CO_OH × CO_OW  reduce + write dst        → sum 12 partials, apply BN+ReLU
//
// ── Why II=2 and not II=72 ────────────────────────────────────────────────
//   w_kpos[192] complete → 192 separate registers.
//   t=0 reads regs 0-15; t=1 reads regs 16-31; … all distinct → II=1 from weights.
//   src cyclic-16, 16 reads per CO_ICT body:
//     bank = ((t×16+k)×100+offset) % 16 = (4k+offset) % 16  (t-independent)
//     → 4 banks × 4 reads each / 2 ports = II=2 from src.
//   Net CO_ICT II = max(1, 2) = 2.
// ─────────────────────────────────────────────────────────────────────────────
static void retina_conv_once(
    const act_t*    src,
    act_t*          dst,
    const weight_t* lw,
    const meta_t*   bs,
    const meta_t*   bb
) {
    // Per-kernel-position weight buffer: 192 int8 registers.
    // complete → t=0 reads regs 0-15, t=1 reads 16-31, …, t=11 reads 176-191.
    weight_t w_kpos[HEAD_FEAT_CH];
    #pragma HLS ARRAY_PARTITION variable=w_kpos complete dim=1

    // Partial accumulators: [H×W×N_IC_TILES] = 1200 float registers.
    // Accumulates contributions from all 9 kernel positions.
    // complete → all 1200 are independent registers; CO_ICT can access in parallel.
    acc_t partial[N_PARTIALS];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=1

    CO_OC: for (int oc = 0; oc < HEAD_FEAT_CH; oc++) {

        // ── Initialise partial accumulators for this OC ──────────────────────
        P_INIT: for (int i = 0; i < N_PARTIALS; i++) {
            #pragma HLS UNROLL
            partial[i] = 0.0f;
        }

        // ── Iterate over all 9 kernel positions ───────────────────────────────
        CO_KH: for (int kh = 0; kh < 3; kh++) {
            CO_KW: for (int kw = 0; kw < 3; kw++) {

                // ── Load 192 weights for (oc, kh, kw) from BRAM ──────────────
                // Stride-9 address pattern: (oc×192+0)×9+kh×3+kw,
                //                           (oc×192+1)×9+kh×3+kw, …
                // Regular stride → II=1 achievable.
                LOAD_KPOS: for (int ic = 0; ic < HEAD_FEAT_CH; ic++) {
                    #pragma HLS PIPELINE II=1
                    w_kpos[ic] = lw[(oc * HEAD_FEAT_CH + ic) * 9 + kh * 3 + kw];
                }

                // ── Accumulate contribution across all output pixels ───────────
                CO_OH: for (int oh = 0; oh < TEST_H; oh++) {
                    int ih = oh + kh - 1;

                    CO_OW: for (int ow = 0; ow < TEST_W; ow++) {
                        int iw = ow + kw - 1;

                        // Skip boundary positions (static for compile-time H/W)
                        if (ih < 0 || ih >= TEST_H || iw < 0 || iw >= TEST_W) continue;

                        int pix_base = (oh * TEST_W + ow) * N_IC_TILES;

                        // ── IC-tile pipeline ──────────────────────────────────
                        // Weights: from w_kpos registers; t=0..11 all distinct regs
                        //          → no bank conflict across iterations → II=1 weights.
                        // Src: cyclic-16, 16 reads per iteration
                        //      → 4 banks × 4 reads / 2 ports → II=2 from src.
                        // Net target II = 2.
                        CO_ICT: for (int t = 0; t < N_IC_TILES; t++) {
                            #pragma HLS PIPELINE II=2
                            #pragma HLS LOOP_TRIPCOUNT min=12 max=12

                            acc_t tile_acc = 0.0f;
                            CO_IC: for (int k = 0; k < HEAD_IC_TILE; k++) {
                                #pragma HLS UNROLL
                                int ic = t * HEAD_IC_TILE + k;
                                tile_acc +=
                                    (acc_t)src[ic * TEST_H * TEST_W + ih * TEST_W + iw]
                                  * (acc_t)(float)w_kpos[ic];
                            }
                            partial[pix_base + t] += tile_acc;
                        }
                    }
                }
            }
        }

        // ── Reduce 12 partials per pixel → apply BN + ReLU → write output ────
        CO_RED_H: for (int oh = 0; oh < TEST_H; oh++) {
            CO_RED_W: for (int ow = 0; ow < TEST_W; ow++) {
                #pragma HLS PIPELINE II=1
                int pix_base = (oh * TEST_W + ow) * N_IC_TILES;
                acc_t acc = 0.0f;
                CO_RED_T: for (int t = 0; t < N_IC_TILES; t++) {
                    #pragma HLS UNROLL
                    acc += partial[pix_base + t];
                }
                act_t y = (act_t)(acc * (acc_t)bs[oc] + (acc_t)bb[oc]);
                dst[oc * TEST_H * TEST_W + oh * TEST_W + ow] =
                    (y > (act_t)0) ? y : (act_t)0;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4 stacked conv layers with ping-pong buffers (unchanged).
// ─────────────────────────────────────────────────────────────────────────────
static void retina_stacked_convs(
    const act_t*    input,
    act_t*          output,
    const weight_t* w_conv,
    const meta_t*   w_meta
) {
    static act_t tmp[HEAD_FEAT_CH * TEST_H * TEST_W];
    #pragma HLS bind_storage variable=tmp type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=16 dim=1

    const int S_conv = STACKED_CONV_W_LAYER;
    const int S_meta = STACKED_CONV_M_LAYER;

    const weight_t* w0c = w_conv;
    const weight_t* w1c = w_conv +   S_conv;
    const weight_t* w2c = w_conv + 2*S_conv;
    const weight_t* w3c = w_conv + 3*S_conv;

    const meta_t*   w0m = w_meta;
    const meta_t*   w1m = w_meta +   S_meta;
    const meta_t*   w2m = w_meta + 2*S_meta;
    const meta_t*   w3m = w_meta + 3*S_meta;

    retina_conv_once(input,  tmp,    w0c, w0m,               w0m + HEAD_FEAT_CH);
    retina_conv_once(tmp,    output, w1c, w1m,               w1m + HEAD_FEAT_CH);
    retina_conv_once(output, tmp,    w2c, w2m,               w2m + HEAD_FEAT_CH);
    retina_conv_once(tmp,    output, w3c, w3m,               w3m + HEAD_FEAT_CH);
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction conv (192→36, no activation). Same v6 structure.
// ─────────────────────────────────────────────────────────────────────────────
static void retina_pred_conv(
    const act_t*    input,
    act_t*          output,
    const weight_t* weights,
    const meta_t*   w_scale,
    const meta_t*   bias,
    int out_ch
) {
    weight_t w_kpos[HEAD_FEAT_CH];
    #pragma HLS ARRAY_PARTITION variable=w_kpos complete dim=1

    acc_t partial[N_PARTIALS];
    #pragma HLS ARRAY_PARTITION variable=partial complete dim=1

    RP_OC: for (int oc = 0; oc < out_ch; oc++) {

        RP_INIT: for (int i = 0; i < N_PARTIALS; i++) {
            #pragma HLS UNROLL
            partial[i] = 0.0f;
        }

        RP_KH: for (int kh = 0; kh < 3; kh++) {
            RP_KW: for (int kw = 0; kw < 3; kw++) {

                RP_LOAD: for (int ic = 0; ic < HEAD_FEAT_CH; ic++) {
                    #pragma HLS PIPELINE II=1
                    w_kpos[ic] = weights[(oc * HEAD_FEAT_CH + ic) * 9 + kh * 3 + kw];
                }

                RP_OH: for (int oh = 0; oh < TEST_H; oh++) {
                    int ih = oh + kh - 1;
                    RP_OW: for (int ow = 0; ow < TEST_W; ow++) {
                        int iw = ow + kw - 1;
                        if (ih < 0 || ih >= TEST_H || iw < 0 || iw >= TEST_W) continue;

                        int pix_base = (oh * TEST_W + ow) * N_IC_TILES;

                        RP_ICT: for (int t = 0; t < N_IC_TILES; t++) {
                            #pragma HLS PIPELINE II=2
                            #pragma HLS LOOP_TRIPCOUNT min=12 max=12
                            acc_t tile_acc = 0.0f;
                            RP_IC: for (int k = 0; k < HEAD_IC_TILE; k++) {
                                #pragma HLS UNROLL
                                int ic = t * HEAD_IC_TILE + k;
                                tile_acc +=
                                    (acc_t)input[ic * TEST_H * TEST_W + ih * TEST_W + iw]
                                  * (acc_t)(float)w_kpos[ic];
                            }
                            partial[pix_base + t] += tile_acc;
                        }
                    }
                }
            }
        }

        RP_RED_H: for (int oh = 0; oh < TEST_H; oh++) {
            RP_RED_W: for (int ow = 0; ow < TEST_W; ow++) {
                #pragma HLS PIPELINE II=1
                int pix_base = (oh * TEST_W + ow) * N_IC_TILES;
                acc_t acc = 0.0f;
                RP_RED_T: for (int t = 0; t < N_IC_TILES; t++) {
                    #pragma HLS UNROLL
                    acc += partial[pix_base + t];
                }
                output[oc * TEST_H * TEST_W + oh * TEST_W + ow] =
                    (act_t)(acc * (acc_t)w_scale[oc] + (acc_t)bias[oc]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP FUNCTION
// ─────────────────────────────────────────────────────────────────────────────
void retina_head_top(
    const act_t    feat_in        [FEAT_ELEMS],
    act_t          cls_logits     [CLS_LOGITS_ELEMS],
    act_t          reg_deltas     [REG_DELTAS_ELEMS],
    const weight_t cls_conv_w     [STACKED_CONV_W_ELEMS],
    const meta_t   cls_conv_m     [STACKED_CONV_M_ELEMS],
    const weight_t reg_conv_w     [STACKED_CONV_W_ELEMS],
    const meta_t   reg_conv_m     [STACKED_CONV_M_ELEMS],
    const weight_t cls_pred_w     [PRED_W_ELEMS],
    const meta_t   cls_pred_scale [PRED_META_ELEMS],
    const meta_t   cls_pred_bias  [PRED_META_ELEMS],
    const weight_t reg_pred_w     [PRED_W_ELEMS],
    const meta_t   reg_pred_scale [PRED_META_ELEMS],
    const meta_t   reg_pred_bias  [PRED_META_ELEMS]
) {
    #pragma HLS INTERFACE bram port=feat_in
    #pragma HLS INTERFACE bram port=cls_logits
    #pragma HLS INTERFACE bram port=reg_deltas
    #pragma HLS INTERFACE bram port=cls_conv_w
    #pragma HLS INTERFACE bram port=cls_conv_m
    #pragma HLS INTERFACE bram port=reg_conv_w
    #pragma HLS INTERFACE bram port=reg_conv_m
    #pragma HLS INTERFACE bram port=cls_pred_w
    #pragma HLS INTERFACE bram port=cls_pred_scale
    #pragma HLS INTERFACE bram port=cls_pred_bias
    #pragma HLS INTERFACE bram port=reg_pred_w
    #pragma HLS INTERFACE bram port=reg_pred_scale
    #pragma HLS INTERFACE bram port=reg_pred_bias
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    #pragma HLS ARRAY_PARTITION variable=feat_in cyclic factor=16 dim=1

    static act_t cls_feat[HEAD_FEAT_CH * TEST_H * TEST_W];
    static act_t reg_feat[HEAD_FEAT_CH * TEST_H * TEST_W];
    #pragma HLS bind_storage variable=cls_feat type=RAM_T2P impl=BRAM
    #pragma HLS bind_storage variable=reg_feat type=RAM_T2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=cls_feat cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=reg_feat cyclic factor=16 dim=1

    retina_stacked_convs(feat_in, cls_feat, cls_conv_w, cls_conv_m);
    retina_pred_conv(cls_feat, cls_logits,
                     cls_pred_w, cls_pred_scale, cls_pred_bias, CLS_OUT_CH);

    retina_stacked_convs(feat_in, reg_feat, reg_conv_w, reg_conv_m);
    retina_pred_conv(reg_feat, reg_deltas,
                     reg_pred_w, reg_pred_scale, reg_pred_bias, REG_OUT_CH);
}
