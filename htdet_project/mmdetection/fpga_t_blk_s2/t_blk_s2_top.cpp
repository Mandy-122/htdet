/*
 * t_blk_s2_top.cpp
 * HLS top for transformer_block_s2 — isolated C-synthesis / CSIM entry point.
 *
 * Implements one stack of TB_DEPTH TransformerBlock(s):
 *   for each block d:
 *     x = x + MHSA(LN1(x))      ← self-attention + residual
 *     x = x + MLP(LN2(x))       ← feed-forward   + residual
 *
 * All working buffers are static (allocated once, reused across depth layers).
 * For TB_DEPTH=1 (current), the depth loop runs once — straightforward INLINE.
 * Scaling note: when increasing TB_DEPTH > 1, the static buffers inside
 *   transformer_block() are shared correctly for sequential iterations.
 *
 * Buffer map (sizes for DIM=16, SEQ=4, MLP_HID=32):
 *   x_buf     [64]  — running token sequence (updated in-place each depth layer)
 *   ln_buf    [64]  — LN1 / LN2 output (reused)
 *   q_buf     [64]  — Q projection (also reused as temp before output proj)
 *   k_buf     [64]  — K projection
 *   v_buf     [64]  — V projection
 *   a_buf     [16]  — attention scores, one head at a time (SEQ*SEQ, fully partitioned)
 *   attn_out  [64]  — MHSA output after output proj / MLP fc2 output (reused)
 *   mlp_buf  [128]  — MLP fc1+SiLU intermediate
 */

#include "t_blk_s2_top.h"
#include "t_blk_s2_utils.h"

// ── Single transformer block (inlined into top) ───────────────────────────────
static void transformer_block(
    act_t*        x,           // [TB_SEQ * TB_DIM] — updated in-place
    const meta_t* bw,          // block weights starting at BLK_Q_W_OFF
    act_t*        ln_buf,
    act_t*        q_buf,
    act_t*        k_buf,
    act_t*        v_buf,
    act_t*        a_buf,       // [TB_SEQ * TB_SEQ]
    act_t*        attn_out,    // [TB_SEQ * TB_DIM] — reused for MLP fc2 out
    act_t*        mlp_buf      // [TB_SEQ * TB_MLP_HID]
) {
    //#pragma HLS INLINE

    // ── Self-Attention branch ─────────────────────────────────────────────────
    // LN1
    layer_norm(x, ln_buf,
               bw + BLK_LN1_W_OFF, bw + BLK_LN1_B_OFF,
               TB_SEQ, TB_DIM);

    // MHSA (Q/K/V project → per-head attn → output proj) → attn_out
    mhsa(ln_buf, attn_out,
         bw + BLK_Q_W_OFF, bw + BLK_Q_B_OFF,
         bw + BLK_K_W_OFF, bw + BLK_K_B_OFF,
         bw + BLK_V_W_OFF, bw + BLK_V_B_OFF,
         bw + BLK_O_W_OFF, bw + BLK_O_B_OFF,
         q_buf, k_buf, v_buf, a_buf);

    // Residual 1: x += attn_out
    add_residual(x, attn_out, TB_SEQ * TB_DIM);

    // ── MLP branch ───────────────────────────────────────────────────────────
    // LN2  (reuse ln_buf)
    layer_norm(x, ln_buf,
               bw + BLK_LN2_W_OFF, bw + BLK_LN2_B_OFF,
               TB_SEQ, TB_DIM);

    // fc1 + SiLU
    linear_f(ln_buf, mlp_buf,
             bw + BLK_FC1_W_OFF, bw + BLK_FC1_B_OFF,
             TB_SEQ, TB_DIM, TB_MLP_HID);

    MLP_ACT: for (int i = 0; i < TB_SEQ * TB_MLP_HID; i++) {
        //#pragma HLS PIPELINE II=1
        mlp_buf[i] = silu_f(mlp_buf[i]);
    }

    // fc2 → reuse attn_out (MHSA is done; safe to overwrite)
    linear_f(mlp_buf, attn_out,
             bw + BLK_FC2_W_OFF, bw + BLK_FC2_B_OFF,
             TB_SEQ, TB_MLP_HID, TB_DIM);

    // Residual 2: x += fc2_out
    add_residual(x, attn_out, TB_SEQ * TB_DIM);
}

// ── HLS top function ─────────────────────────────────────────────────────────
void transformer_blk_s2_top(
    const act_t  in  [TB_IN_ELEMS],
          act_t  out [TB_OUT_ELEMS],
    const meta_t w   [TB_TOTAL_W_ELEMS]
) {
    //#pragma HLS INTERFACE bram port=in
    //#pragma HLS INTERFACE bram port=out
    //#pragma HLS INTERFACE bram port=w
    //#pragma HLS INTERFACE ap_ctrl_hs port=return

    // ── Working buffers (static → BRAM / register in RTL) ────────────────────
    static act_t x_buf   [TB_SEQ * TB_DIM];
    static act_t ln_buf  [TB_SEQ * TB_DIM];
    static act_t q_buf   [TB_SEQ * TB_DIM];
    static act_t k_buf   [TB_SEQ * TB_DIM];
    static act_t v_buf   [TB_SEQ * TB_DIM];
    static act_t a_buf   [TB_SEQ * TB_SEQ];   // 16 floats — fits in registers
    static act_t attn_out[TB_SEQ * TB_DIM];
    static act_t mlp_buf [TB_SEQ * TB_MLP_HID];

    //#pragma HLS bind_storage variable=attn_out type=RAM_T2P impl=BRAM
    //#pragma HLS bind_storage variable=mlp_buf  type=RAM_T2P impl=BRAM

    // ── a_buf: cyclic factor=5 (not complete) ────────────────────────────────
    // Complete partition at SEQ=100 → 10,000 registers → compile time explosion.
    // Factor=5 (prime): 5 banks, 2 ports each = 10 simultaneous accesses.
    //   SM_MAX PIPELINE: 1 read/cycle → 1 bank → fine.
    //   SM_EXP/SM_NORM PIPELINE: 1 read/write/cycle → fine.
    //   MHSA_QI_MHSA_KI: 1 write/cycle → fine.
    //   MHSA_OK_T (tile, UNROLL factor=4): reads at qi*SEQ+t*4+0..3; consecutive
    //     addresses modulo 5 → 4 distinct banks → II=1 on inner tile loop. ✓
    //   MHSA_SM PIPELINE+DEPENDENCE: no inter-iteration dependency.
    //#pragma HLS ARRAY_PARTITION variable=a_buf cyclic factor=5 dim=1
    //#pragma HLS DEPENDENCE variable=a_buf inter false

    // ── x_buf & ln_buf: cyclic factor=4 ──────────────────────────────────────
    // LN_MEAN/LN_VAR tile loop reads 4 consecutive elements per pipeline stage.
    // factor=4 → 4 banks, each tile's 4 reads hit unique banks → II=1.
    //#pragma HLS ARRAY_PARTITION variable=x_buf  cyclic factor=4 dim=1
    //#pragma HLS ARRAY_PARTITION variable=ln_buf cyclic factor=4 dim=1

    // ── q_buf & k_buf: cyclic factor=TB_HEAD_DIM ─────────────────────────────
    // MHSA_D inner loop (HEAD_DIM reads, stride=1) is fully unrolled.
    // factor=HD → each of the HD reads maps to a unique bank → II=1.
    // Verified: (h*HD + d) % HD = d for d=0..HD-1 → HD unique banks. ✓
    // Applies to MHSA_QI_MHSA_KI which reads q_buf and k_buf with stride-1.
    //#pragma HLS ARRAY_PARTITION variable=q_buf cyclic factor=TB_HEAD_DIM dim=1
    //#pragma HLS ARRAY_PARTITION variable=k_buf cyclic factor=TB_HEAD_DIM dim=1

    // ── v_buf: cyclic factor=TB_VBUF_FACTOR (= TB_SEQ+1, a prime) ────────────
    // MHSA_OQ_MHSA_OD unrolls MHSA_OK: TB_SEQ reads at stride TB_DIM per stage.
    // Factor = smallest prime > TB_SEQ ensures all ki*DIM values hit unique banks.
    //   Step1/2: factor=5  → 4 reads at stride DIM, all to unique banks ✓
    //   Step3:   factor=17 → 16 reads at stride 64, all to unique banks ✓
    //   Step4/5: factor=101/401 (prime) → verified ✓
    // WARNING: do NOT use factor=TB_HEAD_DIM — ki*DIM%HD=0 for DIM multiple of HD.
    //#pragma HLS ARRAY_PARTITION variable=v_buf cyclic factor=TB_VBUF_FACTOR dim=1

    // ── Load input ────────────────────────────────────────────────────────────
    COPY_IN: for (int i = 0; i < TB_IN_ELEMS; i++) {
        //#pragma HLS PIPELINE II=1
        x_buf[i] = in[i];
    }

    // ── Apply TB_DEPTH transformer blocks sequentially ────────────────────────
    DEPTH: for (int d = 0; d < TB_DEPTH; d++) {
        transformer_block(
            x_buf,
            w + d * TB_BLK_W_ELEMS,
            ln_buf, q_buf, k_buf, v_buf,
            a_buf, attn_out, mlp_buf
        );
    }

    // ── Store output ──────────────────────────────────────────────────────────
    COPY_OUT: for (int i = 0; i < TB_OUT_ELEMS; i++) {
        //#pragma HLS PIPELINE II=1
        out[i] = x_buf[i];
    }
}
