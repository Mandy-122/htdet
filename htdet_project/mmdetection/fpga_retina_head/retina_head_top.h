    /*
    * retina_head_top.h
    * Isolated RetinaNet detection head for Vitis HLS C-synthesis.
    *
    * Tests ONE FPN level end-to-end:
    *
    *   feat_in [HEAD_FEAT_CH × TEST_H × TEST_W]  (CHW layout)
    *       ├─ cls branch: 4× Conv3x3(192→192, int8 w + dequant-scale + bias, ReLU)
    *       │              → Conv3x3(192→36, int8 w + scale + bias)
    *       │              → cls_logits [36 × TEST_H × TEST_W]   (pre-sigmoid)
    *       └─ reg branch: same 4-layer stack
    *                      → Conv3x3(192→36, …)
    *                      → reg_deltas [36 × TEST_H × TEST_W]   (pre-exp)
    *
    * Set TEST_H / TEST_W in fpga_types.h (or via -D flag) to select the level:
    *   P6: 10×10   P5: 20×20   P4: 40×40   P3: 80×80   P2: 160×160
    *
    * ══ WEIGHT LAYOUT ════════════════════════════════════════════════════════════
    *
    * cls_conv_w  [int8,  STACKED_CONV_W_ELEMS = 1 327 104]
    *   Flat 4 × layer; each layer = HEAD_FEAT_CH × HEAD_FEAT_CH × 9 = 331 776
    *   Index: layer * 331776 + oc * 192 * 9 + ic * 9 + kh*3+kw
    *
    * cls_conv_m  [float, STACKED_CONV_M_ELEMS = 1 536]
    *   Flat 4 × layer; each layer = [w_dequant_scale[192], bias[192]]
    *   Index: layer * 384 + [0..191]=scale, [192..383]=bias
    *
    * reg_conv_w / reg_conv_m  — identical layout to cls
    *
    * cls_pred_w  [int8,  PRED_W_ELEMS = 62 208]
    *   out_ch × HEAD_FEAT_CH × 9   where out_ch = CLS_OUT_CH = 36
    *   Index: oc * 192 * 9 + ic * 9 + kh*3+kw
    *
    * cls_pred_scale  [float, PRED_META_ELEMS = 36]   per-channel dequant scale
    * cls_pred_bias   [float, PRED_META_ELEMS = 36]   per-channel bias
    *
    * reg_pred_w / reg_pred_scale / reg_pred_bias  — identical layout to cls_pred
    *   (REG_OUT_CH = 36 = CLS_OUT_CH, so PRED_W_ELEMS / PRED_META_ELEMS reused)
    * ═════════════════════════════════════════════════════════════════════════════
    */

    #ifndef RETINA_HEAD_TOP_H
    #define RETINA_HEAD_TOP_H

    #include "fpga_types.h"

    // ── output channel counts ──────────────────────────────────────────────────
    #define CLS_OUT_CH  (ANCHORS_PER_LOC * NUM_CLASSES)   // 9×4 = 36
    #define REG_OUT_CH  (ANCHORS_PER_LOC * 4)             // 9×4 = 36

    // ── element counts ─────────────────────────────────────────────────────────
    #define FEAT_ELEMS            (HEAD_FEAT_CH * TEST_H * TEST_W)          // 192×H×W
    #define CLS_LOGITS_ELEMS      (CLS_OUT_CH   * TEST_H * TEST_W)          //  36×H×W
    #define REG_DELTAS_ELEMS      (REG_OUT_CH   * TEST_H * TEST_W)          //  36×H×W

    // Stacked conv weights (int8): 4 layers × [192×192×9]
    #define STACKED_CONV_W_LAYER  (HEAD_FEAT_CH * HEAD_FEAT_CH * 9)         // 331 776
    #define STACKED_CONV_W_ELEMS  (HEAD_STACKED_CONVS * STACKED_CONV_W_LAYER) // 1 327 104

    // Stacked conv meta (float): 4 layers × [scale[192] + bias[192]]
    #define STACKED_CONV_M_LAYER  (HEAD_FEAT_CH * 2)                        //    384
    #define STACKED_CONV_M_ELEMS  (HEAD_STACKED_CONVS * STACKED_CONV_M_LAYER) //  1 536

    // Pred conv weights (int8): 36 × 192 × 9   (same for cls and reg since both out_ch=36)
    #define PRED_W_ELEMS          (CLS_OUT_CH * HEAD_FEAT_CH * 9)           //  62 208

    // Pred conv scale / bias (float): one value per output channel
    #define PRED_META_ELEMS       CLS_OUT_CH                                //     36

    // ── top function ──────────────────────────────────────────────────────────
    void retina_head_top(
        // input feature map from FPN — CHW [192, TEST_H, TEST_W]
        const act_t    feat_in        [FEAT_ELEMS],

        // outputs: raw logits / deltas  (CHW, no sigmoid / no exp applied)
        act_t          cls_logits     [CLS_LOGITS_ELEMS],
        act_t          reg_deltas     [REG_DELTAS_ELEMS],

        // cls stacked-conv weights
        const weight_t cls_conv_w     [STACKED_CONV_W_ELEMS],
        const meta_t   cls_conv_m     [STACKED_CONV_M_ELEMS],

        // reg stacked-conv weights
        const weight_t reg_conv_w     [STACKED_CONV_W_ELEMS],
        const meta_t   reg_conv_m     [STACKED_CONV_M_ELEMS],

        // cls prediction conv
        const weight_t cls_pred_w     [PRED_W_ELEMS],
        const meta_t   cls_pred_scale [PRED_META_ELEMS],
        const meta_t   cls_pred_bias  [PRED_META_ELEMS],

        // reg prediction conv
        const weight_t reg_pred_w     [PRED_W_ELEMS],
        const meta_t   reg_pred_scale [PRED_META_ELEMS],
        const meta_t   reg_pred_bias  [PRED_META_ELEMS]
    );

    #endif // RETINA_HEAD_TOP_H
