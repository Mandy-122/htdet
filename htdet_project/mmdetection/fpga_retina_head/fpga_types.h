    /*
    * fpga_types.h  (retina_head isolated module)
    *
    * Configurable spatial test size — change TEST_H / TEST_W to step through levels:
    *
    *   Phase 1 (smallest): P6   TEST_H=10   TEST_W=10   stride=64
    *   Phase 2:            P5   TEST_H=20   TEST_W=20   stride=32
    *   Phase 3:            P4   TEST_H=40   TEST_W=40   stride=16
    *   Phase 4:            P3   TEST_H=80   TEST_W=80   stride=8
    *   Phase 5 (largest):  P2   TEST_H=160  TEST_W=160  stride=4
    *
    * Only TEST_H / TEST_W affects array sizes; HEAD_FEAT_CH=192 is always full.
    * Override from TCL: add_files ... -cflags "-DTEST_H=20 -DTEST_W=20 ..."
    *
    * Data types: W8A32 (int8 weights, float32 activations/accumulators)
    */

    #ifndef FPGA_TYPES_H
    #define FPGA_TYPES_H

    #include <ap_int.h>
    #include <ap_fixed.h>
    #include <hls_stream.h>
    #include <cstdint>

    typedef int8_t  weight_t;    // INT8 quantised conv kernel weights
    typedef float   meta_t;      // float: BN eff-scale, bias, dequant-scale
    typedef float   bias_t;
    typedef float   act_t;       // float32 activations (W8A32)
    typedef float   acc_t;       // float32 accumulators
    typedef float   bbox_t;
    typedef float   score_t;

    // Image bounds (for bbox clamping in score-filter stage)
    #define INPUT_H   640
    #define INPUT_W   640

    // FPN / head channel width — fixed by the trained model
    #define FPN_OUT_CH    192
    #define HEAD_FEAT_CH  192

    // RetinaNet head architecture
    #define HEAD_STACKED_CONVS   4
    #define NUM_CLASSES          4          // holothurian, echinus, scallop, starfish
    #define ANCHOR_SCALES_N      3          // {4, 6, 8}
    #define ANCHOR_RATIOS_N      3          // {0.5, 1.0, 2.0}
    #define ANCHORS_PER_LOC      (ANCHOR_SCALES_N * ANCHOR_RATIOS_N)   // 9

    #define SCORE_THR     0.20f
    #define NMS_IOU_THR   0.5f

    // ── TEST SPATIAL SIZE ──────────────────────────────────────────────────────
    // P6 = 10×10 is the default (smallest, fastest CSIM).
    // Override with -DTEST_H=20 -DTEST_W=20 for P5, etc.
    #ifndef TEST_H
    #define TEST_H  10
    #endif
    #ifndef TEST_W
    #define TEST_W  10
    #endif

    // ── UTILITY ────────────────────────────────────────────────────────────────
    #define DIV_CEIL(a,b)  (((a)+(b)-1)/(b))

    #endif // FPGA_TYPES_H
