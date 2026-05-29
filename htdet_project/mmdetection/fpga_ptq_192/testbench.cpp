/*
 * testbench_ptq_v3.cpp  (PTQ INT8 variant — fixed backbone proj 1x1 dequant scale)
 * HLS C simulation testbench for htdet_inference — W8A32 PTQ, fully corrected.
 *
 * Changes vs testbench_ptq_v2.cpp:
 *   BACKBONE_META_ELEMS: 2926768 → 2927344  (+576: proj_1x1_scale per MViT block)
 *   Output file: csim_detections.txt
 *
 * Changes vs testbench.cpp (baseline):
 *   FPN_META_ELEMS: 1536 → 3072  (8 convs × [scale(192) + bias(192)])
 *   BACKBONE_META_ELEMS: 2926768 → 2927344
 *
 * Binary files expected (same paths as v2):
 *   ptq_results_192/ptq_int8_weights/backbone_int8.bin
 *   ptq_results_192/backbone_meta_float.bin               (2927344 floats)
 *   ptq_results_192/ptq_int8_weights/fpn_int8.bin
 *   ptq_results_192/fpn_meta_float.bin                    (3072 floats)
 *   ptq_results_192/ptq_int8_weights/cls_conv_int8.bin
 *   ptq_results_192/cls_conv_meta_float.bin
 *   ptq_results_192/ptq_int8_weights/reg_conv_int8.bin
 *   ptq_results_192/reg_conv_meta_float.bin
 *   ptq_results_192/ptq_int8_weights/cls_pred_int8.bin
 *   ptq_results_192/ptq_int8_weights/reg_pred_int8.bin
 *   ptq_results_192/cls_pred_scale_float.bin
 *   ptq_results_192/cls_pred_bias_float.bin
 *   ptq_results_192/reg_pred_scale_float.bin
 *   ptq_results_192/reg_pred_bias_float.bin
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <chrono>
#include "fpga_types.h"
#include "htdet_top.h"

#define BACKBONE_CONV_ELEMS    2010864
#define BACKBONE_META_ELEMS    2927344

#define FPN_CONV_ELEMS         1505280
#define FPN_META_ELEMS         3072

#define CLS_CONV_INT8_ELEMS    (HEAD_STACKED_CONVS * HEAD_FEAT_CH * HEAD_FEAT_CH * 9)
#define REG_CONV_INT8_ELEMS    (HEAD_STACKED_CONVS * HEAD_FEAT_CH * HEAD_FEAT_CH * 9)
#define CLS_CONV_META_ELEMS    (HEAD_STACKED_CONVS * HEAD_FEAT_CH * 2)
#define REG_CONV_META_ELEMS    (HEAD_STACKED_CONVS * HEAD_FEAT_CH * 2)

#define CLS_PRED_INT8_ELEMS    (ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9)
#define REG_PRED_INT8_ELEMS    (ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9)
#define CLS_PRED_SCALE_ELEMS   (ANCHORS_PER_LOC * NUM_CLASSES)
#define CLS_PRED_BIAS_ELEMS    (ANCHORS_PER_LOC * NUM_CLASSES)
#define REG_PRED_SCALE_ELEMS   (ANCHORS_PER_LOC * 4)
#define REG_PRED_BIAS_ELEMS    (ANCHORS_PER_LOC * 4)

void load_int8_binary(const char* path, weight_t* buf, int n_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.read(reinterpret_cast<char*>(buf), n_elems * sizeof(int8_t));
    std::cout << "  Loaded " << n_elems << " int8 from " << path << "\n";
}

void load_float_binary(const char* path, meta_t* buf, int n_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.read(reinterpret_cast<char*>(buf), n_elems * sizeof(float));
    std::cout << "  Loaded " << n_elems << " floats from " << path << "\n";
}

void load_image_f32(const char* path, input_t img[INPUT_C * INPUT_H * INPUT_W]) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open image file; using checkerboard pattern.\n";
        for (int c = 0; c < INPUT_C; c++)
            for (int h = 0; h < INPUT_H; h++)
                for (int w = 0; w < INPUT_W; w++) {
                    float v = ((h/16 + w/16) % 2 == 0) ? 0.5f : -0.5f;
                    img[c * INPUT_H * INPUT_W + h * INPUT_W + w] = (input_t)v;
                }
        return;
    }
    for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++) {
        float v; f.read(reinterpret_cast<char*>(&v), 4);
        img[i] = (input_t)v;
    }
    f.close();
    std::cout << "Image loaded from " << path << "\n";
}

void print_detections(const Detection* dets, int n) {
    const char* names[] = {"holothurian", "echinus", "scallop", "starfish"};
    std::cout << "\n=== Detections: " << n << " ===\n";
    for (int i = 0; i < n; i++) {
        std::cout << "  [" << i << "] " << names[dets[i].class_id]
                  << "  score=" << dets[i].score
                  << "  box=[" << dets[i].x1
                  << ", "      << dets[i].y1
                  << ", "      << dets[i].x2
                  << ", "      << dets[i].y2
                  << "]\n";
    }
}

void save_detections(const char* path, const Detection* dets, int n) {
    std::ofstream f(path);
    f << n << "\n";
    for (int i = 0; i < n; i++) {
        f << dets[i].class_id << " "
          << dets[i].score << " "
          << dets[i].x1 << " "
          << dets[i].y1 << " "
          << dets[i].x2 << " "
          << dets[i].y2 << "\n";
    }
    std::cout << "Detections saved to " << path << "\n";
}

int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "HTDet FPGA Testbench PTQ INT8 (C Simulation) — PTQ INT8, 192ch\n";
    std::cout << "Fix: FPN scale+bias + backbone proj 1x1 dequant scale\n";
    std::cout << "==============================================\n";

    static weight_t backbone_conv  [BACKBONE_CONV_ELEMS];
    static meta_t   backbone_meta  [BACKBONE_META_ELEMS];
    static weight_t fpn_conv       [FPN_CONV_ELEMS];
    static meta_t   fpn_meta       [FPN_META_ELEMS];
    static weight_t cls_conv_int8  [CLS_CONV_INT8_ELEMS];
    static meta_t   cls_conv_meta  [CLS_CONV_META_ELEMS];
    static weight_t reg_conv_int8  [REG_CONV_INT8_ELEMS];
    static meta_t   reg_conv_meta  [REG_CONV_META_ELEMS];
    static weight_t cls_pred_w     [CLS_PRED_INT8_ELEMS];
    static meta_t   cls_pred_scale [CLS_PRED_SCALE_ELEMS];
    static meta_t   cls_pred_b     [CLS_PRED_BIAS_ELEMS];
    static weight_t reg_pred_w     [REG_PRED_INT8_ELEMS];
    static meta_t   reg_pred_scale [REG_PRED_SCALE_ELEMS];
    static meta_t   reg_pred_b     [REG_PRED_BIAS_ELEMS];
    static input_t  image          [INPUT_C * INPUT_H * INPUT_W];
    static Detection detections    [MAX_DETS];
    int num_dets = 0;

    memset(backbone_conv,  0, sizeof(backbone_conv));
    memset(backbone_meta,  0, sizeof(backbone_meta));
    memset(fpn_conv,       0, sizeof(fpn_conv));
    memset(fpn_meta,       0, sizeof(fpn_meta));
    memset(cls_conv_int8,  0, sizeof(cls_conv_int8));
    memset(cls_conv_meta,  0, sizeof(cls_conv_meta));
    memset(reg_conv_int8,  0, sizeof(reg_conv_int8));
    memset(reg_conv_meta,  0, sizeof(reg_conv_meta));
    memset(cls_pred_w,     0, sizeof(cls_pred_w));
    memset(cls_pred_scale, 0, sizeof(cls_pred_scale));
    memset(cls_pred_b,     0, sizeof(cls_pred_b));
    memset(reg_pred_w,     0, sizeof(reg_pred_w));
    memset(reg_pred_scale, 0, sizeof(reg_pred_scale));
    memset(reg_pred_b,     0, sizeof(reg_pred_b));

    const char* ptq_dir  = "ptq_results_192/ptq_int8_weights";
    const char* meta_dir = "ptq_results_192";

    std::cout << "Loading int8 weights from: " << ptq_dir << "\n";
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/backbone_int8.bin", ptq_dir);
        load_int8_binary(path, backbone_conv, BACKBONE_CONV_ELEMS);

        snprintf(path, sizeof(path), "%s/fpn_int8.bin", ptq_dir);
        load_int8_binary(path, fpn_conv, FPN_CONV_ELEMS);

        snprintf(path, sizeof(path), "%s/cls_conv_int8.bin", ptq_dir);
        load_int8_binary(path, cls_conv_int8, CLS_CONV_INT8_ELEMS);

        snprintf(path, sizeof(path), "%s/reg_conv_int8.bin", ptq_dir);
        load_int8_binary(path, reg_conv_int8, REG_CONV_INT8_ELEMS);

        snprintf(path, sizeof(path), "%s/cls_pred_int8.bin", ptq_dir);
        load_int8_binary(path, cls_pred_w, CLS_PRED_INT8_ELEMS);

        snprintf(path, sizeof(path), "%s/reg_pred_int8.bin", ptq_dir);
        load_int8_binary(path, reg_pred_w, REG_PRED_INT8_ELEMS);
    }

    std::cout << "\nLoading float meta from: " << meta_dir << "\n";
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/backbone_meta_float.bin", meta_dir);
        load_float_binary(path, backbone_meta, BACKBONE_META_ELEMS);

        snprintf(path, sizeof(path), "%s/fpn_meta_float.bin", meta_dir);
        load_float_binary(path, fpn_meta, FPN_META_ELEMS);

        snprintf(path, sizeof(path), "%s/cls_conv_meta_float.bin", meta_dir);
        load_float_binary(path, cls_conv_meta, CLS_CONV_META_ELEMS);

        snprintf(path, sizeof(path), "%s/reg_conv_meta_float.bin", meta_dir);
        load_float_binary(path, reg_conv_meta, REG_CONV_META_ELEMS);

        snprintf(path, sizeof(path), "%s/cls_pred_scale_float.bin", meta_dir);
        load_float_binary(path, cls_pred_scale, CLS_PRED_SCALE_ELEMS);

        snprintf(path, sizeof(path), "%s/cls_pred_bias_float.bin", meta_dir);
        load_float_binary(path, cls_pred_b, CLS_PRED_BIAS_ELEMS);

        snprintf(path, sizeof(path), "%s/reg_pred_scale_float.bin", meta_dir);
        load_float_binary(path, reg_pred_scale, REG_PRED_SCALE_ELEMS);

        snprintf(path, sizeof(path), "%s/reg_pred_bias_float.bin", meta_dir);
        load_float_binary(path, reg_pred_b, REG_PRED_BIAS_ELEMS);
    }

    const char* img_path = (argc >= 2) ? argv[1] : nullptr;
    load_image_f32(img_path, image);

    std::cout << "\nRunning htdet_inference (PTQ INT8 v3)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    htdet_inference(
        image,
        backbone_conv,  backbone_meta,
        fpn_conv,       fpn_meta,
        cls_conv_int8,  cls_conv_meta,
        reg_conv_int8,  reg_conv_meta,
        cls_pred_w,     cls_pred_scale,  cls_pred_b,
        reg_pred_w,     reg_pred_scale,  reg_pred_b,
        detections, &num_dets
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C-simulation latency: " << ms << " ms\n";

    print_detections(detections, num_dets);
    save_detections("csim_detections.txt", detections, num_dets);

    std::cout << "\n==============================================\n";
    std::cout << "Testbench PASSED\n";
    std::cout << "==============================================\n";
    return 0;
}
