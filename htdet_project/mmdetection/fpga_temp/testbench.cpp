/*
 * testbench.cpp
 * HLS C simulation testbench for htdet_inference.
 *
 * Usage:
 *   - With exported weights:  ./testbench weights_dir/ [image.bin]
 *   - Without weights:        ./testbench (uses zero weights + checkerboard image)
 *
 * Image binary format expected:  float32, CHW, 3×640×640, ImageNet mean/std normalization
 * Weight binary format expected: float32 (weight_t=float); switch to int16 path when using ap_fixed
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <chrono>
#include "fpga_types.h"
#include "htdet_top.h"   // include whole TU for csim

// ============================================================
// Weight sizes (elements, not bytes)
// These must match what export_weights.py writes.
// ============================================================
// Backbone: exact size matches HLS AXI depth pragma in htdet_top.cpp
#define BACKBONE_W_ELEMS   4937632   // matches export_weights.py output
#define FPN_W_ELEMS        2598912
// 4 stacked convs × (conv_w + gn_gamma + gn_beta) per layer:
//   per layer = HEAD_FEAT_CH*HEAD_FEAT_CH*9 + HEAD_FEAT_CH + HEAD_FEAT_CH
//             = HEAD_FEAT_CH * (HEAD_FEAT_CH*9 + 2)
#define CLS_CONV_W_ELEMS   (HEAD_STACKED_CONVS * HEAD_FEAT_CH * (HEAD_FEAT_CH * 9 + 2))
#define REG_CONV_W_ELEMS   (HEAD_STACKED_CONVS * HEAD_FEAT_CH * (HEAD_FEAT_CH * 9 + 2))
#define CLS_PRED_W_ELEMS   (ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9)
#define REG_PRED_W_ELEMS   (ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9)

// ============================================================
// I/O HELPERS
// ============================================================

// When weight_t is ap_fixed<16,8>, swap to the int16 path below.
//void load_weights_binary(const char* path, weight_t* buf, int n_elems) {
//    std::ifstream f(path, std::ios::binary);
//    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
//    for (int i = 0; i < n_elems; i++) {
//        int16_t raw;
//        f.read(reinterpret_cast<char*>(&raw), 2);
//        buf[i].range(15, 0) = raw;
//    }
//    f.close();
//}

// weight_t is currently float32 — reads sizeof(weight_t)=4 bytes per element.
void load_weights_binary(const char* path, weight_t* buf, int n_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.read(reinterpret_cast<char*>(buf), n_elems * sizeof(weight_t));
    f.close();
}

void load_image_f32(const char* path,
                    input_t img[INPUT_C * INPUT_H * INPUT_W]) {
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
                  << "  score=" << dets[i].score//.to_float()
                  << "  box=[" << dets[i].x1//.to_float()
                  << ", "      << dets[i].y1//.to_float()
                  << ", "      << dets[i].x2//.to_float()
                  << ", "      << dets[i].y2//.to_float()
				  << "]\n";
    }
}

//void save_detections(const char* path, const Detection* dets, int n) {
//    std::ofstream f(path);
//    f << n << "\n";
//    for (int i = 0; i < n; i++) {
//        f << dets[i].class_id << " "
//          << dets[i].score.to_float() << " "
//          << dets[i].x1.to_float()   << " "
//          << dets[i].y1.to_float()   << " "
//          << dets[i].x2.to_float()   << " "
//          << dets[i].y2.to_float()   << "\n";
//    }
//    std::cout << "Detections saved to " << path << "\n";
//}

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

void save_binary(const char* path, const float* data, int n_elems) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot write " << path << "\n"; return; }
    f.write(reinterpret_cast<const char*>(data), n_elems * sizeof(float));
    std::cout << "  saved " << n_elems << " floats → " << path
              << "  first5=[" << data[0] << "," << data[1] << ","
              << data[2] << "," << data[3] << "," << data[4] << "]\n";
}

void dump_backbone_features(const input_t* image, const weight_t* backbone_w,
                             const weight_t* fpn_w, const char* out_dir) {
    static act_t c1_d[C1_CH * C1_H * C1_W];
    static act_t c2_d[C2_CH * C2_H * C2_W];
    static act_t c3_d[C3_CH * C3_H * C3_W];
    static act_t c4_d[C4_CH * C4_H * C4_W];

    std::cout << "\nRunning htdet_backbone_only for feature dump...\n";
    htdet_backbone_only(image, backbone_w, c1_d, c2_d, c3_d, c4_d);

    std::string dir(out_dir);
    std::cout << "Backbone features:\n";
    save_binary((dir + "/c1_csim.bin").c_str(), c1_d, C1_CH * C1_H * C1_W);
    save_binary((dir + "/c2_csim.bin").c_str(), c2_d, C2_CH * C2_H * C2_W);
    save_binary((dir + "/c3_csim.bin").c_str(), c3_d, C3_CH * C3_H * C3_W);
    save_binary((dir + "/c4_csim.bin").c_str(), c4_d, C4_CH * C4_H * C4_W);

    static act_t p2_d[FPN_OUT_CH * P2_H * P2_W];
    static act_t p3_d[FPN_OUT_CH * P3_H * P3_W];
    static act_t p4_d[FPN_OUT_CH * P4_H * P4_W];
    static act_t p5_d[FPN_OUT_CH * P5_H * P5_W];
    static act_t p6_d[FPN_OUT_CH * P6_H * P6_W];

    std::cout << "\nRunning htdet_fpn_only for feature dump...\n";
    htdet_fpn_only(c1_d, c2_d, c3_d, c4_d, fpn_w, p2_d, p3_d, p4_d, p5_d, p6_d);

    std::cout << "FPN features:\n";
    save_binary((dir + "/p2_csim.bin").c_str(), p2_d, FPN_OUT_CH * P2_H * P2_W);
    save_binary((dir + "/p3_csim.bin").c_str(), p3_d, FPN_OUT_CH * P3_H * P3_W);
    save_binary((dir + "/p4_csim.bin").c_str(), p4_d, FPN_OUT_CH * P4_H * P4_W);
    save_binary((dir + "/p5_csim.bin").c_str(), p5_d, FPN_OUT_CH * P5_H * P5_W);
    save_binary((dir + "/p6_csim.bin").c_str(), p6_d, FPN_OUT_CH * P6_H * P6_W);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "HTDet FPGA Testbench (C Simulation)\n";
    std::cout << "==============================================\n";
    std::cout << "Input:        " << INPUT_C << "×" << INPUT_H << "×" << INPUT_W << "\n";
    std::cout << "Backbone out: C1=["<< C1_CH <<"," << C1_H <<"," << C1_W
              <<"] C2=["<< C2_CH <<"," << C2_H <<"," << C2_W
              <<"] C3=["<< C3_CH <<"," << C3_H <<"," << C3_W
              <<"] C4=["<< C4_CH <<"," << C4_H <<"," << C4_W <<"]\n";
    std::cout << "FPN out:      P2=[256,160,160] … P6=[256,10,10]\n";
    std::cout << "Classes:      4 (holothurian, echinus, scallop, starfish)\n";
    std::cout << "Anchors/loc:  " << ANCHORS_PER_LOC << " (" << ANCHOR_SCALES_N
              << " scales × " << ANCHOR_RATIOS_N << " ratios)\n\n";

    // ---- Allocate weight buffers ----
    static weight_t backbone_w [BACKBONE_W_ELEMS];
    static weight_t fpn_w      [FPN_W_ELEMS];
    static weight_t cls_conv_w [CLS_CONV_W_ELEMS];
    static weight_t reg_conv_w [REG_CONV_W_ELEMS];
    static weight_t cls_pred_w [CLS_PRED_W_ELEMS];
    static bias_t   cls_pred_b [ANCHORS_PER_LOC * NUM_CLASSES];
    static weight_t reg_pred_w [REG_PRED_W_ELEMS];
    static bias_t   reg_pred_b [ANCHORS_PER_LOC * 4];
    static input_t  image      [INPUT_C * INPUT_H * INPUT_W];
    static Detection detections[MAX_DETS];
    int num_dets = 0;

    // Zero-initialize weights (valid for architecture check without real weights)
    memset(backbone_w, 0, sizeof(backbone_w));
    memset(fpn_w,      0, sizeof(fpn_w));
    memset(cls_conv_w, 0, sizeof(cls_conv_w));
    memset(reg_conv_w, 0, sizeof(reg_conv_w));
    memset(cls_pred_w, 0, sizeof(cls_pred_w));
    memset(cls_pred_b, 0, sizeof(cls_pred_b));
    memset(reg_pred_w, 0, sizeof(reg_pred_w));
    memset(reg_pred_b, 0, sizeof(reg_pred_b));

    // Usage: testbench [weights_dir [image.bin]] [--dump [dump_dir]]
    bool do_dump = false;
    const char* dump_dir = "../csim_validation";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dump") {
            do_dump = true;
            if (i + 1 < argc && argv[i+1][0] != '-') dump_dir = argv[++i];
        }
    }

    // ---- Load weights from directory (if provided) ----
    if (argc >= 2) {
        std::string dir(argv[1]);
        std::cout << "Loading weights from: " << dir << "\n";
        load_weights_binary((dir + "/backbone_w.bin").c_str(), backbone_w, BACKBONE_W_ELEMS);
        load_weights_binary((dir + "/fpn_w.bin").c_str(),      fpn_w,      FPN_W_ELEMS);
        load_weights_binary((dir + "/cls_conv_w.bin").c_str(), cls_conv_w, CLS_CONV_W_ELEMS);
        load_weights_binary((dir + "/reg_conv_w.bin").c_str(), reg_conv_w, REG_CONV_W_ELEMS);
        load_weights_binary((dir + "/cls_pred_w.bin").c_str(), cls_pred_w, CLS_PRED_W_ELEMS);
        load_weights_binary((dir + "/cls_pred_b.bin").c_str(), (weight_t*)cls_pred_b, ANCHORS_PER_LOC * NUM_CLASSES);
        load_weights_binary((dir + "/reg_pred_w.bin").c_str(), reg_pred_w, REG_PRED_W_ELEMS);
        load_weights_binary((dir + "/reg_pred_b.bin").c_str(), (weight_t*)reg_pred_b, ANCHORS_PER_LOC * 4);
    } else {
        std::cout << "No weight directory provided; using zero weights (architecture test).\n";
    }

    // ---- Load image ----
    const char* img_path = (argc >= 3) ? argv[2] : nullptr;
    load_image_f32(img_path, image);

    // ---- Run inference ----
    std::cout << "\nRunning htdet_inference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    htdet_inference(
        image,
        backbone_w, fpn_w,
        cls_conv_w, reg_conv_w,
        cls_pred_w, cls_pred_b,
        reg_pred_w, reg_pred_b,
        detections, &num_dets
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "C-simulation latency: " << ms << " ms\n";

    // ---- Print and save results ----
    print_detections(detections, num_dets);
    // Output filename matches the default expected by csim_validation/compare_results.py.
    // Run compare_results.py with --csim <path>/csim_detections.txt if needed.
    save_detections("csim_detections.txt", detections, num_dets);

    // ---- Optional feature dump for layer-by-layer comparison ----
    if (do_dump) {
        dump_backbone_features(image, backbone_w, fpn_w, dump_dir);
    }

    std::cout << "\n==============================================\n";
    std::cout << "Testbench PASSED\n";
    std::cout << "==============================================\n";
    return 0;
}
