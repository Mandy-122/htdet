/*
 * dump_features.cpp
 * Minimal testbench: runs backbone + FPN only, saves feature maps for
 * comparison with csim_validation/c1_python.bin … p6_python.bin.
 *
 * Usage:
 *   ./dump_features <weights_dir> <image.bin> [output_dir]
 *   ./dump_features ../weights/ ../csim_validation/input_image.bin ../csim_validation
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include "fpga_types.h"
#include "htdet_top.h"

void load_binary(const char* path, float* buf, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.read(reinterpret_cast<char*>(buf), n * sizeof(float));
    std::cout << "  loaded " << n << " floats from " << path << "\n";
}

void save_binary(const char* path, const float* data, int n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot write " << path << "\n"; return; }
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
    float mn = data[0], mx = data[0];
    for (int i = 1; i < n && i < 50000; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    std::cout << "  saved " << n << " floats → " << path
              << "  first5=[" << data[0] << "," << data[1] << ","
              << data[2] << "," << data[3] << "," << data[4] << "]"
              << "  range=[" << mn << "," << mx << "]\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <image.bin> [out_dir]\n";
        return 1;
    }
    std::string wdir(argv[1]);
    const char* img_path = argv[2];
    std::string odir = (argc >= 4) ? argv[3] : ".";

    // ---- Allocate buffers ----
    static float backbone_w[4938784];
    static float image[INPUT_C * INPUT_H * INPUT_W];
    static float c1[C1_CH * C1_H * C1_W];
    static float c2[C2_CH * C2_H * C2_W];
    static float c3[C3_CH * C3_H * C3_W];
    static float c4[C4_CH * C4_H * C4_W];
    static float fpn_w[2600960];
    static float p2[FPN_OUT_CH * P2_H * P2_W];
    static float p3[FPN_OUT_CH * P3_H * P3_W];
    static float p4[FPN_OUT_CH * P4_H * P4_W];
    static float p5[FPN_OUT_CH * P5_H * P5_W];
    static float p6[FPN_OUT_CH * P6_H * P6_W];

    memset(backbone_w, 0, sizeof(backbone_w));
    memset(fpn_w, 0, sizeof(fpn_w));

    std::cout << "Loading weights...\n";
    load_binary((wdir + "/backbone_w.bin").c_str(), backbone_w, 4938784);
    load_binary((wdir + "/fpn_w.bin").c_str(), fpn_w, 2600960);
    load_binary(img_path, image, INPUT_C * INPUT_H * INPUT_W);

    std::cout << "\nRunning backbone...\n";
    htdet_backbone_only(image, backbone_w, c1, c2, c3, c4);

    std::cout << "\nBackbone features:\n";
    save_binary((odir + "/c1_csim.bin").c_str(), c1, C1_CH * C1_H * C1_W);
    save_binary((odir + "/c2_csim.bin").c_str(), c2, C2_CH * C2_H * C2_W);
    save_binary((odir + "/c3_csim.bin").c_str(), c3, C3_CH * C3_H * C3_W);
    save_binary((odir + "/c4_csim.bin").c_str(), c4, C4_CH * C4_H * C4_W);

    std::cout << "\nRunning FPN...\n";
    htdet_fpn_only(c1, c2, c3, c4, fpn_w, p2, p3, p4, p5, p6);

    std::cout << "\nFPN features:\n";
    save_binary((odir + "/p2_csim.bin").c_str(), p2, FPN_OUT_CH * P2_H * P2_W);
    save_binary((odir + "/p3_csim.bin").c_str(), p3, FPN_OUT_CH * P3_H * P3_W);
    save_binary((odir + "/p4_csim.bin").c_str(), p4, FPN_OUT_CH * P4_H * P4_W);
    save_binary((odir + "/p5_csim.bin").c_str(), p5, FPN_OUT_CH * P5_H * P5_W);
    save_binary((odir + "/p6_csim.bin").c_str(), p6, FPN_OUT_CH * P6_H * P6_W);

    std::cout << "\nDone.\n";
    return 0;
}
