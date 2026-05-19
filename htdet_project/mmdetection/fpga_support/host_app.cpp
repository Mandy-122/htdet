/*
 * host_app.cpp
 * Zynq UltraScale+ PS-side host application for HTDet FPGA inference.
 *
 * Build (on-board or cross-compile):
 *   g++ -O2 -std=c++14 host_app.cpp -o htdet_host \
 *       -lxrt_coreutil -luuid -pthread \
 *       -I${XILINX_XRT}/include -L${XILINX_XRT}/lib
 *
 * Usage:
 *   ./htdet_host --xclbin htdet.xclbin \
 *                --weights ./weights \
 *                --image   image.bin  \
 *                [--thresh 0.3]       \
 *                [--out    results.txt]
 *
 * The xclbin is produced by Vitis (v++ linker) after HLS synthesis.
 * The weights directory is produced by export_weights.py.
 * The image binary is CHW float32, 3×320×320, normalised to [-1, 1].
 *
 * Detection output format (results.txt):
 *   <num_dets>
 *   <class_id> <score> <x1> <y1> <x2> <y2>
 *   ...
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>

// XRT headers (Vitis 2022.x and later)
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include "weights_loader.h"

// ============================================================
// Model constants (must match fpga_types.h)
// ============================================================
static const int INPUT_C   = 3;
static const int INPUT_H   = 320;
static const int INPUT_W   = 320;
static const int MAX_DETS  = 100;

// ============================================================
// Detection struct (on-host representation)
// ============================================================
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

// ============================================================
// Argument parsing
// ============================================================
struct Args {
    std::string xclbin   = "htdet.xclbin";
    std::string weights  = "./weights";
    std::string image    = "";
    std::string out      = "results.txt";
    float       thresh   = 0.05f;
    int         device   = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s(argv[i]);
        if (s == "--xclbin"  && i+1 < argc) { a.xclbin  = argv[++i]; }
        else if (s == "--weights" && i+1 < argc) { a.weights = argv[++i]; }
        else if (s == "--image"   && i+1 < argc) { a.image   = argv[++i]; }
        else if (s == "--out"     && i+1 < argc) { a.out     = argv[++i]; }
        else if (s == "--thresh"  && i+1 < argc) { a.thresh  = std::stof(argv[++i]); }
        else if (s == "--device"  && i+1 < argc) { a.device  = std::stoi(argv[++i]); }
        else if (s == "--help" || s == "-h") {
            std::cout
                << "Usage: htdet_host [options]\n"
                << "  --xclbin  <path>   FPGA bitstream (.xclbin)  [htdet.xclbin]\n"
                << "  --weights <dir>    Weight binary directory    [./weights]\n"
                << "  --image   <path>   Input image (.bin float32) [checkerboard]\n"
                << "  --out     <path>   Output detection file      [results.txt]\n"
                << "  --thresh  <float>  Score threshold            [0.05]\n"
                << "  --device  <int>    XRT device index           [0]\n";
            exit(0);
        }
    }
    return a;
}

// ============================================================
// Print and save detections
// ============================================================
static const char* CLASS_NAMES[] = {"holothurian", "echinus", "scallop", "starfish"};

static void print_detections(const Detection* dets, int n) {
    std::cout << "\n=== Detections: " << n << " ===\n";
    for (int i = 0; i < n; i++) {
        const Detection& d = dets[i];
        std::cout << "  [" << i << "] " << CLASS_NAMES[d.class_id]
                  << "  score=" << d.score
                  << "  box=[" << d.x1 << ", " << d.y1
                  << ", "      << d.x2 << ", " << d.y2 << "]\n";
    }
}

static void save_detections(const char* path, const Detection* dets, int n) {
    std::ofstream f(path);
    f << n << "\n";
    for (int i = 0; i < n; i++) {
        f << dets[i].class_id << " " << dets[i].score << " "
          << dets[i].x1 << " " << dets[i].y1 << " "
          << dets[i].x2 << " " << dets[i].y2 << "\n";
    }
    std::cout << "Detections saved to " << path << "\n";
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::cout << "==============================================\n";
    std::cout << "HTDet FPGA Host Application\n";
    std::cout << "==============================================\n";
    std::cout << "xclbin : " << args.xclbin  << "\n";
    std::cout << "weights: " << args.weights << "\n";
    std::cout << "image  : " << (args.image.empty() ? "(checkerboard)" : args.image) << "\n";
    std::cout << "thresh : " << args.thresh  << "\n";

    // --------------------------------------------------------
    // 1. Load weights from disk into host memory
    // --------------------------------------------------------
    WeightBuffers wb;
    if (weights_load_all(args.weights.c_str(), &wb) != 0) {
        std::cerr << "Failed to load weights from " << args.weights << "\n";
        return 1;
    }

    // --------------------------------------------------------
    // 2. Load image
    // --------------------------------------------------------
    const int img_n = INPUT_C * INPUT_H * INPUT_W;
    std::vector<int16_t> image_q(img_n, 0);

    if (!args.image.empty()) {
        float* img_f = load_image_bin(args.image.c_str(), INPUT_C, INPUT_H, INPUT_W);
        if (img_f) {
            image_f32_to_q8_8(img_f, image_q.data(), img_n);
            free(img_f);
            std::cout << "Image loaded and quantised to Q8.8.\n";
        }
    } else {
        std::cout << "No image provided; using zero-filled input.\n";
    }

    // --------------------------------------------------------
    // 3. Open XRT device and load xclbin
    // --------------------------------------------------------
    std::cout << "\nOpening FPGA device " << args.device << "...\n";
    xrt::device device(args.device);

    std::cout << "Loading xclbin: " << args.xclbin << "...\n";
    xrt::uuid uuid = device.load_xclbin(args.xclbin);

    // Kernel name must match the top function set in Vitis HLS.
    xrt::kernel kernel(device, uuid, "htdet_inference");
    std::cout << "Kernel loaded.\n\n";

    // --------------------------------------------------------
    // 4. Allocate XRT buffer objects in FPGA DDR
    //    Group number must match bundle index in HLS pragmas:
    //      gmem0=image, gmem1=backbone, gmem2=fpn,
    //      gmem3=cls_conv, gmem4=reg_conv, gmem5=pred_w/b,
    //      gmem6=detections, gmem7=num_dets
    //
    //    For Zynq (embedded), all share one DDR bank.  On PCIe
    //    cards use XCL_MEM_DDR_BANK0 .. BANK3 to route bundles.
    // --------------------------------------------------------

    auto grp = [](int g){ return xrt::bo::flags::normal; }; // Zynq: all same bank

    xrt::bo bo_image    (device, img_n                    * sizeof(int16_t), grp(0), kernel.group_id(0));
    xrt::bo bo_backbone (device, wb.backbone_n            * sizeof(int16_t), grp(0), kernel.group_id(1));
    xrt::bo bo_fpn      (device, wb.fpn_n                 * sizeof(int16_t), grp(0), kernel.group_id(2));
    xrt::bo bo_cls_conv (device, wb.cls_conv_n            * sizeof(int16_t), grp(0), kernel.group_id(3));
    xrt::bo bo_reg_conv (device, wb.reg_conv_n            * sizeof(int16_t), grp(0), kernel.group_id(4));
    xrt::bo bo_cls_pw   (device, wb.cls_pred_w_n          * sizeof(int16_t), grp(0), kernel.group_id(5));
    xrt::bo bo_cls_pb   (device, wb.cls_pred_b_n          * sizeof(int16_t), grp(0), kernel.group_id(5));
    xrt::bo bo_reg_pw   (device, wb.reg_pred_w_n          * sizeof(int16_t), grp(0), kernel.group_id(5));
    xrt::bo bo_reg_pb   (device, wb.reg_pred_b_n          * sizeof(int16_t), grp(0), kernel.group_id(5));
    xrt::bo bo_dets     (device, MAX_DETS * 6             * sizeof(float),   grp(0), kernel.group_id(9));
    xrt::bo bo_num_dets (device,                          sizeof(int),       grp(0), kernel.group_id(10));

    // --------------------------------------------------------
    // 5. Copy weight data into FPGA DDR
    // --------------------------------------------------------
    std::cout << "Copying weights to FPGA DDR...\n";
    auto t_cp0 = std::chrono::high_resolution_clock::now();

    std::memcpy(bo_image.map<int16_t*>(),   image_q.data(),    img_n          * sizeof(int16_t));
    std::memcpy(bo_backbone.map<int16_t*>(),wb.backbone_w,     wb.backbone_n  * sizeof(int16_t));
    std::memcpy(bo_fpn.map<int16_t*>(),     wb.fpn_w,          wb.fpn_n       * sizeof(int16_t));
    std::memcpy(bo_cls_conv.map<int16_t*>(),wb.cls_conv_w,     wb.cls_conv_n  * sizeof(int16_t));
    std::memcpy(bo_reg_conv.map<int16_t*>(),wb.reg_conv_w,     wb.reg_conv_n  * sizeof(int16_t));
    std::memcpy(bo_cls_pw.map<int16_t*>(),  wb.cls_pred_w,     wb.cls_pred_w_n* sizeof(int16_t));
    std::memcpy(bo_cls_pb.map<int16_t*>(),  wb.cls_pred_b,     wb.cls_pred_b_n* sizeof(int16_t));
    std::memcpy(bo_reg_pw.map<int16_t*>(),  wb.reg_pred_w,     wb.reg_pred_w_n* sizeof(int16_t));
    std::memcpy(bo_reg_pb.map<int16_t*>(),  wb.reg_pred_b,     wb.reg_pred_b_n* sizeof(int16_t));

    bo_image.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_backbone.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fpn.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_cls_conv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_reg_conv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_cls_pw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_cls_pb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_reg_pw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_reg_pb.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto t_cp1 = std::chrono::high_resolution_clock::now();
    double cp_ms = std::chrono::duration<double, std::milli>(t_cp1 - t_cp0).count();
    std::cout << "DDR copy time: " << cp_ms << " ms\n\n";

    // --------------------------------------------------------
    // 6. Run kernel
    //    Argument order matches htdet_inference signature in htdet_top.cpp
    // --------------------------------------------------------
    std::cout << "Launching htdet_inference kernel...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto run = kernel(
        bo_image,    // arg 0: image
        bo_backbone, // arg 1: backbone_w
        bo_fpn,      // arg 2: fpn_w
        bo_cls_conv, // arg 3: cls_conv_w
        bo_reg_conv, // arg 4: reg_conv_w
        bo_cls_pw,   // arg 5: cls_pred_w
        bo_cls_pb,   // arg 6: cls_pred_b
        bo_reg_pw,   // arg 7: reg_pred_w
        bo_reg_pb,   // arg 8: reg_pred_b
        bo_dets,     // arg 9: detections
        bo_num_dets  // arg 10: num_dets
    );
    run.wait();

    auto t1 = std::chrono::high_resolution_clock::now();
    double inf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Kernel execution time: " << inf_ms << " ms\n\n";

    // --------------------------------------------------------
    // 7. Read back results
    // --------------------------------------------------------
    bo_dets.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_num_dets.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int num_dets = *bo_num_dets.map<int*>();
    if (num_dets < 0) num_dets = 0;
    if (num_dets > MAX_DETS) num_dets = MAX_DETS;

    // Detections are packed as float32: [x1, y1, x2, y2, score, class_id_as_float]
    float* raw = bo_dets.map<float*>();
    std::vector<Detection> dets(num_dets);
    for (int i = 0; i < num_dets; i++) {
        dets[i].x1       = raw[i*6 + 0];
        dets[i].y1       = raw[i*6 + 1];
        dets[i].x2       = raw[i*6 + 2];
        dets[i].y2       = raw[i*6 + 3];
        dets[i].score    = raw[i*6 + 4];
        dets[i].class_id = (int)raw[i*6 + 5];
    }

    // Apply host-side score threshold (FPGA uses 0.05 internally)
    std::vector<Detection> filtered;
    for (auto& d : dets) {
        if (d.score >= args.thresh) filtered.push_back(d);
    }

    print_detections(filtered.data(), (int)filtered.size());
    save_detections(args.out.c_str(), filtered.data(), (int)filtered.size());

    // --------------------------------------------------------
    // 8. Cleanup
    // --------------------------------------------------------
    weights_free(&wb);

    std::cout << "\n==============================================\n";
    std::cout << "Total time (DDR copy + kernel): " << (cp_ms + inf_ms) << " ms\n";
    std::cout << "==============================================\n";
    return 0;
}
