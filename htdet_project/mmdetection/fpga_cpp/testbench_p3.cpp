/*
 * testbench_p3.cpp
 * Minimal head test: loads P3 features + head weights, runs process_level
 * for P3 only, saves reg_feat/reg_deltas/cls_feat/cls_logits for comparison
 * with Python reference (csim_validation/dump_head_p3.py outputs).
 *
 * Build:
 *   g++ -O2 -std=c++14 -I. -I./hls_stubs -DDEBUG_HEAD_DUMP=1 testbench_p3.cpp -lm -o testbench_p3
 *
 * Run:
 *   ./testbench_p3 ../weights/ ../csim_validation/p3_csim.bin
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "fpga_types.h"
#include "fpga_utils.h"

// Include the full head implementation
#define DEBUG_HEAD_DUMP 1
#include "retina_head.h"

#define CLS_CONV_W_ELEMS   (HEAD_STACKED_CONVS * HEAD_FEAT_CH * (HEAD_FEAT_CH * 9 + 2))
#define REG_CONV_W_ELEMS   (HEAD_STACKED_CONVS * HEAD_FEAT_CH * (HEAD_FEAT_CH * 9 + 2))
#define CLS_PRED_W_ELEMS   (ANCHORS_PER_LOC * NUM_CLASSES * HEAD_FEAT_CH * 9)
#define REG_PRED_W_ELEMS   (ANCHORS_PER_LOC * 4           * HEAD_FEAT_CH * 9)

void load_binary(const char* path, float* buf, int n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return; }
    f.read(reinterpret_cast<char*>(buf), n * sizeof(float));
    std::cout << "  loaded " << n << " floats from " << path << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <p3_csim.bin>\n";
        return 1;
    }
    std::string wdir(argv[1]);
    const char* p3_path = argv[2];

    // ---- Allocate buffers ----
    static weight_t cls_conv_w[CLS_CONV_W_ELEMS];
    static weight_t reg_conv_w[REG_CONV_W_ELEMS];
    static weight_t cls_pred_w[CLS_PRED_W_ELEMS];
    static bias_t   cls_pred_b[ANCHORS_PER_LOC * NUM_CLASSES];
    static weight_t reg_pred_w[REG_PRED_W_ELEMS];
    static bias_t   reg_pred_b[ANCHORS_PER_LOC * 4];
    static act_t    p3[FPN_OUT_CH * P3_H * P3_W];
    static bbox_t   cand_boxes[CAND_BUF_SIZE * 4];
    static score_t  cand_scores[CAND_BUF_SIZE];
    static int      cand_cls[CAND_BUF_SIZE];

    memset(cls_conv_w, 0, sizeof(cls_conv_w));
    memset(reg_conv_w, 0, sizeof(reg_conv_w));
    memset(cls_pred_w, 0, sizeof(cls_pred_w));
    memset(cls_pred_b, 0, sizeof(cls_pred_b));
    memset(reg_pred_w, 0, sizeof(reg_pred_w));
    memset(reg_pred_b, 0, sizeof(reg_pred_b));

    std::cout << "Loading weights...\n";
    load_binary((wdir + "/cls_conv_w.bin").c_str(), cls_conv_w, CLS_CONV_W_ELEMS);
    load_binary((wdir + "/reg_conv_w.bin").c_str(), reg_conv_w, REG_CONV_W_ELEMS);
    load_binary((wdir + "/cls_pred_w.bin").c_str(), cls_pred_w, CLS_PRED_W_ELEMS);
    load_binary((wdir + "/cls_pred_b.bin").c_str(), cls_pred_b, ANCHORS_PER_LOC * NUM_CLASSES);
    load_binary((wdir + "/reg_pred_w.bin").c_str(), reg_pred_w, REG_PRED_W_ELEMS);
    load_binary((wdir + "/reg_pred_b.bin").c_str(), reg_pred_b, ANCHORS_PER_LOC * 4);

    std::cout << "Loading P3 features...\n";
    load_binary(p3_path, p3, FPN_OUT_CH * P3_H * P3_W);

    // Print P3 input stats
    float mn=p3[0], mx=p3[0], sm=0;
    for (int i = 0; i < FPN_OUT_CH*P3_H*P3_W; i++) { sm+=p3[i]; if(p3[i]<mn)mn=p3[i]; if(p3[i]>mx)mx=p3[i]; }
    std::cout << "P3 input: min=" << mn << " max=" << mx << " mean=" << sm/(FPN_OUT_CH*P3_H*P3_W) << "\n";
    std::cout << "P3 first5=[" << p3[0]<<"," <<p3[1]<<","<<p3[2]<<","<<p3[3]<<","<<p3[4]<<"]\n";

    // Print reg_pred_b (the bias of the final reg pred conv)
    std::cout << "\nreg_pred_b first9: [";
    for (int i = 0; i < 9; i++) std::cout << reg_pred_b[i] << (i<8?",":"");
    std::cout << "]\n";
    std::cout << "cls_pred_b first9: [";
    for (int i = 0; i < 9; i++) std::cout << cls_pred_b[i] << (i<8?",":"");
    std::cout << "]\n";

    // Print a few reg_conv_w values to verify loading
    std::cout << "reg_conv_w first5=[" << reg_conv_w[0]<<","<<reg_conv_w[1]<<","<<reg_conv_w[2]<<","<<reg_conv_w[3]<<","<<reg_conv_w[4]<<"]\n";

    std::cout << "\nRunning process_level for P3 (stride=8, H=80, W=80)...\n";
    int nc = process_level(
        p3,
        cls_conv_w, reg_conv_w,
        cls_pred_w, cls_pred_b,
        reg_pred_w, reg_pred_b,
        P3_H, P3_W, 8,
        cand_boxes, cand_scores, cand_cls,
        CAND_BUF_SIZE
    );

    std::cout << "\nP3 candidates above score_thr=0.05: " << nc << "\n";

    // Print first 5 candidates
    const char* names[] = {"holothurian", "echinus", "scallop", "starfish"};
    for (int i = 0; i < std::min(nc, 5); i++) {
        std::cout << "  cand[" << i << "] " << names[cand_cls[i]]
                  << " score=" << cand_scores[i]
                  << " box=[" << cand_boxes[i*4+0] << "," << cand_boxes[i*4+1]
                  << "," << cand_boxes[i*4+2] << "," << cand_boxes[i*4+3] << "]\n";
    }

    std::cout << "\nDone. Check ../csim_validation/ for dumped binary files.\n";
    return 0;
}
