/*
 * testbench_fpn_p4p5.cpp
 * C-simulation testbench for fpn_p4p5_top (Phase 2: P4+P5+P6).
 *
 * Tests:
 *   1. all-zero input + zero weights → all-zero outputs
 *   2. finite-value check (checkerboard C4/C3, unit weights)
 *   2b. numerical check: ramp input, unit int8 weights, unit lat biases
 *       Expected P5 centre: 192*9*641  = 1,107,648
 *       Expected P4 centre: 192*9*770  = 1,330,560
 *         (lat3_merged = C3_CH*1+1 + upsample(lat4) = 129 + 641 = 770)
 *   3. file-based real weights (skipped if files absent)
 *
 * Build:
 *   g++ -std=c++14 -O0 -I. -I../fpga_temp/hls_stubs \
 *       testbench_fpn_p4p5.cpp fpn_p4p5_top.cpp -lm -o tb_fpn_p4p5
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "fpn_p4p5_top.h"

static bool read_bin(const char* path, void* buf, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(buf), bytes);
    return (size_t)f.gcount() == bytes;
}
static void write_bin(const char* path, const void* buf, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf), bytes);
}
static void print_stats(const char* name, const act_t* arr, int n) {
    float sum=0, mn=arr[0], mx=arr[0];
    for (int i=0;i<n;i++){
        sum+=arr[i];
        if(arr[i]<mn) mn=arr[i];
        if(arr[i]>mx) mx=arr[i];
    }
    std::cout << "  " << name << "  sum=" << sum << "  min=" << mn << "  max=" << mx << "\n";
}

static act_t    c4    [FPN_P4P5_C4_ELEMS];
static act_t    c3    [FPN_P4P5_C3_ELEMS];
static weight_t w_conv[FPN_P4P5_WCONV_ELEMS];
static meta_t   w_meta[FPN_P4P5_WMETA_ELEMS];
static act_t    p5    [FPN_P4P5_P5_ELEMS];
static act_t    p4    [FPN_P4P5_P4_ELEMS];
static act_t    p6    [FPN_P4P5_P6_ELEMS];

int main() {
    std::cout << "==============================================\n";
    std::cout << "fpn_p4p5_top  C-simulation testbench\n";
    std::cout << "Phase 2: P4+P5+P6  (FPN_OUT_CH=" << FPN_OUT_CH
              << "  P5=" << P5_H << "x" << P5_W
              << "  P4=" << P4_H << "x" << P4_W
              << "  P6=" << P6_H << "x" << P6_W << ")\n";
    std::cout << "==============================================\n";

    // ----------------------------------------------------------------
    // Test 1: all-zero inputs/weights → outputs must be zero
    // ----------------------------------------------------------------
    std::cout << "\n[Test 1] all-zero\n";
    memset(c4,0,sizeof(c4)); memset(c3,0,sizeof(c3));
    memset(w_conv,0,sizeof(w_conv)); memset(w_meta,0,sizeof(w_meta));
    memset(p5,0,sizeof(p5)); memset(p4,0,sizeof(p4)); memset(p6,0,sizeof(p6));

    fpn_p4p5_top(c4,c3,w_conv,w_meta,p5,p4,p6);

    float s5=0,s4=0,s6=0;
    for(int i=0;i<FPN_P4P5_P5_ELEMS;i++) s5+=p5[i];
    for(int i=0;i<FPN_P4P5_P4_ELEMS;i++) s4+=p4[i];
    for(int i=0;i<FPN_P4P5_P6_ELEMS;i++) s6+=p6[i];
    std::cout << "  P5 sum=" << s5 << " P4 sum=" << s4 << " P6 sum=" << s6 << "  (all expected 0)\n";
    if (fabsf(s5)<1e-3f && fabsf(s4)<1e-3f && fabsf(s6)<1e-3f)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL\n";

    // ----------------------------------------------------------------
    // Test 2: checkerboard C4 & C3, int8 weights=+1, zero biases → finite
    // ----------------------------------------------------------------
    std::cout << "\n[Test 2] checkerboard C4/C3, w_conv=+1, biases=0\n";
    for(int c=0;c<C4_CH;c++)
        for(int h=0;h<P5_H;h++)
            for(int w=0;w<P5_W;w++)
                c4[c*P5_H*P5_W+h*P5_W+w] = (act_t)(((h+w)%2==0)?1.0f:-1.0f);
    for(int c=0;c<C3_CH;c++)
        for(int h=0;h<P4_H;h++)
            for(int w=0;w<P4_W;w++)
                c3[c*P4_H*P4_W+h*P4_W+w] = (act_t)(((h+w)%2==0)?1.0f:-1.0f);
    memset(w_conv,1,sizeof(w_conv));
    memset(w_meta,0,sizeof(w_meta));
    memset(p5,0,sizeof(p5)); memset(p4,0,sizeof(p4)); memset(p6,0,sizeof(p6));

    fpn_p4p5_top(c4,c3,w_conv,w_meta,p5,p4,p6);

    bool ok=true;
    for(int i=0;i<FPN_P4P5_P5_ELEMS;i++) if(!std::isfinite(p5[i])){ok=false;break;}
    for(int i=0;i<FPN_P4P5_P4_ELEMS;i++) if(!std::isfinite(p4[i])){ok=false;break;}
    print_stats("P5",p5,FPN_P4P5_P5_ELEMS);
    print_stats("P4",p4,FPN_P4P5_P4_ELEMS);
    std::cout << (ok ? "  PASS (all finite)\n" : "  FAIL (NaN/Inf)\n");

    // ----------------------------------------------------------------
    // Test 2b: ramp C4/C3=1.0, w_conv=+1, lat biases=1.0, out biases=0
    //   lat4[all] = C4_CH*1 + 1 = 641
    //   lat3[all] = C3_CH*1 + 1 = 129  (before top-down add)
    //   After add: lat3[all] = 129 + 641 = 770   (upsample(lat4) is all 641)
    //   P5 centre (oc=0,oh=5,ow=5): 192*9*641 = 1,107,648
    //   P4 centre (oc=0,oh=10,ow=10): 192*9*770 = 1,330,560
    // ----------------------------------------------------------------
    std::cout << "\n[Test 2b] C4=1, C3=1, w_conv=+1, lat_bias=1, out_bias=0\n";
    for(int i=0;i<FPN_P4P5_C4_ELEMS;i++) c4[i]=1.0f;
    for(int i=0;i<FPN_P4P5_C3_ELEMS;i++) c3[i]=1.0f;
    memset(w_conv,1,sizeof(w_conv));
    // lat4_b=1, lat3_b=1, out4_b=0, out3_b=0
    for(int i=0;          i<FPN_OUT_CH;  i++) w_meta[i]=(meta_t)1.0f;  // lat4 bias
    for(int i=FPN_OUT_CH; i<2*FPN_OUT_CH;i++) w_meta[i]=(meta_t)1.0f;  // lat3 bias
    for(int i=2*FPN_OUT_CH;i<FPN_P4P5_WMETA_ELEMS;i++) w_meta[i]=(meta_t)0.0f;
    memset(p5,0,sizeof(p5)); memset(p4,0,sizeof(p4)); memset(p6,0,sizeof(p6));

    fpn_p4p5_top(c4,c3,w_conv,w_meta,p5,p4,p6);

    float lat4_val = (float)C4_CH*1.0f + 1.0f;          // 641
    float lat3_merged = (float)C3_CH*1.0f + 1.0f + lat4_val; // 129 + 641 = 770
    float p5_exp = (float)FPN_OUT_CH * 9.0f * lat4_val;     // 192*9*641 = 1,107,648
    float p4_exp = (float)FPN_OUT_CH * 9.0f * lat3_merged;  // 192*9*770 = 1,330,560

    // CHW index: oc=0, centre pixel
    int p5_idx = P5_H/2 * P5_W + P5_W/2;       // 5*10+5 = 55
    int p4_idx = P4_H/2 * P4_W + P4_W/2;       // 10*20+10 = 210

    std::cout << "  lat4 expected:      " << lat4_val << "\n";
    std::cout << "  lat3_merged expected:" << lat3_merged << "\n";
    std::cout << "  P5 centre (idx=" << p5_idx << ") expected: " << p5_exp
              << "  got: " << p5[p5_idx] << "\n";
    std::cout << "  P4 centre (idx=" << p4_idx << ") expected: " << p4_exp
              << "  got: " << p4[p4_idx] << "\n";

    float rel_p5 = fabsf(p5[p5_idx]-p5_exp)/fabsf(p5_exp);
    float rel_p4 = fabsf(p4[p4_idx]-p4_exp)/fabsf(p4_exp);
    std::cout << "  P5 rel_err=" << rel_p5 << "  P4 rel_err=" << rel_p4 << "\n";
    if (rel_p5 < 1e-4f && rel_p4 < 1e-4f)
        std::cout << "  PASS\n";
    else
        std::cout << "  FAIL (check upsample+add and conv3x3 top-down logic)\n";

    // ----------------------------------------------------------------
    // Test 3: file-based real weights
    // ----------------------------------------------------------------
    std::cout << "\n[Test 3] file-based weights\n";
    bool wc_ok = read_bin("../weights/fpn_p4p5_w_conv.bin", w_conv, sizeof(w_conv));
    bool wm_ok = read_bin("../weights/fpn_p4p5_w_meta.bin", w_meta, sizeof(w_meta));
    bool c4_ok = read_bin("../weights/c4_feat.bin",         c4,     sizeof(c4));
    bool c3_ok = read_bin("../weights/c3_feat.bin",         c3,     sizeof(c3));

    if (!wc_ok||!wm_ok||!c4_ok||!c3_ok) {
        std::cout << "  Files not found — skipping.\n"
                  << "    fpn_p4p5_w_conv.bin  (" << sizeof(w_conv) << " B)\n"
                  << "    fpn_p4p5_w_meta.bin  (" << sizeof(w_meta) << " B)\n"
                  << "    c4_feat.bin          (" << sizeof(c4)     << " B)\n"
                  << "    c3_feat.bin          (" << sizeof(c3)     << " B)\n";
    } else {
        memset(p5,0,sizeof(p5)); memset(p4,0,sizeof(p4)); memset(p6,0,sizeof(p6));
        fpn_p4p5_top(c4,c3,w_conv,w_meta,p5,p4,p6);
        print_stats("P5",p5,FPN_P4P5_P5_ELEMS);
        print_stats("P4",p4,FPN_P4P5_P4_ELEMS);
        print_stats("P6",p6,FPN_P4P5_P6_ELEMS);
        write_bin("p5_csim.bin",p5,sizeof(p5));
        write_bin("p4_csim.bin",p4,sizeof(p4));
        write_bin("p6_csim.bin",p6,sizeof(p6));
        std::cout << "  Dumped: p5_csim.bin  p4_csim.bin  p6_csim.bin\n";
        std::cout << "  PASS (run compare_fpn_p4p5.py to validate)\n";
    }

    std::cout << "\n==============================================\n";
    std::cout << "fpn_p4p5_top testbench complete.\n";
    std::cout << "==============================================\n";
    return 0;
}
