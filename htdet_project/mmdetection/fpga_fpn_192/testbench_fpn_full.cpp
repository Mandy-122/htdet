/*
 * testbench_fpn_full.cpp  — Phase 4: full FPN P2+P3+P4+P5+P6
 *
 * Test 1: all-zero → all-zero
 * Test 2: checkerboard, finite check
 * Test 2b: C1=C2=C3=C4=1, w=+1, lat_bias=1, out_bias=0
 *   lat4=641, lat3_m=770, lat2_m=867, lat1_m=C1+1+867=932
 *   P5 centre (idx=55):   192*9*641 = 1,107,648
 *   P4 centre (idx=210):  192*9*770 = 1,330,560
 *   P3 centre (idx=820):  192*9*867 = 1,498,176
 *   P2 centre (idx=3240): 192*9*932 = 1,610,496
 * Test 3: file-based weights (if present)
 *
 * Build:
 *   g++ -std=c++14 -O0 -I. -I../fpga_temp/hls_stubs \
 *       testbench_fpn_full.cpp fpn_full_top.cpp -lm -o tb_fpn_full
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include "fpn_full_top.h"

static bool read_bin(const char* p,void* b,size_t n){
    std::ifstream f(p,std::ios::binary);if(!f)return false;
    f.read((char*)b,n);return(size_t)f.gcount()==n;}
static void write_bin(const char* p,const void* b,size_t n){
    std::ofstream f(p,std::ios::binary);f.write((const char*)b,n);}
static void stats(const char* nm,const act_t* a,int n){
    float s=0,mn=a[0],mx=a[0];
    for(int i=0;i<n;i++){s+=a[i];if(a[i]<mn)mn=a[i];if(a[i]>mx)mx=a[i];}
    std::cout<<"  "<<nm<<" sum="<<s<<"  min="<<mn<<"  max="<<mx<<"\n";}
static bool all_finite(const act_t* a,int n){
    for(int i=0;i<n;i++)if(!std::isfinite(a[i]))return false;return true;}

// Static buffers — P2 is 1,228,800 floats = 4.8 MB, must be static
static act_t    c4    [FPN_FULL_C4_ELEMS];
static act_t    c3    [FPN_FULL_C3_ELEMS];
static act_t    c2    [FPN_FULL_C2_ELEMS];
static act_t    c1    [FPN_FULL_C1_ELEMS];
static weight_t w_conv[FPN_FULL_WCONV_ELEMS];
static meta_t   w_meta[FPN_FULL_WMETA_ELEMS];
static act_t    p5    [FPN_FULL_P5_ELEMS];
static act_t    p4    [FPN_FULL_P4_ELEMS];
static act_t    p3    [FPN_FULL_P3_ELEMS];
static act_t    p2    [FPN_FULL_P2_ELEMS];
static act_t    p6    [FPN_FULL_P6_ELEMS];

int main() {
    std::cout << "==============================================\n";
    std::cout << "fpn_full_top  testbench — Phase 4\n";
    std::cout << "P2+P3+P4+P5+P6  (FPN_OUT_CH=" << FPN_OUT_CH
              << "  P2=" << P2_H << "x" << P2_W
              << "  P3=" << P3_H << "x" << P3_W << ")\n";
    std::cout << "==============================================\n";

    // ---- Test 1: all-zero ----
    std::cout << "\n[Test 1] all-zero\n";
    memset(c4,0,sizeof(c4)); memset(c3,0,sizeof(c3));
    memset(c2,0,sizeof(c2)); memset(c1,0,sizeof(c1));
    memset(w_conv,0,sizeof(w_conv)); memset(w_meta,0,sizeof(w_meta));
    memset(p5,0,sizeof(p5)); memset(p4,0,sizeof(p4));
    memset(p3,0,sizeof(p3)); memset(p2,0,sizeof(p2)); memset(p6,0,sizeof(p6));

    fpn_full_top(c4,c3,c2,c1,w_conv,w_meta,p5,p4,p3,p2,p6);

    float s2=0,s3=0,s4=0,s5=0,s6=0;
    for(int i=0;i<FPN_FULL_P2_ELEMS;i++) s2+=p2[i];
    for(int i=0;i<FPN_FULL_P3_ELEMS;i++) s3+=p3[i];
    for(int i=0;i<FPN_FULL_P4_ELEMS;i++) s4+=p4[i];
    for(int i=0;i<FPN_FULL_P5_ELEMS;i++) s5+=p5[i];
    for(int i=0;i<FPN_FULL_P6_ELEMS;i++) s6+=p6[i];
    bool ok1=fabsf(s2)<1e-3f&&fabsf(s3)<1e-3f&&fabsf(s4)<1e-3f&&fabsf(s5)<1e-3f&&fabsf(s6)<1e-3f;
    std::cout<<"  P2="<<s2<<" P3="<<s3<<" P4="<<s4<<" P5="<<s5<<" P6="<<s6<<"\n";
    std::cout<<(ok1?"  PASS\n":"  FAIL\n");

    // ---- Test 2: checkerboard, finite check ----
    std::cout << "\n[Test 2] checkerboard, unit weights, zero biases\n";
    for(int c=0;c<C1_CH;c++) for(int h=0;h<P2_H;h++) for(int w=0;w<P2_W;w++)
        c1[c*P2_H*P2_W+h*P2_W+w]=(act_t)(((h+w)%2==0)?1.f:-1.f);
    for(int c=0;c<C2_CH;c++) for(int h=0;h<P3_H;h++) for(int w=0;w<P3_W;w++)
        c2[c*P3_H*P3_W+h*P3_W+w]=(act_t)(((h+w)%2==0)?1.f:-1.f);
    for(int c=0;c<C3_CH;c++) for(int h=0;h<P4_H;h++) for(int w=0;w<P4_W;w++)
        c3[c*P4_H*P4_W+h*P4_W+w]=(act_t)(((h+w)%2==0)?1.f:-1.f);
    for(int c=0;c<C4_CH;c++) for(int h=0;h<P5_H;h++) for(int w=0;w<P5_W;w++)
        c4[c*P5_H*P5_W+h*P5_W+w]=(act_t)(((h+w)%2==0)?1.f:-1.f);
    memset(w_conv,1,sizeof(w_conv)); memset(w_meta,0,sizeof(w_meta));
    memset(p2,0,sizeof(p2)); memset(p3,0,sizeof(p3));
    memset(p4,0,sizeof(p4)); memset(p5,0,sizeof(p5)); memset(p6,0,sizeof(p6));

    fpn_full_top(c4,c3,c2,c1,w_conv,w_meta,p5,p4,p3,p2,p6);

    bool ok2=all_finite(p2,FPN_FULL_P2_ELEMS)&&all_finite(p3,FPN_FULL_P3_ELEMS)
           &&all_finite(p4,FPN_FULL_P4_ELEMS)&&all_finite(p5,FPN_FULL_P5_ELEMS);
    stats("P5",p5,FPN_FULL_P5_ELEMS); stats("P4",p4,FPN_FULL_P4_ELEMS);
    stats("P3",p3,FPN_FULL_P3_ELEMS); stats("P2",p2,FPN_FULL_P2_ELEMS);
    std::cout<<(ok2?"  PASS (all finite)\n":"  FAIL (NaN/Inf)\n");

    // ---- Test 2b: numerical check ----
    std::cout << "\n[Test 2b] all-1 inputs, w=+1, lat_bias=1, out_bias=0\n";
    for(int i=0;i<FPN_FULL_C4_ELEMS;i++) c4[i]=1.f;
    for(int i=0;i<FPN_FULL_C3_ELEMS;i++) c3[i]=1.f;
    for(int i=0;i<FPN_FULL_C2_ELEMS;i++) c2[i]=1.f;
    for(int i=0;i<FPN_FULL_C1_ELEMS;i++) c1[i]=1.f;
    memset(w_conv,1,sizeof(w_conv));
    for(int i=0;i<4*FPN_OUT_CH;i++) w_meta[i]=(meta_t)1.f;  // lat4..lat1 bias=1
    for(int i=4*FPN_OUT_CH;i<FPN_FULL_WMETA_ELEMS;i++) w_meta[i]=(meta_t)0.f;
    memset(p2,0,sizeof(p2)); memset(p3,0,sizeof(p3));
    memset(p4,0,sizeof(p4)); memset(p5,0,sizeof(p5)); memset(p6,0,sizeof(p6));

    fpn_full_top(c4,c3,c2,c1,w_conv,w_meta,p5,p4,p3,p2,p6);

    float lat4v=(float)C4_CH+1.f;                    // 641
    float lat3v=(float)C3_CH+1.f+lat4v;              // 770
    float lat2v=(float)C2_CH+1.f+lat3v;              // 867
    float lat1v=(float)C1_CH+1.f+lat2v;              // 932
    float exp_p5=(float)FPN_OUT_CH*9.f*lat4v;        // 1,107,648
    float exp_p4=(float)FPN_OUT_CH*9.f*lat3v;        // 1,330,560
    float exp_p3=(float)FPN_OUT_CH*9.f*lat2v;        // 1,498,176
    float exp_p2=(float)FPN_OUT_CH*9.f*lat1v;        // 1,610,496

    float got_p5=p5[P5_H/2*P5_W+P5_W/2];            // idx=55
    float got_p4=p4[P4_H/2*P4_W+P4_W/2];            // idx=210
    float got_p3=p3[P3_H/2*P3_W+P3_W/2];            // idx=820
    float got_p2=p2[P2_H/2*P2_W+P2_W/2];            // idx=3240

    std::cout<<"  lat4="<<lat4v<<" lat3_m="<<lat3v
             <<" lat2_m="<<lat2v<<" lat1_m="<<lat1v<<"\n";
    std::cout<<"  P5 exp="<<exp_p5<<" got="<<got_p5
             <<" err="<<fabsf(got_p5-exp_p5)/exp_p5<<"\n";
    std::cout<<"  P4 exp="<<exp_p4<<" got="<<got_p4
             <<" err="<<fabsf(got_p4-exp_p4)/exp_p4<<"\n";
    std::cout<<"  P3 exp="<<exp_p3<<" got="<<got_p3
             <<" err="<<fabsf(got_p3-exp_p3)/exp_p3<<"\n";
    std::cout<<"  P2 exp="<<exp_p2<<" got="<<got_p2
             <<" err="<<fabsf(got_p2-exp_p2)/exp_p2<<"\n";

    bool ok2b=fabsf(got_p5-exp_p5)/exp_p5<1e-4f
           &&fabsf(got_p4-exp_p4)/exp_p4<1e-4f
           &&fabsf(got_p3-exp_p3)/exp_p3<1e-4f
           &&fabsf(got_p2-exp_p2)/exp_p2<1e-4f;
    std::cout<<(ok2b?"  PASS\n":"  FAIL\n");

    // ---- Test 3: file-based ----
    std::cout << "\n[Test 3] file-based weights\n";
    bool ok=read_bin("../weights/fpn_full_w_conv.bin",w_conv,sizeof(w_conv))
          &&read_bin("../weights/fpn_full_w_meta.bin",w_meta,sizeof(w_meta))
          &&read_bin("../weights/c4_feat.bin",c4,sizeof(c4))
          &&read_bin("../weights/c3_feat.bin",c3,sizeof(c3))
          &&read_bin("../weights/c2_feat.bin",c2,sizeof(c2))
          &&read_bin("../weights/c1_feat.bin",c1,sizeof(c1));
    if(!ok){
        std::cout<<"  Files not found — skipping.\n"
                 <<"    fpn_full_w_conv.bin ("<<sizeof(w_conv)<<" B)\n"
                 <<"    fpn_full_w_meta.bin ("<<sizeof(w_meta)<<" B)\n"
                 <<"    c1_feat.bin / c2_feat.bin / c3_feat.bin / c4_feat.bin\n";
    } else {
        memset(p2,0,sizeof(p2));memset(p3,0,sizeof(p3));
        memset(p4,0,sizeof(p4));memset(p5,0,sizeof(p5));memset(p6,0,sizeof(p6));
        fpn_full_top(c4,c3,c2,c1,w_conv,w_meta,p5,p4,p3,p2,p6);
        stats("P5",p5,FPN_FULL_P5_ELEMS); stats("P4",p4,FPN_FULL_P4_ELEMS);
        stats("P3",p3,FPN_FULL_P3_ELEMS); stats("P2",p2,FPN_FULL_P2_ELEMS);
        write_bin("p2_csim.bin",p2,sizeof(p2)); write_bin("p3_csim.bin",p3,sizeof(p3));
        write_bin("p4_csim.bin",p4,sizeof(p4)); write_bin("p5_csim.bin",p5,sizeof(p5));
        write_bin("p6_csim.bin",p6,sizeof(p6));
        std::cout<<"  Dumped p2–p6_csim.bin\n  PASS\n";
    }

    std::cout<<"\n==============================================\n";
    std::cout<<"fpn_full_top testbench complete.\n";
    std::cout<<"==============================================\n";
    return 0;
}
