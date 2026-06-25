/* testbench_mvit_blk_s4.cpp */
#include "mvit_blk_s4_top.h"
#include <cstdlib>
#include <cstdio>
static act_t    in    [MVIT_S4_IN_ELEMS];
static act_t    out   [MVIT_S4_OUT_ELEMS];
static weight_t w_conv[MVIT_S4_WCONV_ELEMS];
static meta_t   w_meta[MVIT_S4_WMETA_ELEMS];
int main() {
    srand(42);
    for (int i = 0; i < MVIT_S4_IN_ELEMS;    i++) in    [i] = (act_t)((rand()%200-100)*0.01f);
    for (int i = 0; i < MVIT_S4_WCONV_ELEMS; i++) w_conv[i] = (weight_t)(rand()%255-127);
    for (int i = 0; i < MVIT_S4_WMETA_ELEMS; i++) w_meta[i] = (meta_t)((rand()%200-100)*0.001f);
    printf("Running mvit_blk_s4_top C-sim...\n");
    mvit_blk_s4_top(in, out, w_conv, w_meta);
    int nonzero = 0;
    for (int i = 0; i < MVIT_S4_OUT_ELEMS; i++) if (out[i] != 0.0f) nonzero++;
    printf("mvit_blk_s4_top: out[0]=%.4f  nonzero=%d/%d\n", (float)out[0], nonzero, MVIT_S4_OUT_ELEMS);
    if (nonzero == 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n"); return 0;
}
