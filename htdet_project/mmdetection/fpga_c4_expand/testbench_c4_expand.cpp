/*
 * testbench_c4_expand.cpp — C-sim for c4_expand_top
 */
#include "c4_expand_top.h"
#include <cstdlib>
#include <cstdio>

static act_t    in    [C4EXP_IN_ELEMS];
static act_t    out   [C4EXP_OUT_ELEMS];
static weight_t w_conv[C4EXP_WCONV_ELEMS];
static meta_t   w_meta[C4EXP_WMETA_ELEMS];

int main() {
    srand(42);
    for (int i = 0; i < C4EXP_IN_ELEMS;    i++) in    [i] = (act_t)((rand()%200-100)*0.01f);
    for (int i = 0; i < C4EXP_WCONV_ELEMS; i++) w_conv[i] = (weight_t)(rand()%255-127);
    for (int i = 0; i < C4EXP_WMETA_ELEMS; i++) w_meta[i] = (meta_t)((rand()%100)*0.01f+0.01f);

    c4_expand_top(in, out, w_conv, w_meta);

    int nonzero = 0;
    for (int i = 0; i < C4EXP_OUT_ELEMS; i++)
        if (out[i] != 0.0f) nonzero++;
    printf("c4_expand_top: out[0]=%.4f  nonzero=%d/%d\n",
           (float)out[0], nonzero, C4EXP_OUT_ELEMS);
    if (nonzero == 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
