/*
 * testbench_mbconv_s2a.cpp — C-sim for mbconv_s2a_top
 */
#include "mbconv_s2a_top.h"
#include <cstdlib>
#include <cstdio>

static act_t    in    [MBCONVS2A_IN_ELEMS];
static act_t    out   [MBCONVS2A_OUT_ELEMS];
static weight_t w_conv[MBCONVS2A_WCONV_ELEMS];
static meta_t   w_meta[MBCONVS2A_WMETA_ELEMS];

int main() {
    srand(42);
    for (int i = 0; i < MBCONVS2A_IN_ELEMS;    i++) in    [i] = (act_t)((rand()%200-100)*0.01f);
    for (int i = 0; i < MBCONVS2A_WCONV_ELEMS; i++) w_conv[i] = (weight_t)(rand()%255-127);
    for (int i = 0; i < MBCONVS2A_WMETA_ELEMS; i++) w_meta[i] = (meta_t)((rand()%100)*0.01f+0.01f);

    mbconv_s2a_top(in, out, w_conv, w_meta);

    int nonzero = 0;
    for (int i = 0; i < MBCONVS2A_OUT_ELEMS; i++)
        if (out[i] != 0.0f) nonzero++;
    printf("mbconv_s2a_top: out[0]=%.4f  nonzero=%d/%d\n",
           (float)out[0], nonzero, MBCONVS2A_OUT_ELEMS);
    if (nonzero == 0) { printf("FAIL: all outputs zero\n"); return 1; }
    printf("PASS\n");
    return 0;
}
