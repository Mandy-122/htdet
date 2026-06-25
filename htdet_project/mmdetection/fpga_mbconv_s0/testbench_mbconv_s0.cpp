/*
 * testbench_mbconv_s0.cpp — C-sim for mbconv_s0_top
 */
#include "mbconv_s0_top.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

static act_t    in    [MBCONVS0_IN_ELEMS];
static act_t    out   [MBCONVS0_OUT_ELEMS];
static weight_t w_conv[MBCONVS0_WCONV_ELEMS];
static meta_t   w_meta[MBCONVS0_WMETA_ELEMS];

int main() {
    srand(42);
    for (int i = 0; i < MBCONVS0_IN_ELEMS;    i++) in    [i] = (act_t)((rand()%200-100)*0.01f);
    for (int i = 0; i < MBCONVS0_WCONV_ELEMS; i++) w_conv[i] = (weight_t)(rand()%255-127);
    for (int i = 0; i < MBCONVS0_WMETA_ELEMS; i++) w_meta[i] = (meta_t)((rand()%100)*0.01f+0.01f);

    mbconv_s0_top(in, out, w_conv, w_meta);

    int nonzero = 0;
    float sum = 0.0f;
    for (int i = 0; i < MBCONVS0_OUT_ELEMS; i++) {
        if (out[i] != 0.0f) nonzero++;
        sum += (float)out[i];
    }
    printf("mbconv_s0_top: out[0]=%.4f  nonzero=%d/%d  sum=%.2f\n",
           (float)out[0], nonzero, MBCONVS0_OUT_ELEMS, sum);
    if (nonzero == 0) { printf("FAIL: all outputs zero\n"); return 1; }
    printf("PASS\n");
    return 0;
}
