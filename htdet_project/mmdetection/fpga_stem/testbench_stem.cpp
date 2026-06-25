/*
 * testbench_stem.cpp — C-sim testbench for stem_top
 */

#include "stem_top.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>

static act_t    in    [STEM_IN_ELEMS];
static act_t    out   [STEM_OUT_ELEMS];
static weight_t w_conv[STEM_WCONV_ELEMS];
static meta_t   w_meta[STEM_WMETA_ELEMS];

int main() {
    // Random init
    srand(42);
    for (int i = 0; i < STEM_IN_ELEMS;     i++) in    [i] = (act_t)((rand()%100-50) * 0.05f);
    for (int i = 0; i < STEM_WCONV_ELEMS;  i++) w_conv[i] = (weight_t)(rand()%255 - 127);
    for (int i = 0; i < STEM_WMETA_ELEMS;  i++) w_meta[i] = (meta_t)((rand()%100) * 0.01f + 0.01f);

    stem_top(in, out, w_conv, w_meta);

    // Sanity: count non-zero outputs (SiLU suppresses negatives)
    int nonzero = 0;
    float sum = 0.0f;
    for (int i = 0; i < STEM_OUT_ELEMS; i++) {
        if (out[i] != 0.0f) nonzero++;
        sum += out[i];
    }
    printf("stem_top: out[0]=%.4f  nonzero=%d/%d  sum=%.2f\n",
           (float)out[0], nonzero, STEM_OUT_ELEMS, sum);
    if (nonzero == 0) { printf("FAIL: all outputs zero\n"); return 1; }
    printf("PASS\n");
    return 0;
}
