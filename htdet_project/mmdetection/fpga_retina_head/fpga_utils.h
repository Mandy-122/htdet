/*
 * fpga_utils.h  (retina_head isolated module)
 * Activation functions used by the retina head.
 */

#ifndef FPGA_UTILS_H
#define FPGA_UTILS_H

#include "fpga_types.h"
#include <cmath>

inline act_t relu(act_t x) {
    #pragma HLS INLINE
    return (x > (act_t)0) ? x : (act_t)0;
}

// Piecewise-linear sigmoid for synthesis; exact expf for CSIM / reference.
inline score_t sigmoid(score_t x) {
    #pragma HLS INLINE
    #ifdef __SYNTHESIS__
    if (x < (score_t)(-4.0f)) return (score_t)0.0f;
    if (x > (score_t)( 4.0f)) return (score_t)1.0f;
    return (score_t)0.5f + x * (score_t)0.125f;
    #else
    return (score_t)(1.0f / (1.0f + expf(-(float)x)));
    #endif
}

#endif // FPGA_UTILS_H
