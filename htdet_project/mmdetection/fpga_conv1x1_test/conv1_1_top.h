#ifndef CONV1_1_TOP_H
#define CONV1_1_TOP_H

#include "conv1_1_types.h"

void conv1x1_top(
    const act_t    in     [TEST_IN_ELEMS],
          act_t    out    [TEST_OUT_ELEMS],
    const weight_t w_conv [TEST_W_ELEMS],
    const meta_t   w_meta [TEST_META_ELEMS]
);

#endif
