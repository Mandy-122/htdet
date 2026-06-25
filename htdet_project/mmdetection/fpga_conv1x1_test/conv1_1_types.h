#ifndef CONV1_1_TYPES_H
#define CONV1_1_TYPES_H

#include <stdint.h>

typedef float   act_t;
typedef int8_t  weight_t;
typedef float   acc_t;
typedef float   meta_t;

#define TEST_IN_CH  256   // full scale IC
#define TEST_OUT_CH 256   // full scale OC
#define TEST_H       80   // full scale H
#define TEST_W       80   // full scale W

#define TEST_IN_ELEMS   (TEST_IN_CH  * TEST_H * TEST_W)
#define TEST_OUT_ELEMS  (TEST_OUT_CH * TEST_H * TEST_W)
#define TEST_W_ELEMS    (TEST_OUT_CH * TEST_IN_CH)
#define TEST_META_ELEMS (TEST_OUT_CH + TEST_OUT_CH)

#endif
