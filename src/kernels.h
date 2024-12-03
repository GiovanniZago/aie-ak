#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#define __X86DEBUG__
#define __X86DEBUGDISTBEAM__
// #define __X86DEBUGDIST__
#define __X86DEBUGRECOMB__

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "adf.h"

#define EV_SIZE 64
#define V_SIZE 32
#define P_BUNCHES 2
#define N_EPOCH 2

static const int16 PI = 720;
static const int16 MPI = -720;
static const int16 TWOPI = 1440;
static const int16 MTWOPI = -1440;
static const float invR02 = 2.7778; // R0 = 0.6
static const float PI_FLOAT = 3.1415926;
static const float INVPT_CONV2 = 1 / (0.25 * 0.25);
static const float ANG_CONV2 = (PI_FLOAT / PI) * (PI_FLOAT / PI);

using namespace adf;

void antiKt(input_stream<int16> * __restrict in, output_stream<int16> * __restrict out);

#endif
