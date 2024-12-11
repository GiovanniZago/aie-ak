#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#define __X86DEBUG__
#define __X86DEBUGDATA__
#define __X86DEBUGDIST__

#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "adf.h"

#define V_SIZE 8
#define P_BUNCHES 16
#define NUM_PARTICLES 128

static const int16 PI = 720;
static const int16 MPI = -720;
static const int16 TWOPI = 1440;
static const int16 MTWOPI = -1440;
static const float invR02 = 1 / (0.4 * 0.4); 
static const float PI_FLOAT = 3.1415926;
static const float MPI_FLOAT = -3.1415926;
static const float TWOPI_FLOAT = 6.2831852;
static const float MTWOPI_FLOAT = -6.2831852;

using namespace adf;

void antiKt(input_stream<float> * __restrict in, output_stream<float> * __restrict out);

#endif