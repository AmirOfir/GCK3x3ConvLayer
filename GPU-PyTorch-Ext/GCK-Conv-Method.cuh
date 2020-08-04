#pragma once
#define DTYPE float

#include <vector>
#include "cuda_runtime.h"

void Convolution3x3ToBasis(
    const DTYPE *input, /* Input channel*/
    DTYPE *colResults[3], /* 3 * input_dim * result_dim */
    DTYPE **res, /* 9 * result_dim * result_dim */
    const int input_dim,
    const int result_dim);

std::vector<float *> createColResults(int colwiseSize);