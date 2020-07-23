#pragma once
#include <vector>
#include <list>

#define DTYPE float
#define INT int64_t

void Convolution3x3ToBasis(const DTYPE *input, DTYPE **res, const size_t input_dim, const size_t result_dim);