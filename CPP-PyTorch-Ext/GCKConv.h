#pragma once
#include "Header.h"

void ConvolutionColwise(const DTYPE *input, DTYPE **res, const size_t input_dim, const size_t input_size, const size_t result_dim);
void ConvolutionRowwise(DTYPE **res, const int row, const size_t input_dim);