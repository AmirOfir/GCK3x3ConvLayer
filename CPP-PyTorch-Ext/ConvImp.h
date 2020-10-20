#pragma once
#include "Header.h"

void ConvolutionRowwise(const DTYPE *input, DTYPE *res[3], const size_t input_dim, const size_t input_size, const bool singleCellPadding);
void ConvolutionColwise(const DTYPE *input, DTYPE *res[3], const size_t result_dim);
void ConvolutionColwiseSingleCellPadding(const DTYPE *input, DTYPE *res[3], const size_t result_dim);