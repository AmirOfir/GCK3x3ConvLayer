#ifndef __FAST_CONV_H__
#define __FAST_CONV_H__

//#define DTYPE float
#include <torch/extension.h>
#include <vector>
#include <algorithm>

void Convolution3x3ToBasis(
    const torch::Tensor &input, /* Input channel*/
    torch::Tensor &colwiseResults, /* 3 * input_dim * result_dim */
    torch::Tensor &basisResultsTensor, /* 9 * result_dim * result_dim */
    int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim,
    int input_size);

#endif