
#include <torch/extension.h>
#include <functional>
#include <vector>
#include "FastConv_Conv_Method.cuh"


// LinCombs shape: (out_channels, in_channels*9). Assumed to be contiguous
// basisResultsTensor shape: (in_channels*9, H*W). Assumed to be contiguous. values are discarded
// colwiseResults shape (3, inputDim * resultDim). input_dim rows, result_dim cols
torch::Tensor forward(
    const torch::Tensor &input, 
    const torch::Tensor &linCombs, 
    torch::Tensor &basisResultsTensor,
    torch::Tensor &colwiseResults,
    bool applyConvolution)
{
    const int batchSize         = input.size(0);
	const int inChannels        = input.size(1);
	const int inputDim          = input.size(2);
    const int inputSize         = inputDim * inputDim;
	const int resultDim         = inputDim - 2;  // Explanation: (I - K + 2*pad)/stride+1 =>_{stride=1} I - K + 2*pad + 1 =>_{K=4} I -3 + 2*pad 
    const int resultSize        = resultDim * resultDim;
    const int outChannels       = linCombs.size(0);

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(colwiseResults.type().is_cuda(), "colwiseResults must be a CUDA tensor");
    AT_ASSERTM(basisResultsTensor.type().is_cuda(), "basisResultsTensor must be a CUDA tensor");
    // cudaStream_t stream = torch::cuda::getCurrentCUDAStream();

    // Create output
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false).device(input.device());
    auto resultTensor = torch::empty({ batchSize, outChannels, resultSize }, options);
    
	for (int b_ix = 0; b_ix < batchSize; b_ix++)
	{
        int lastUsedBasisResult = 0;
        for (int c_ix = 0; c_ix < inChannels; c_ix++)
        {
            if (applyConvolution)
            {
                Convolution3x3ToBasis(input, colwiseResults, basisResultsTensor/*.data<float>()*/, b_ix, c_ix, inputDim, resultDim, inputSize);
            }
        }
       
        auto batchResults = torch::matmul_out(resultTensor[b_ix], linCombs, basisResultsTensor);
	}

    auto ret = resultTensor.view({ batchSize, outChannels, resultDim, resultDim });
    
    return resultTensor;
}

torch::Tensor conv_fwd_3x3(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor, torch::Tensor &colwiseResults)
{
    return forward(input_tensor, linCombs, basisResultsTensor, colwiseResults, true);
}

torch::Tensor matmul_only(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor, torch::Tensor &colwiseResults)
{
    return forward(input_tensor, linCombs, basisResultsTensor, colwiseResults, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_only", &matmul_only, "Perform only the second part of the convolution - MatMul");
    m.def("conv_fwd_3x3", &conv_fwd_3x3, "forward with 3x3 kernel");
}