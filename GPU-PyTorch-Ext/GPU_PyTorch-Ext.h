#pragma once

#include <torch/extension.h>
#include <functional>
#include <vector>
#include "GCK-Conv-Method.cuh"

#ifdef VSA
namespace torch
{
#define C10_API
    
    template <typename T>
	class ArrayRef final {
	public:
		template<typename A>
		ArrayRef(const std::vector<int, A> &Vec)
		{ }
		ArrayRef(const std::initializer_list<int> &l) { }
		ArrayRef(int batchSize, int inChannels, int kernel_dim, int inputDim, int resultsWidth) {}
	};
	using IntArrayRef = ArrayRef<int>;

    class DataType { };
    class Device { };
	DataType kF32;
    Device kCUDA;
    struct C10_API TensorOptions
    {
        TensorOptions dtype(DataType p) { return *this; }
        TensorOptions requires_grad(bool b) { return *this; }
        TensorOptions is_variable(bool b) { return *this; }
        TensorOptions device(Device p, int a) { return *this; }
        TensorOptions device(Device p) { return *this; }
    };

	class Tensor
	{
	public:
		int size(int i) const { return 0; }
		Tensor operator[](int i) const { return Tensor(); }
		template<typename T> T *data() const { return NULL; }
		template<typename T> T item() { return T(); }
		size_t itemsize() const { return 0; }
		void operator=(DTYPE d) { }
		const Tensor contiguous() const { return Tensor(); }
        Tensor view(IntArrayRef rew) const { return Tensor();  }
        Tensor cuda() { return Tensor(); }
        DataType type() { return kF32; }
        Device device() const { return kCUDA; }
	};

	Tensor empty(IntArrayRef a, DataType k) { return Tensor(); }
    Tensor empty(IntArrayRef a, const TensorOptions k) { return Tensor(); }

    using Deleter = std::function<void(void*)>;
    
    Tensor from_blob(void *data, IntArrayRef sizes, const TensorOptions &options = TensorOptions())
    {
        return Tensor();
    }
    Tensor from_blob(void *data, IntArrayRef sizes, IntArrayRef strides, const Deleter &deleter, const TensorOptions &options = TensorOptions())
    {
        return Tensor();
    }
    Tensor mm(const Tensor &self, const Tensor &other) { return Tensor(); }
    Tensor matmul_out(const Tensor &out,  const Tensor &self, const Tensor &other) { return Tensor(); }
    Tensor stack(const std::vector<Tensor> &self)
    {
        return Tensor();
    }
}

int TORCH_EXTENSION_NAME = 5;
class Installer {
public: void def(const char *str, void *func, const char *str2) {}
};
Installer m;
#define PYBIND11_MODULE(s,m) void A(int s, Installer m)

#endif


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

    // Create output
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false).device(input.device());
    auto resultTensor = torch::empty({ batchSize, outChannels, resultSize }, options);
    
	for (int b_ix = 0; b_ix < batchSize; b_ix++)
	{
        int lastUsedBasisResult = 0;
        for (int c_ix = 0; c_ix < inChannels; c_ix++)
        {
            if (applyConvolution)
                Convolution3x3ToBasis(input, colwiseResults, basisResultsTensor, b_ix, c_ix, inputDim, resultDim);
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
