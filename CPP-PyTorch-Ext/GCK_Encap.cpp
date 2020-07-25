#include <torch/extension.h>
#include "Header.h"
#include <functional>

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
	DataType kF32;

    struct C10_API TensorOptions
    {
        TensorOptions dtype(DataType p) { return *this; }
        TensorOptions requires_grad(bool b) { return *this; }
        TensorOptions is_variable(bool b) { return *this; }
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
torch::Tensor forward(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor, 
    void (*convolution3x3ToBasis)(const float *, float **res, const size_t input_dim, const size_t result_dim))
{
    
	const torch::Tensor input = input_tensor.contiguous();
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false).is_variable(false);

	const DTYPE *inpPtr = input.data<DTYPE>();
	const int batchSize = input.size(0);
	const int inChannels = input.size(1);
	const int inputDim = input.size(2);
    const int inputSize = inputDim * inputDim;

    // Explanation: (I - K + 2*pad)/stride+1 =>_{stride=1} I - K + 2*pad + 1 =>_{K=4} I -3 + 2*pad
	const int resultDim = inputDim - 2; 
	const int resultSize = resultDim * resultDim;
	const int outChannels = linCombs.size(0);
    const int basisResultsNum = inChannels * 9;

    const auto resultTensor = torch::empty({ batchSize, outChannels, resultSize }, options);

    DTYPE *basisResults = basisResultsTensor.data<DTYPE>();
	
	for (int b_ix = 0; b_ix < batchSize; b_ix++)
	{
        DTYPE *lastUsedBasisResult = basisResults;
        for (int c_ix = 0; c_ix < inChannels; c_ix++)
        {
            DTYPE *basisResults2d[9];
		    for (auto i = 0; i < 9; i++)
		    {
			    basisResults2d[i] = lastUsedBasisResult;
                lastUsedBasisResult += resultSize;
		    }
            convolution3x3ToBasis(inpPtr, basisResults2d, inputDim, resultDim);
            inpPtr += inputSize;
        }
       
        auto batchResults = torch::matmul_out(resultTensor[b_ix], linCombs, basisResultsTensor);
	}

    auto ret = resultTensor.view({ batchSize, outChannels, resultDim, resultDim });

    return ret;
}

torch::Tensor conv_fwd_3x3(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor)
{
    return forward(input_tensor, linCombs, basisResultsTensor, &Convolution3x3ToBasis);
}

void nop(const float *, float **res, const size_t input_dim, const size_t result_dim) { }
torch::Tensor matmul_only(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor)
{
    return forward(input_tensor, linCombs, basisResultsTensor, &nop);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_only", &matmul_only, "Perform only the second part of the convolution - MatMul");
	m.def("conv_fwd_3x3", &conv_fwd_3x3, "forward with 3x3 kernel");
}