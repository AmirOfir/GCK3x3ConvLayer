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
	DataType kF32;
    DataType kCUDA;
    struct C10_API TensorOptions
    {
        TensorOptions dtype(DataType p) { return *this; }
        TensorOptions requires_grad(bool b) { return *this; }
        TensorOptions is_variable(bool b) { return *this; }
        TensorOptions device(DataType p, int a) { return *this; }
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
    void (*convolution3x3ToBasis)(const float *, torch::Tensor &colwiseResults, float **res, const int input_dim, const int result_dim))
{
	const DTYPE *inpPtr         = input.data<DTYPE>();
    const int batchSize         = input.size(0);
	const int inChannels        = input.size(1);
	const int inputDim          = input.size(2);
    const int inputSize         = inputDim * inputDim;
	const int resultDim         = inputDim - 2;  // Explanation: (I - K + 2*pad)/stride+1 =>_{stride=1} I - K + 2*pad + 1 =>_{K=4} I -3 + 2*pad 
    const int resultSize        = resultDim * resultDim;
    const int outChannels       = linCombs.size(0);

    // Create output
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false).device(torch::kCUDA, 0);
    auto resultTensor     = torch::empty({ batchSize, outChannels, resultSize }, options);
    
	for (int b_ix = 0; b_ix < batchSize; b_ix++)
	{
        int lastUsedBasisResult = 0;
        for (int c_ix = 0; c_ix < inChannels; c_ix++)
        {
            DTYPE *basisResults2d[9];
		    for (auto i = 0; i < 9; i++)
		    {
			    basisResults2d[i] = basisResultsTensor[lastUsedBasisResult].data<DTYPE>();
                ++lastUsedBasisResult;
		    }
            if (convolution3x3ToBasis != NULL)
                convolution3x3ToBasis(inpPtr, colwiseResults, basisResults2d, inputDim, resultDim);
            inpPtr += inputSize;
        }
       
        auto batchResults = torch::matmul_out(resultTensor[b_ix], linCombs, basisResultsTensor);
	}

    auto ret = resultTensor.view({ batchSize, outChannels, resultDim, resultDim });
    
    return resultTensor;
}

torch::Tensor conv_fwd_3x3(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor, torch::Tensor &colwiseResults)
{
    return forward(input_tensor, linCombs, basisResultsTensor, colwiseResults, NULL);
}

void nop(const float *, float *colResults[3], float **res, const int input_dim, const int result_dim) { }
torch::Tensor matmul_only1(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &basisResultsTensor, torch::Tensor &colwiseResults)
{
    return forward(input_tensor, linCombs, basisResultsTensor, colwiseResults, NULL);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_only1", &matmul_only1, "Perform only the second part of the convolution - MatMul");
	m.def("conv_fwd_3x3", &conv_fwd_3x3, "forward with 3x3 kernel");
}