#ifndef VSA
#include <torch/extension.h>

#if __has_include("ATen/Parallel.h") && __has_include(<stdint.h>)
#include <ATen/Parallel.h>
#else
#include <ATen/ParallelOpenMP.h>
#endif



#endif

#include "Header.h"
#include "ConvImp.h"
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

    template<class _Fty, class... _ArgTypes>
	_NODISCARD inline void
        parallel_for(int begin, int end, int grain_size, _Fty&& _Fnarg) {}
}

int TORCH_EXTENSION_NAME = 5;
class Installer {
public: void def(const char *str, void *func, const char *str2) {}
};
Installer m;
#define PYBIND11_MODULE(s,m) void A(int s, Installer m)

#endif

// LinCombs shape: (out_channels, in_channels*9). Assumed to be contiguous. (Remember that the set is transposed
// rowwiseResults shape (in_channels*3, inputDim * resultDim). Assumed to be contiguous. values are discarded
// colwiseResults shape: (in_channels*9, H*W). Assumed to be contiguous. values are discarded
// Padding = {0,1}
torch::Tensor forward(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &rowwiseResults, torch::Tensor &colwiseResults, 
    bool execRowwiseConv, bool execColwiseConv, int padding)
{
    // ResultDim explanation: 
    // General formula: (I - K + 2*P)/S + 1.
    // Supporting stride=1,kernel=3, padding={0,1} => {padding=0: I-3+1 = I-2}, {padding=1: I-3+2+1=I}

	const torch::Tensor input = input_tensor.contiguous();
    auto options = torch::TensorOptions().dtype(torch::kF32).requires_grad(false);
    
	const int batchSize = input.size(0);
	const int inChannels = input.size(1);
	const int inputDim = input.size(2);
    const int inputSize = inputDim * inputDim;
    
	const int resultDim = padding ? inputDim : inputDim - 2;  
    const int rowwiseResultSize = inputDim * resultDim;
	const int colwiseResultSize = resultDim * resultDim;
	const int outChannels = linCombs.size(0);

    torch::Tensor resultTensor = torch::empty({ outChannels, colwiseResultSize }, options);

    DTYPE *basisResults = colwiseResults.data<DTYPE>();
	
	for (int b_ix = 0; b_ix < batchSize; b_ix++)
	{
        // Execute the rowwise convolution
        torch::parallel_for(0, inChannels, 0, [&](int64_t start, int64_t end) {
            DTYPE *inpPtr = input[b_ix][start].data<DTYPE>();
            DTYPE *rowwiseResultsCurr = rowwiseResults[start*3].data<DTYPE>();
            for (int64_t input_channel = start; input_channel < end; input_channel++)
            {
                DTYPE *rowwiseResArr[3] = { rowwiseResultsCurr, rowwiseResultsCurr += rowwiseResultSize, rowwiseResultsCurr += rowwiseResultSize };
                if (execRowwiseConv)
                    ConvolutionRowwise(inpPtr, rowwiseResArr, inputDim, inputSize, padding);
                inpPtr += inputSize;
            }
         });
        
        
        // Execute the colwise convolution (parallel option)
        torch::parallel_for(0, inChannels * 3, 0, [&](int64_t index, int64_t stop) {
            
            DTYPE *rowwiseResultsCurr = rowwiseResults[index].data<DTYPE>();
            DTYPE *colwiseResultsCurr = colwiseResults[index * 3].data<DTYPE>();
            for (; index < stop; ++index) 
            {
                DTYPE *colwiseResArr[3] = { colwiseResultsCurr, colwiseResultsCurr += colwiseResultSize, colwiseResultsCurr += colwiseResultSize };
                if (execColwiseConv && padding)
                    ConvolutionColwiseSingleCellPadding(rowwiseResultsCurr, colwiseResArr, resultDim, colwiseResultSize);
                else if (execColwiseConv)
                    ConvolutionColwise(rowwiseResultsCurr, colwiseResArr, resultDim);

                rowwiseResultsCurr += rowwiseResultSize;
            }
        });/**/

        /*
        // Execute the colwise convolution (non-parallel option)
        DTYPE *rowwiseResultsCurr = rowwiseResults[0].data<DTYPE>();
        DTYPE *colwiseResultsCurr = colwiseResults[0].data<DTYPE>();
        for (int64_t index = 0; index < inChannels * 3; ++index)
        {
            DTYPE *colwiseResArr[3] = { colwiseResultsCurr, colwiseResultsCurr += colwiseResultSize, colwiseResultsCurr += colwiseResultSize };
            if (execColwiseConv && padding)
                ConvolutionColwiseSingleCellPadding(rowwiseResultsCurr, colwiseResArr, resultDim, colwiseResultSize);
            else if (execColwiseConv)
                ConvolutionColwise(rowwiseResultsCurr, colwiseResArr, resultDim);

            rowwiseResultsCurr += rowwiseResultSize;
        }
        /**/
        


        /*DTYPE *lastUsedBasisResult = basisResults;
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
        }*/

        torch::Tensor resultTensorBatch = resultTensor[b_ix];
        torch::matmul_out(resultTensorBatch, linCombs, colwiseResults);
	}

    auto ret = resultTensor.view({ batchSize, outChannels, resultDim, resultDim });

    return ret;
}

torch::Tensor conv_fwd_3x3(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &rowwiseResults, torch::Tensor &colwiseResults, int padding)
{
    return forward(input_tensor, linCombs, rowwiseResults, colwiseResults, true, true, padding);
}

torch::Tensor matmul_only(const torch::Tensor &input_tensor, const torch::Tensor &linCombs, torch::Tensor &rowwiseResults, torch::Tensor &colwiseResults, int padding)
{
    return forward(input_tensor, linCombs, rowwiseResults, colwiseResults, false, false, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_only", &matmul_only, "Perform only the second part of the convolution - MatMul");
	m.def("conv_fwd_3x3", &conv_fwd_3x3, "forward with 3x3 kernel");
}
