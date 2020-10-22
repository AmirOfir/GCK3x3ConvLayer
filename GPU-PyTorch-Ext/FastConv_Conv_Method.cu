#include "FastConv_Conv_Method.cuh"

#include <THC/THCNumerics.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THCUNN/SharedMem.cuh>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;
const int CUDA_NUM_THREADS = 1024;

int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
static int getGradParamsNumThreads(int batchSize){
    //warp per item in a batch, up to a maximum
    return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);    
}

/*
template <typename T>
__global__ void ConvolutionRowwise(const T *input, T *colwiseResults, int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim )
{
    T a,b,c;
    T* res1 = colwiseResults;
    T* res2 = res1 + (input_dim * result_dim);
    T* res3 = res2 + (input_dim * result_dim);
}
*/
__global__ void ConvolutionRowwise(const float *input, float *rowwiseResults, int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim )
{
    float* res1 = rowwiseResults + (blockIdx.x * result_dim);
    float* res2 = res1 + (input_dim * result_dim);
    float* res3 = res2 + (input_dim * result_dim);

    input = input + (blockIdx.x * input_dim);
    float l1 = input[0],
        l2 = input[1],
        l3 = input[2];
    for (int i = 3; i < input_dim; ++i)
    {
        *res1 = (l1 + l2 + l3);
        ++res1;
        *res2 = (l1 - l2 + l3);
        ++res2;
        *res3 = (l1 + l2 - l3);
        ++res3;
        l1 = l2;
        l2 = l3;
        l3 = input[i];
    }

    *res1 = (l1 + l2 + l3);
    *res2 = (l1 - l2 + l3);
    *res3 = (l1 + l2 - l3);
}

void Convolution3x3ToBasis(
    const torch::Tensor &input, /* Input channel*/
    torch::Tensor &colwiseResults, /* 3 * input_dim * result_dim */
    torch::Tensor &basisResultsTensor, /* 9 * result_dim * result_dim */
    int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // One thread per output value
    int nthreads = input_dim;
    int blocks = GET_BLOCKS(nthreads);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS);
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "Convolution3x3ToBasis", [&] {
    ConvolutionRowwise <<<grid, block, 0, stream>>>
        (
            input.data<float>(),
            colwiseResults.data<float>(),
            batch_ix, 
            channel_ix,
            input_dim,
            result_dim
        );
  });

  THCudaCheck(cudaGetLastError());
}
