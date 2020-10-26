#include "FastConv_Conv_Method.cuh"

#include <THC/THCNumerics.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THCUNN/SharedMem.cuh>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <iostream>


const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;
const int CUDA_NUM_THREADS = 1024;
const int TOTAL_SHARED_MEMORY_PER_BLOCK_BYTES = 49152;
const int TOTAL_SHARED_MEMORY_PER_BLOCK_FLOATS = TOTAL_SHARED_MEMORY_PER_BLOCK_BYTES / 4;
const int TOTAL_CELLS_POSSIBLE_PER_BLOCK = TOTAL_SHARED_MEMORY_PER_BLOCK_FLOATS / 3;
const int CONSTRAINT_MAX_ROWS_PER_BLOCK = std::min(TOTAL_CELLS_POSSIBLE_PER_BLOCK, CUDA_NUM_THREADS);

using namespace std;
void ThreadsAndBlocksRequiredForInput(int input_dim, int &rows_per_block, int &blocks)
{
    // One thread per input pixel.
    // thread X is the column, Y is the row. 
    // Therefore:
    // Blocks will have A rows, and each block will have 2 overlapping rows with its parent (except the first).
    // Also, block have shared memory (between its threads) that should cover that part of the input, and thec rowwise-results (in-place, i.e. 3*input size)
    int constraintByMaxThreadsPerBlock = CUDA_NUM_THREADS / input_dim;
    int constraintByMaxCellsInSharedMem = TOTAL_CELLS_POSSIBLE_PER_BLOCK / input_dim;

    rows_per_block = std::min(std::min(constraintByMaxCellsInSharedMem, constraintByMaxThreadsPerBlock), input_dim);
    blocks = input_dim / (rows_per_block-2);
}


int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
static int getGradParamsNumThreads(int batchSize){
    //warp per item in a batch, up to a maximum
    return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);    
}

///////////////////////////////////////////////////////////////////////
// ROW CONVOLUTION FILTER
///////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void ConvolutionRowwise(const T *input, T *rowwiseResults, 
    int batch_ix, int channel_ix, int input_dim, int input_size, int result_dim)
{
    extern __shared__ int s[];

    // Rowwise results
    int row = threadIdx.y * input_dim;
    int col = threadIdx.x;
    
    // Copy to shared memory
    T inputCellValue = input[col + row + (input_dim * (blockIdx.x * (blockDim.x-2)))];
    s[row + col] = inputCellValue;
    s[row + col +  + input_size] = inputCellValue;
    s[row + col +  + input_size + input_size] = inputCellValue;
    __syncthreads();
    return;
    // Compute rowwise-convolution into shared memory
    
    float* res1 = rowwiseResults + (row * input_dim) + col;
    float* res2 = res1 + (input_dim * input_dim);
    float* res3 = res2 + (input_dim * input_dim);
    input = input 
            /* Current batch */ + (batch_ix /* * num_channels */ * input_dim * input_dim) 
            /* Current input channel: */ + (channel_ix * input_dim * input_dim)
            /* Current Row: */ + (blockIdx.x * input_dim);
    float l1 = input[0],
        l2 = input[1],
        l3 = input[2];
    
    /*for (int i = 3; i < input_dim; ++i)
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
    }*/
    /*
    *res1 = (l1 + l2 + l3);
    *res2 = (l1 - l2 + l3);
    *res3 = (l1 + l2 - l3);
    */
}

template <typename T>
__global__ void ConvolutionColwise(const T *rowwiseResults, T *colwiseResults, int input_dim, int result_dim)
{
    // blockDim
    // Z tells us which rowwiseResults matrix to work on {0,1,2}
    // X tells us the rowwiseResults matrix top-row
    // Y tells us the rowwiseResults matrix col

    int topCell = (blockIdx.z *input_dim*result_dim) + (blockIdx.x * result_dim) + blockIdx.y;
    T l1 = rowwiseResults[topCell];
    T l2 = rowwiseResults[topCell + result_dim];
    T l3 = rowwiseResults[topCell + result_dim + result_dim];
    
    topCell = (blockIdx.z * result_dim * result_dim * 3) + (blockIdx.x * result_dim) + blockIdx.y;
    colwiseResults[topCell] = l1 + l2 + l3;
    topCell += result_dim * result_dim;
    colwiseResults[topCell] = l1 - l2 + l3;
    topCell += result_dim * result_dim;
    colwiseResults[topCell] = l1 + l2 - l3;
}

void Convolution3x3ToBasis(
    const torch::Tensor &input, /* Input channel*/
    torch::Tensor &rowwiseResults, /* 3 * input_dim * result_dim */
    torch::Tensor &basisResultsTensor, /* 9 * result_dim * result_dim */
    int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim,
    int input_size)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // One thread per input pixel.
    // thread X is the column, Y is the row. 
    // Therefore:
    // Blocks will have A rows, and each block will have 2 overlapping rows with its parent (except the first).
    // Also, block have shared memory (between its threads) that should cover that part of the input, and thec rowwise-results (in-place, i.e. 3*input size)
    int rowsPerBlock, numberOfBlocks;
    rowsPerBlock = std::min(CONSTRAINT_MAX_ROWS_PER_BLOCK / input_dim, input_dim);
    numberOfBlocks = input_dim / (rowsPerBlock-2);
    dim3 threads(input_dim, rowsPerBlock);
    dim3 blocks(numberOfBlocks);
    int sharedMemorySize = input_dim * rowsPerBlock * sizeof(float) * 3;
    // cout << input_dim << "," << rowsPerBlock << "," << numberOfBlocks << endl;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "Convolution3x3ToBasis", [&] {
        ConvolutionRowwise <<<blocks, threads, sharedMemorySize, stream>>>
        (
            input.data<float>(),
            rowwiseResults.data<float>(),
            batch_ix, 
            channel_ix,
            input_dim,
            input_size,
            result_dim
        );
    
    //ConvolutionColwise <<< grid_cols, block, 0, stream >>>
    //    (
    //        rowwiseResults.data<float>(),
    //       basisResultsTensor.data<float>(),
    //       input_dim,
    //       result_dim
    //    );
    
    });
  
    THCudaCheck(cudaGetLastError());
    THCudaCheck(cudaStreamSynchronize(stream));

}
