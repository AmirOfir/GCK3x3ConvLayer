
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#include <random> // For kernel generation
#include <algorithm>
#include <list>

using namespace std;

#pragma region Cuda

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

__global__ void ConvolutionColwise(const float *rowwiseResults, float *colwiseResults, int inputDim, int resultDim)
{
    // blockDim
    // Z tells us which rowwiseResults matrix to work on {0,1,2}
    // X tells us the rowwiseResults matrix top-row
    // Y tells us the rowwiseResults matrix col

    int topCell = (blockIdx.z *inputDim*resultDim) + (blockIdx.x * resultDim) + blockIdx.y;
    float l1 = rowwiseResults[topCell];
    float l2 = rowwiseResults[topCell + resultDim];
    float l3 = rowwiseResults[topCell + resultDim + resultDim];
    
    topCell = (blockIdx.z * resultDim * resultDim * 3) + (blockIdx.x * resultDim) + blockIdx.y;
    colwiseResults[topCell] = l1 + l2 + l3;
    topCell += resultDim * resultDim;
    colwiseResults[topCell] = l1 - l2 + l3;
    topCell += resultDim * resultDim;
    colwiseResults[topCell] = l1 + l2 - l3;

}
#pragma endregion

#pragma region Misc
std::default_random_engine randomGeneratorEngine;
std::uniform_real_distribution<float> randomGenerator;
float *CreateArray(int size)
{
    int i;
    float *arr;
    cudaMallocManaged(&arr, size);

	for (i = 0; i < size; ++i)
	{
		arr[i] = i + 1;

        cout << arr[i] << " ";
			
		//arr[i] = (int)(randomGenerator(randomGeneratorEngine) * 10);
	}
    cout << endl;
	return arr;
}

void PrintMat(float *mat, int rows, int cols)
{
	for (int i = 0; i < rows; ++i)
	{
		cout << "[";
		for (int j = 0; j < cols - 1; ++j)
		{
			cout << mat[i*cols + j] << " ";
		}

		cout << mat[i*cols + (cols - 1)] << "]";
		
		if (i < rows - 1)
			cout << endl;
	}
	cout << endl;
}

template <typename Function>
void zip(const vector<int> &batchSizes, const vector<int> &inputChannels, const vector<int> &outputChannels, const vector<int> &inputDims,
    Function function)
{
    for (int batchSize : batchSizes)
        for (int inputChannel : inputChannels)
            for (int outputChannel : outputChannels)
                for (int inputDim : inputDims)
                    function(batchSize, inputChannel, outputChannel, inputDim);
}

#pragma endregion

int main()
{
    const vector<int> batchSizes = { 1 };
    const vector<int> inputChannels = { 1 };
    const vector<int> outputChannels = { 1 };
    const vector<int> inputDims = { 10 }; // 16, 32, 64, 128, 256, 512, 650, 1024, 1280, 1500

    std::cout << std::setfill('0') << std::setw(5) << std::fixed << std::setprecision(1);
    zip(batchSizes, inputChannels, outputChannels, inputDims,
        [](int batchIndex, int inputChannel, int outputChannel, int inputDim) {
            float *rowwiseResults;
            int resultDim = inputDim - 2;
            int rowwiseResultsSize = 3 * inputDim * resultDim;
            cudaMallocManaged(&rowwiseResults, rowwiseResultsSize);
            
            float *arr = CreateArray(inputDim * inputDim);
            dim3 grid(inputDim);
            ConvolutionRowwise <<< grid, 1 >>> (arr, rowwiseResults, batchIndex, inputChannel, inputDim, resultDim);
            cudaDeviceSynchronize();
            PrintMat(rowwiseResults, inputDim, resultDim); cout << endl;
            PrintMat(rowwiseResults + inputDim * resultDim, inputDim, resultDim); cout << endl;
            PrintMat(rowwiseResults + (2 * (inputDim * resultDim)), inputDim, resultDim); cout << endl;

            float *colResults;
            cudaMallocManaged <float>(&colResults, 9 * resultDim * resultDim);
            grid = dim3(resultDim, resultDim, 3);
            ConvolutionColwise <<< grid, 1 >>> (rowwiseResults, colResults, inputDim, resultDim);
            cudaDeviceSynchronize();

            for (int i = 0; i < 9; i++)
            {
                PrintMat(colResults + (i*resultDim*resultDim), inputDim, resultDim); cout << endl;
            }
            
            
        });
    
    

    //for (auto& [a, b] : zip(batchSizes, inputChannels)) {






    return 0;
}
