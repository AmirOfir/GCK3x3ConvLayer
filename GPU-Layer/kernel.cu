
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


__global__ void ConvolutionRowwise(const float *input, float *colwiseResults, int batch_ix,
    int channel_ix,
    int input_dim,
    int result_dim )
{
    float* res1 = colwiseResults + (blockIdx.x * result_dim);
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
            float *colwiseResults;
            int resultDim = inputDim - 2;
            int colwiseResultsSize = 3 * inputDim * resultDim;
            cudaMallocManaged(&colwiseResults, colwiseResultsSize);
            
            for (size_t i = 0; i < colwiseResultsSize; i++)
            {
                colwiseResults[i] = 1;
            }
            PrintMat(colwiseResults, inputDim, resultDim); cout << endl;
            PrintMat(colwiseResults + inputDim * resultDim, inputDim, resultDim); cout << endl;
            PrintMat(colwiseResults + (2 * (inputDim * resultDim)), inputDim, resultDim); cout << endl;

            float *arr = CreateArray(inputDim * inputDim);
            dim3 grid(inputDim);
            ConvolutionRowwise <<< grid, 1 >>> (arr, colwiseResults, batchIndex, inputChannel, inputDim, resultDim);
            cudaDeviceSynchronize();

            PrintMat(colwiseResults, inputDim, resultDim); cout << endl;
            PrintMat(colwiseResults + inputDim * resultDim, inputDim, resultDim); cout << endl;
            PrintMat(colwiseResults + (2 * (inputDim * resultDim)), inputDim, resultDim); cout << endl;
        });
    
    

    //for (auto& [a, b] : zip(batchSizes, inputChannels)) {






    return 0;
}
