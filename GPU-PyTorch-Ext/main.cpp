
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#define DTYPE float
using namespace std;

#pragma region Data creation

std::default_random_engine randomGeneratorEngine;
std::uniform_real_distribution<DTYPE> randomGenerator;
// 0 - all zeros
// 1 - all ones
// 2 - incremental rize, start with 0
// 3 - incremental rise, start with 1
// 4 - random
// 5 - random int
DTYPE *CreateArray(int size, int flag)
{
    int i;
    DTYPE *input = new DTYPE[size];
	for (i = 0; i < size; ++i)
	{
		switch (flag)
		{
		case 0:
			input[i] = 0;
			break;
		case 1:
			input[i] = 1;
			break;
		case 2:
			input[i] = i;
			break;
		case 3:
			input[i] = i + 1;
			break;
		case 4:
			input[i] = randomGenerator(randomGeneratorEngine);
			break;
		case 5:
			input[i] = (int)(randomGenerator(randomGeneratorEngine) * 10);
            break;
        case 6:
            input[i] = i % 10;
            break;
		}
	}
	return input;
}
float *CreateInput(int input_dim, int flag)
{
    int input_size = input_dim * input_dim;
    return CreateArray(input_size, flag);
}
#pragma endregion

int main()
{
    const int image_dim = 3072;
    const int iterations = 16;

    int input_method = 0;

    auto h_Input = CreateInput(3072, input_method);










    return 0;
}
