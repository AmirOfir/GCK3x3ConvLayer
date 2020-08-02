
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>

float *CreateInput(int input_dim, int flag)
{
    int input_size = input_dim * input_dim;
    return CreateArray(input_size, flag);
}

float *createInput(int imageW, int imageH)
{
    float *h_Input = new float[imageW * imageH];
    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }
    return h_Input;
}

int main()
{
    const int imageW = 3072;
    const int imageH = 3072;
    const int iterations = 16;
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));










    return 0;
}
