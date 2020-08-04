#include "GCK-Conv-Method.cuh"

void Convolution3x3ToBasis(
    const DTYPE *input, /* Input channel*/
    DTYPE *colResults[3], /* 3 * input_dim * result_dim */
    DTYPE **res, /* 9 * result_dim * result_dim */
    const int input_dim,
    const int result_dim)
{

}

std::vector<float *> createColResults(int colwiseSize)
{
    float *a;
    float *b;
    float *c;
    cudaMalloc(&a, colwiseSize * sizeof(float));
    cudaMalloc(&b, colwiseSize * sizeof(float));
    cudaMalloc(&c, colwiseSize * sizeof(float));
    return { a,b,c };
}
