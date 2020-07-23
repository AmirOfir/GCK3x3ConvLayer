
#include "Header.h"

#include <iostream>
#include <chrono>
#include <math.h>
#include <random> // For kernel generation
#include <thread>
#include <cassert>

using namespace std;

#pragma region Displaying

template <class D>
void PrintNum(D n)
{
	cout << (n >= 0 ? " " : "") << n;
}
void PrintVec(vector<int> a)
{
	cout << "[";
	for (int i = 0; i < a.size(); i++)
	{
		PrintNum((DTYPE)a[i]);
		cout << " ";
	}
	cout << "]";
}
void PrintVec(int *a, int size)
{
	auto b = vector<int>(a, a +size);
	PrintVec(b);
}

void PrintMat(bool *mat, int dim)
{
	for (int i = 0; i < dim; ++i)
	{
		cout << "[";
		for (int j = 0; j < dim - 1; ++j)
		{
			bool b = mat[i*dim + j];
			cout << (b ? "true " : "false");
			cout << " ";
		}

		bool b = mat[i*dim + (dim - 1)];
		cout << (b ? "true " : "false");
		cout << "]";
			
		
		if (i < dim - 1)
			cout << endl;
	}
	cout << endl;
}
template <typename D> void PrintMat(D *mat, int rows, int cols)
{
	for (int i = 0; i < rows; ++i)
	{
		cout << "[";
		for (int j = 0; j < cols - 1; ++j)
		{
			D v = mat[i*cols + j];
			PrintNum(v);
			cout << " ";
		}
		PrintNum(mat[i*cols + (cols - 1)]);
		cout << "]";
			
		
		if (i < rows - 1)
			cout << endl;
	}
	cout << endl;
}
template <typename D> void PrintMat(D *mat, int dim)
{
	PrintMat(mat, dim, dim);
}

template void PrintMat(DTYPE *mat, int rows, int cols);
template void PrintMat(DTYPE *mat, int dim);
template void PrintMat(int *mat, int dim);


#pragma endregion

#pragma region GCK Basis creation

int CommonPrefix(int *mat1, int *mat2, int mat_dim, int axis)
{
	if (axis == 0)
		for (int i = 0; i < mat_dim; i++)
			if (mat1[i] != mat2[i])
				return i;
	if (axis == 1)
		for (int i = 0; i < mat_dim; i++)
			if (mat1[i*mat_dim] != mat2[i*mat_dim])
				return i;
	return 0;
}
vector<int> ConcatenateTwice(vector<int> a, bool minus)
{
	vector<int> ret;
	for (int j = 0; j < 2; j++)
	{
		for (int i = 0; i < a.size(); i++)
		{
			ret.push_back( (j==1&&minus ? -1 : 1) * a[i]);
		}
	}
	return ret;
}

void OuterProduct(int *a, int *b, int *ret, int sizeA, int sizeB)
{
	//DTYPE *ret = new DTYPE[a.size()*b.size()];
	int k = 0;
	for (int i = 0; i < sizeA; i++)
	{
		for (int j = 0; j < sizeB; j++)
		{
			ret[k++] = DTYPE(a[i] * b[j]);
		}
	}
}
DTYPE *OuterProductFlipA(vector<int> a, vector<int> b)
{
	DTYPE *ret = new DTYPE[a.size()*b.size()];
	int k = 0;
	for (size_t i = a.size() - 1; i >= 0; i--)
	{
		for (size_t j = 0; j < b.size(); j++)
		{
			ret[k++] = DTYPE(a[i]) * b[j];
		}
	}
	return ret;
}

int **ConstructGCKBasisVectors(int kernelDim)
{
	int treeDepth =  int(log2(kernelDim));

	// Find the vectors that construct the matrix
	vector<vector<int>> r{ vector<int>{1} };
	
	for (int i = 0; i < treeDepth; ++i)
	{
		vector<vector<int>> tmp;
		for (int j = 0; j < r.size(); j++)
		{
			vector<int> k = r[j];
			tmp.push_back(ConcatenateTwice(k, !(j % 2 == 0)));
			tmp.push_back(ConcatenateTwice(k, (j % 2 == 0)));
		}
		r = tmp;
	}

	int **ret = new int*[kernelDim];
	for (size_t i = 0; i < kernelDim; i++)
	{
		ret[i] = new int[kernelDim];
		std::copy(r[i].begin(), r[i].end(), ret[i]);
	}
	return ret;
}
void ConstructGCKBasisMatricesRow(int **basis, vector<int> &offsets, vector<bool> &positive_signs, int row, int size, int **vectors, int *lastRowHeader, int startIndex)
{
	int k = startIndex;
	for (int j = 0; j < size; j++)
	{
		// Matrix
		OuterProduct(vectors[row], vectors[j], basis[k], size, size);
		int commonPrefix;
		bool positiveSign;

		if (j != 0) // same row
		{
			commonPrefix = CommonPrefix(basis[k], basis[k - 1], size, 0); 
			int compareLocation = commonPrefix;
			positiveSign = basis[k-1][compareLocation] == -1 && basis[k][compareLocation] == 1;
		}
		else if (j == 0 && row != 0) // same col
		{
			commonPrefix = CommonPrefix(basis[k], lastRowHeader, size, 1);
			positiveSign = lastRowHeader[commonPrefix * size] == -1 && basis[k][commonPrefix * size] == 1;
		} 
		else
		{
			commonPrefix = 0;
			positiveSign = true;
		}
		offsets[k] = commonPrefix;
		positive_signs[k] = positiveSign;
		++k;
	}
}

int **GckBasis(int size, vector<int> &offsets, vector<bool> &positive_signs, bool verbose = false)
{
	auto r = ConstructGCKBasisVectors(size);

	// Calculate the basis matrices
	int basisSize = size*size;
	int **basis = new int*[basisSize];
	for (size_t i = 0; i < basisSize; i++)
		basis[i] = new int[basisSize];
	
    offsets.clear();
    offsets.resize(basisSize);
    positive_signs.clear();
    positive_signs.resize(basisSize);

	int *lastRowHeader = NULL;
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		ConstructGCKBasisMatricesRow(basis, offsets, positive_signs, i, size, r, lastRowHeader, k);

		k += size;
		lastRowHeader = basis[k-1];
	}

	if (verbose)
	{
		cout << "Basis expected shape: (" << size << "," << size << ")" << endl;
		for (int i = 0; i < size; i++)
		{
			PrintVec(r[i], size);
			cout << endl;
		}
		for (int i = 0; i < basisSize; ++i)
		{
			cout << "Basis #" << (i + 1) << ":" << endl;
			PrintMat(basis[i], size);
		}
	}

	return basis;
}
int **GckBasis(int size, bool verbose = false)
{
    vector<int> offsets;
    vector<bool> positive_signs;
	int **ret = GckBasis(size, offsets, positive_signs, verbose);
	return ret;
}

#pragma endregion

#pragma region Data Creation

std::default_random_engine randomGeneratorEngine;
std::uniform_real_distribution<DTYPE> randomGenerator;

float round_precision(float a, int precision)
{
    return std::round(a * pow(10, precision)) / pow(10, precision);
}
bool nearly_equal(float a, float b, int precision=4, float epsilon = 1e-3)
  // those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<float>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b) return true;
  a = round_precision(a, precision);
  b = round_precision(b, precision);
  return abs(a - b) < epsilon;
}

bool MatricesEqual(DTYPE *mat1, DTYPE *mat2, int dim)
{
	for (int i = 0; i < dim * dim; i++)
	{
		if (!nearly_equal(mat1[i], mat2[i]))
		{
			cout << "Matrices are not equal at (" << (i / dim) << ", " << (i % dim) << "). diff: " << (double)abs((double)(mat1[i] - mat2[i])) << endl;
			return false;
		}
	}
	return true;
}

int ResultDim(int dim, int kernelDim, int pad, int stride)
{
	return floor((dim - kernelDim + (2 * pad)) / stride) + 1;
}

void DeleteKernels(DTYPE **kernels, int kernels_num)
{
	for (int i = 0; i < kernels_num; ++i)
	{
		delete[] kernels[i];
	}
	delete[] kernels;
}
template <class D> void DeleteArray(D **arr, int outerArrayDim)
{
	int i;
	for (i = 0; i < outerArrayDim; i++)
	{
		delete[] arr[i];
	}
	delete[] arr;
}
template void DeleteArray(DTYPE **arr, int outerArrayDim);
template void DeleteArray(int **arr, int outerArrayDim);

DTYPE **CreateMatrix(int matrices_count, int matrix_size, DTYPE val)
{
	DTYPE **ret = new DTYPE*[matrices_count];
	register DTYPE *ri;
	register int i, j;
	for (i = 0; i < matrices_count; i++)
	{
		ri = ret[i] = new DTYPE[matrix_size];
		for (j = 0; j < matrix_size; j++)
		{
			*ri = val;
			++ri;
		}
	}
	return ret;
}
DTYPE **CreateRandomMatrix(int a, int b)
{
	DTYPE **ret = new DTYPE*[a];
	for (size_t i = 0; i < a; i++)
	{
		register DTYPE *ri = ret[i] = new DTYPE[b];
		for (size_t j = 0; j < b; j++)
		{
			*ri = randomGenerator(randomGeneratorEngine);
			++ri;
		}
	}
	return ret;
}

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
		}
	}
	return input;
}

DTYPE *CreateInput(int input_dim, int flag)
{
    int input_size = input_dim * input_dim;
    return CreateArray(input_size, flag);
}

// Creates random kernels
DTYPE **CreateKernels(int kernels_num, int kernel_dim)
{
	DTYPE **kernels = new DTYPE*[kernels_num];
	for (int i = 0; i < kernels_num; ++i)
	{
		kernels[i] = new DTYPE[kernel_dim*kernel_dim];
		for (int a = 0; a < kernel_dim; ++a)
			for (int b = 0; b < kernel_dim; ++b)
				kernels[i][a*kernel_dim + b] = randomGenerator(randomGeneratorEngine);
	}
	return kernels;
}
DTYPE ***CreateKernels(int in_channels, int out_channels, int kernel_field_size)
{
	DTYPE **c = CreateRandomMatrix(in_channels*out_channels, kernel_field_size);
	DTYPE ***ret = new DTYPE**[in_channels];
	for (size_t i = 0; i < in_channels; i++)
	{
		ret[i] = c + (i * out_channels);
	}
	return ret;
}

#pragma endregion

#pragma region Regular Convolution

template <class KERNEL_TYPE> DTYPE **Conv2dCpu(DTYPE *input, KERNEL_TYPE **kernels, const int kernelsCount, const int input_dim, const int kernel_dim, 
	const int pad, const int stride, const int result_dim, const int result_size)
{
	// One result for each kernel
	DTYPE **result = new DTYPE*[kernelsCount]; 

	// Variables
	DTYPE *kernelResult;
	int kernelIx;
	int outputY;
	int outputX;
	int kernelCurr;
	int inputCurrY;
	int inputCurrX;
	int k1;
	int k2;
	DTYPE sum;

	// Computed loop vars
	int resultSquared = result_dim * result_dim;
	int innerLoopStart = -1 * pad;

	for (kernelIx = 0; kernelIx < kernelsCount; kernelIx++)
	{
		kernelResult = new DTYPE[resultSquared];
		result[kernelIx] = kernelResult;

		for (outputY = 0; outputY < result_dim; outputY++)
		{
			for (outputX = 0; outputX < result_dim; outputX++)
			{
				inputCurrY = innerLoopStart + (stride * outputY);
				inputCurrX = innerLoopStart + (stride * outputX);
				kernelCurr = 0;
				sum = 0.0;
				for (k1 = 0; k1 < kernel_dim; k1++)
				{
					for (k2 = 0; k2 < kernel_dim; k2++)
					{
						if (inputCurrX >= 0 && inputCurrY >= 0)
							sum += kernels[kernelIx][kernelCurr] * input[inputCurrY*input_dim + inputCurrX];
						inputCurrX++;
						kernelCurr++;
					}
					inputCurrY++;
					inputCurrX = innerLoopStart + (stride * outputX);
				}

				kernelResult[outputY * result_dim + outputX] = sum;
			}
		}
	}
	return result;
}

template DTYPE **Conv2dCpu(DTYPE *input, int **kernels, int kernelsCount, int input_dim, int kernel_dim, int pad, int stride, int result_dim, int result_size);
template DTYPE **Conv2dCpu(DTYPE *input, DTYPE **kernels, int kernelsCount, int input_dim, int kernel_dim, int pad, int stride, int result_dim, int result_size);

DTYPE **Conv2dCpuPad(DTYPE *input, DTYPE **kernels, int kernelsCount, int input_dim, int kernel_dim, int padTopLeft, int padBottomRight, int result_dim, int result_size)
{
	// One result for each kernel
	DTYPE **result = new DTYPE*[kernelsCount]; 

	// Variables
	DTYPE *kernelResult;
	int kernelIx;
	int outputY;
	int outputX;
	int inputCurrY;
	int inputCurrX;
	int kernelCurr;
	int kernelY;
	int kernelX;
	DTYPE sum;

	// Computed loop vars
	int resultSquared = result_dim * result_dim;
	
	for (kernelIx = 0; kernelIx < kernelsCount; kernelIx++)
	{
		kernelResult = new DTYPE[resultSquared];
		result[kernelIx] = kernelResult;

		for (outputY = 0; outputY < result_dim; outputY++)
		{
			for (outputX = 0; outputX < result_dim; outputX++)
			{
				inputCurrY = outputY - padTopLeft;
				inputCurrX = outputX - padTopLeft;
				kernelCurr = 0;
				sum = 0.0;
				for (kernelY = 0; kernelY < kernel_dim; kernelY++)
				{
					if (inputCurrY < 0 || inputCurrY > input_dim)
					{
						kernelCurr += kernel_dim;
					}
					else
					{
						for (kernelX = 0; kernelX < kernel_dim; kernelX++)
						{
							if (inputCurrX >= 0 && inputCurrX < input_dim)
								sum += kernels[kernelIx][kernelCurr] * input[inputCurrY*input_dim + inputCurrX];
							inputCurrX++;
							kernelCurr++;
						}
					}
					
					inputCurrY++; // Next row on the input 
					inputCurrX = outputX - padTopLeft; // Begining of the row relative to the kernel
				}

				kernelResult[outputY * result_dim + outputX] = sum;
			}
		}
	}
	return result;
}

#pragma endregion



int main()
{
	

	
    //std::cout << "Hello World!\n" << a(); 
}