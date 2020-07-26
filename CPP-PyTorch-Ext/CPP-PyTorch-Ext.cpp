// CPP Implementation of GCK Convolution for tensors
// Written by Amir Ofir
#include "Header.h"
#include <future>
#include <fstream>
#include <cassert>

using namespace std;

#pragma region ThreadPool

#ifndef ThreadPool
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return move(res);
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}
#endif

#pragma endregion

void DeleteArray(DTYPE **arr, int outerArrayDim)
{
	int i;
	for (i = 0; i < outerArrayDim; i++)
	{
		delete[] arr[i];
	}
	delete[] arr;
}


/*
	Column-wise convolution with 4x1 GCK vectors: [1 1 -1 -1], [1 -1 -1 1], [1 -1 1 -1]

	For each row, calculates:
		For each col:
			Reads 4 cells
			Writes 1x4 conv for the 3 output matrices
*/
void GCKSeparableConvolution1x4_EachRow(const DTYPE *input, DTYPE **results, size_t row, const size_t input_row_width, const size_t output_row_width, const size_t cols_to_conv)
{
	auto input_cell = row * input_row_width;
	auto output_cell = row * output_row_width;
	for (auto col = 0; col < cols_to_conv; col++)
	{
		
		DTYPE a = input[input_cell] + input[input_cell + 1] - input[input_cell + 2] - input[input_cell + 3];
		DTYPE l = input[input_cell] - input[input_cell + 1];
		DTYPE r = input[input_cell + 2] - input[input_cell + 3];
		results[0][output_cell] = a;
		results[1][output_cell] = l - r;
		results[2][output_cell] = l + r;

		output_cell += 1;
		input_cell += 1;
	}
}
void GCKSeparableConvolution1x4(const DTYPE *input, DTYPE **results, const size_t input_rows, const size_t input_row_width, const size_t output_row_width, const size_t cols_to_conv)
{
	std::vector<std::future<void>> futures;
	
	for (auto row = 0; row < input_rows; ++row)
	{
		GCKSeparableConvolution1x4_EachRow(input, results, row, input_row_width, output_row_width, cols_to_conv);

		/*futures.push_back(
			threadPool.enqueue(
				GCKSeparableConvolution1x4_EachRow, input, results, row, input_row_width, output_row_width, cols_to_conv
			)
		);*/
	}
	
	for (auto& task : futures) task.wait();
}

/*
	Row-wise convolution with 1x4 GCK vectors: [1 1 1 1]^T [1 1 -1 -1]^T, [1 -1 -1 1]^T, [1 -1 1 -1]^T
	Expected the inputs 
	(by summing every 4 cells in a row for each of the 3*4 kernels)
*/
void GCKSeparableConvolution4x1_EachCol(DTYPE *input, const vector<DTYPE *> &results, const size_t input_row_width, const size_t col, const size_t result_row_width, const size_t result_rows)
{
	// Targets
	DTYPE *a = results[0]+col;
	DTYPE *b = results[1]+col;
	DTYPE *c = results[2]+col;
	DTYPE *d = results[3]+col;
	DTYPE vals[4];
	for (auto row = 0; row < result_rows; ++row)
	{
		for (int i = 0; i < 4; i++)
		{
			vals[i] = input[(row+i)*input_row_width + col];
		}
		*a = vals[0] + vals[1] + vals[2] + vals[3];
		*b = vals[0] + vals[1] - vals[2] - vals[3];
		*c = vals[0] - vals[1] - vals[2] + vals[3];
		*d = vals[0] - vals[1] + vals[2] - vals[3];

		a += result_row_width;
		b += result_row_width;
		c += result_row_width;
		d += result_row_width;
	}
}
vector<future<void>> GCKSeparableConvolution4x1(DTYPE *input, const vector<DTYPE *> &results, const size_t cols, const size_t result_dim, const size_t input_row_width)
{
	const size_t result_size = result_dim * result_dim;
	std::vector<std::future<void>> futures;
	
	for (auto col = 0; col < cols; col++)
	{
		GCKSeparableConvolution4x1_EachCol(input, results, input_row_width, col, result_dim, result_dim);
	}

	return futures;
}

/* 
	Convolution for the first kernel
*/
void GCKConvolutionFirstKernelNx1_Colwise(const DTYPE *input, DTYPE *rowConv, const size_t input_dim, const size_t kernel_dim, const size_t result_dim)
{
	DTYPE *keeps = new DTYPE[kernel_dim];
	size_t keepsCurr;

	DTYPE sum;
	DTYPE *rowConvCurr;
	
	int64_t i;
	size_t inputCurr;

	for (int64_t row = 0; row < input_dim; row++)
	{
		// Init variables to the beginning of the row
		rowConvCurr = rowConv + (row*result_dim);
		inputCurr = input_dim * row;
		sum = 0.f;
		keepsCurr = 0;

		// Fill the keeps array and set the sum to the beginning of the row
		for (i = 0; i < kernel_dim; i++)
		{
			sum += input[inputCurr];
			keeps[i] = input[inputCurr];
			inputCurr++;
		}

		// Set the results
		*rowConvCurr = sum;
		rowConvCurr++;

		for (i = kernel_dim; i < input_dim; i++)
		{
			// Reduce the old value
			sum -= keeps[keepsCurr];

			// Accumalate the new value
			sum += input[inputCurr];
			keeps[keepsCurr] = input[inputCurr];

			// Advance the pointers
			*rowConvCurr = sum;
			inputCurr++;
			rowConvCurr++;
			keepsCurr++;
			if (keepsCurr == kernel_dim)
				keepsCurr = 0;
		}
	}
	delete[] keeps;
}
void GCKConvolutionFirstKernel1xN_Rowwise(const DTYPE *rowConv, DTYPE *res, const size_t input_dim, const size_t input_size, const size_t result_dim, const size_t result_size, const size_t kernel_dim)
{
	// ---- Second part - col convolution
	size_t resIx;
	DTYPE *resCurr;
	const DTYPE *rowConvAddPtr = rowConv;

	// Compute first row of the output - Sum the top <kernel_dim> cells from the same column
	for (size_t row = 0; row < kernel_dim; ++row)
	{
		resCurr = res;
		for (resIx = 0; resIx < result_dim; resIx++)
		{
			if (row == 0)
				*resCurr = *rowConvAddPtr;
			else
				*resCurr += *rowConvAddPtr;
			++rowConvAddPtr;
			++resCurr;
		}
	}

	// Compute next rows of the output
	DTYPE *resPrevRowIx = res;
	resCurr = res + result_dim;
	size_t rowConvRemoveIx = 0; // index in the line to remove
	//size_t rowConvAddIx = (kernel_dim - 1) * result_dim; // index in the line to add
	for (resIx = result_dim; resIx < result_size; resIx++)
	{
		*resCurr = *resPrevRowIx + *rowConvAddPtr - rowConv[rowConvRemoveIx];
		++resPrevRowIx;
		++resCurr;
		++rowConvRemoveIx;
		++rowConvAddPtr;
	}
}
void GCKConvolutionFirstKernelColumn4x1_AddColwiseForEachRow(const DTYPE *rowConv, DTYPE **results, const size_t input_row_width, const size_t row, const size_t result_row_width)
{
	DTYPE *a, *b, *c; // Targets
	const DTYPE *inputCurr;
	register DTYPE curr;
	
	a = results[4]+(row*result_row_width);
	b = results[8]+(row*result_row_width);
	c = results[12]+(row*result_row_width);
	inputCurr = rowConv + (row*result_row_width);
	for (auto col = 0; col < result_row_width; ++col)
	{
		curr = *inputCurr;
		*a = curr;
		*b = curr;
		*c = curr;

		a++;
		b++;
		c++;
		++inputCurr;
	}

	a = results[4]+(row*result_row_width);
	b = results[8]+(row*result_row_width);
	c = results[12]+(row*result_row_width);
	inputCurr = rowConv + ((row+1)*result_row_width);
	for (auto col = 0; col < result_row_width; ++col)
	{
		curr = *inputCurr;
		*a += curr;
		*b += - curr;
		*c += - curr;

		a++;
		b++;
		c++;
		++inputCurr;
	}

	a = results[4]+(row*result_row_width);
	b = results[8]+(row*result_row_width);
	c = results[12]+(row*result_row_width);
	inputCurr = rowConv + ((row+2)*result_row_width);
	for (auto col = 0; col < result_row_width; ++col)
	{
		curr = *inputCurr;
		*a -= curr;
		*b -= curr;
		*c += curr;

		a++;
		b++;
		c++;
		++inputCurr;
	}

	a = results[4]+(row*result_row_width);
	b = results[8]+(row*result_row_width);
	c = results[12]+(row*result_row_width);
	inputCurr = rowConv + ((row+3)*result_row_width);
	for (auto col = 0; col < result_row_width; ++col)
	{
		curr = *inputCurr;
		*a -= curr;
		*b += curr;
		*c -= curr;

		a++;
		b++;
		c++;
		++inputCurr;
	}
/*
	for (auto col = 0; col < result_row_width; ++col)
	{
		for (int i = 0; i < 4; i++)
		{
			vals[i] = rowConv[i*input_row_width + col];
		}
		*a = vals[0] + vals[1] + vals[2] + vals[3];
		*b = vals[0] + vals[1] - vals[2] - vals[3];
		*c = vals[0] - vals[1] - vals[2] + vals[3];
		*d = vals[0] - vals[1] + vals[2] - vals[3];

		a++;
		b++;
		c++;
		d++;
	}*/
}
void GCKConvolutionFirstKernel4x4(const DTYPE *input, DTYPE **results, const size_t input_dim, const size_t input_size, const size_t result_dim, const size_t result_size)
{
	// This is a 2d matrix - index for the (row,col) item is col*result_dim + row
	DTYPE *rowConv = new DTYPE[result_dim * input_dim]; // <input_dim> rows by <result_dim> columns.

	GCKConvolutionFirstKernelNx1_Colwise(input, rowConv, input_dim, 4, result_dim);
	GCKConvolutionFirstKernel1xN_Rowwise(rowConv, results[0], input_dim, input_size, result_dim, result_size, 4);
	GCKConvolutionFirstKernelColumn4x1_AddColwiseForEachRow(rowConv, results, input_dim, 0, result_dim);
	GCKConvolutionFirstKernelColumn4x1_AddColwiseForEachRow(rowConv, results, input_dim, 1, result_dim);

	delete[] rowConv;
}

void GCKSequentialConvolution_Col(const DTYPE *last_convolved, DTYPE *res, const size_t result_dim, const int shared_prefix_len, const bool sign_pos)
{
	register size_t curr = 0;
	register size_t prev;
	
	// Calc next item sequentially
	prev = 0;
	curr = shared_prefix_len * result_dim;
	for (size_t x = 0; x < result_dim; ++x)
	{
		for (size_t y = shared_prefix_len; y < result_dim; ++y)
		{
			if (sign_pos)
			{
				res[curr] = res[prev] - last_convolved[curr] - last_convolved[prev];
			}
			else
			{
				res[curr] = - res[prev] - last_convolved[curr] + last_convolved[prev];
			}
			++curr;
			++prev;
		}
	}
}
void GCKSequentialConvolution_Row(const DTYPE *last_convolved, DTYPE *res, const size_t result_dim, const int shared_prefix_len, const bool sign_pos)
{
	size_t y;
	size_t x;
	register size_t curr;
	register size_t prev;
	
	// Calc next item sequentially
	for (y = 0; y < result_dim; ++y)
	{
		prev = y * result_dim;
		curr = prev + shared_prefix_len;
		for (x = shared_prefix_len; x < result_dim; ++x)
		{
			if (sign_pos)
			{
				res[curr] = res[prev] - last_convolved[curr] - last_convolved[prev];
			}
			else
			{
				res[curr] = - res[prev] - last_convolved[curr] + last_convolved[prev];
			}
			++curr;
			++prev;
		}
	}
}

void ConvolutionColwise(const DTYPE *input, DTYPE *res[3], const size_t input_dim, const size_t input_size, const size_t result_dim)
{
    DTYPE a, b, c;
    int reset = input_dim;
    DTYPE *first = res[0];
    DTYPE *second = res[1];
    DTYPE *third = res[2];

    a = input[0];
    b = input[1];
    c = input[2];
    for (size_t i = 3; i < input_size;++i)
    {
        *first = (a + b + c);
        ++first;
        *second = (a - b + c);
        ++second;
        *third = (a + b - c);
        ++third;

        if (i == reset)
        {
            reset += input_dim;
            a = input[i++];
            b = input[i++];
            c = input[i];
        }
        else 
        {
            a = b;
            b = c;
            c = input[i];
        }
    }

    // Last results
    *first = (a + b + c);
    *second = (a - b + c);
    *third = (a + b - c);
}
void ConvolutionRowwise(const DTYPE *input, DTYPE **res, const int kernel_column, const size_t result_dim)
{
    DTYPE *first = res[kernel_column];
    DTYPE *second = res[kernel_column + 3];
    DTYPE *third = res[kernel_column + 6];

    auto l1 = input; // First line of the input
    auto l2 = input + result_dim; // Second line of the input
    auto l3 = input + result_dim + result_dim; // Third line of the input
    DTYPE a, b, c;
    for (size_t row = 0; row < result_dim; row++)
    {
        for (size_t col = 0; col < result_dim; col++)
        {
            a = *l1;
            b = *l2;
            c = *l3;
            
            *first = (a + b + c);
            *second = (a - b + c);
            *third = (a + b - c);

            ++l1;
            ++l2;
            ++l3;
            ++first;
            ++second;
            ++third;
        }
    }

}

#ifdef _DEBUG
std::launch policy = std::launch::deferred;
#else
std::launch policy = std::launch::async;
#endif
void Convolution3x3ToBasis(const DTYPE *input, DTYPE **res, const size_t input_dim, const size_t result_dim)
{
    const size_t inputSize = input_dim * input_dim;
    const size_t colwise_size = input_dim * result_dim;
    DTYPE *colResults = new DTYPE[colwise_size * 3]; // input_dim rows, result_dim cols
    DTYPE *colResults1[3] = { colResults, colResults + colwise_size, colResults + colwise_size + colwise_size };
    ConvolutionColwise(input, colResults1, input_dim, inputSize, result_dim);

    auto a = async(policy , &ConvolutionRowwise, colResults1[0], res, 0, result_dim);
    auto b = async(policy, &ConvolutionRowwise, colResults1[1], res, 1, result_dim);
    auto c = async(policy, &ConvolutionRowwise, colResults1[2], res, 2, result_dim);
    a.wait();
	b.wait();
	c.wait();
    /*const int colwise_size = input_dim * result_dim;
    DTYPE **res = new DTYPE*[9];
    for (auto i = 0; i < 9; i++)
    {
        res[i] = new DTYPE[];
    }
    ConvolutionColwise(input, res, input_dim, result_dim)
	ConvolutionBeforeOffset_TopAndFirst(input, res, input_dim, result_dim, 4);
	ConvolutionBeforeOffset_Left(input, res, input_dim, result_dim, 4);
	GCKSequentialConvolution_FirstKernelColumn(res, 4, result_dim, shared_prefixes, sign_pos);

	auto a = async(std::launch::async, &GCKSequentialConvolution_KernelRow, res, 0, 4, result_dim, shared_prefixes, sign_pos);
	auto b = async(std::launch::async, &GCKSequentialConvolution_KernelRow, res, 1, 4, result_dim, shared_prefixes, sign_pos);
	auto c = async(std::launch::async, &GCKSequentialConvolution_KernelRow, res, 2, 4, result_dim, shared_prefixes, sign_pos);
	GCKSequentialConvolution_KernelRow(res, 3, 4, result_dim, shared_prefixes, sign_pos);
	a.wait();
	b.wait();
	c.wait();*/
}
