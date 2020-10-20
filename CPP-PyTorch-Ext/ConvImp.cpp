

// CPP Implementation of GCK Convolution for tensors
// Written by Amir Ofir
#include "Header.h"
#include "ConvImp.h"
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

void ConvolutionRowwise(const DTYPE *input, DTYPE *res[3], const size_t input_dim, const size_t input_size, bool singleCellPadding)
{
    DTYPE a, b, c;
    int reset = input_dim;
    DTYPE *first = res[0];
    DTYPE *second = res[1];
    DTYPE *third = res[2];
    size_t i;
    if (singleCellPadding)
    {
        a = 0;
        b = input[0];
        c = input[1];
        i = 2;
    }
    else
    {
        a = input[0];
        b = input[1];
        c = input[2];
        i = 3;
    }
    for (; i < input_size;++i)
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
            if (singleCellPadding)
            {
                // Manual calculation
                *first = (b + c);
                ++first;
                *second = (-b + c);
                ++second;
                *third = (b - c);
                ++third;

                // Advance current cell
                a = 0;
                b = input[i++];
                c = input[i];
            }
            else
            {
                a = input[i++];
                b = input[i++];
                c = input[i];
            }
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

void ConvolutionColwiseSingleCellPadding(const DTYPE *input, DTYPE *res[3], const size_t result_dim)
{
    DTYPE *first = res[0];
    DTYPE *second = res[1];
    DTYPE *third = res[2];

    const DTYPE *line1 = input; // First line of the input (will be used for the second output row)
    const DTYPE *line2 = input; // First line of the input 
    const DTYPE *line3 = input + result_dim; // Second line of the input
    DTYPE a, b, c;
    size_t col, row;
    // For the first output
    for (col = 0; col < result_dim; col++)
    {
        b = *line2;
        c = *line3;
            
        *first = (b + c);
        *second = (- b + c);
        *third = (b - c);

        ++line2;
        ++line3;

        ++first;
        ++second;
        ++third;
    }

    // For the following output rows
    for (row = 1; row < result_dim - 1; row++)
    {
        for (col = 0; col < result_dim; col++)
        {
            a = *line1;
            b = *line2;
            c = *line3;
            
            *first = (a + b + c);
            *second = (a - b + c);
            *third = (a + b - c);

            ++line1;
            ++line2;
            ++line3;

            ++first;
            ++second;
            ++third;
        }
    }

    // For the last row
    for (col = 0; col < result_dim; col++)
    {
        a = *line1;
        b = *line2;
            
        *first = (a + b);
        *second = (a - b);
        *third = (a + b);

        ++line1;
        ++line2;

        ++first;
        ++second;
        ++third;
    }
}
void ConvolutionColwise(const DTYPE *input, DTYPE *res[3], const size_t result_dim)
{
    DTYPE *first = res[0];
    DTYPE *second = res[1];
    DTYPE *third = res[2];

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
    ConvolutionRowwise(input, colResults1, input_dim, inputSize, false);

    DTYPE *colwiseResults[3][3] = { {res[0], res[1], res[2]}, {res[3], res[4], res[5]}, {res[6], res[7], res[8]} };
    auto a = async(policy, &ConvolutionColwise, colResults1[0], colwiseResults[0], result_dim);
    auto b = async(policy, &ConvolutionColwise, colResults1[1], colwiseResults[1], result_dim);
    auto c = async(policy, &ConvolutionColwise, colResults1[2], colwiseResults[2], result_dim);
    a.wait();
	b.wait();
	c.wait();
}
