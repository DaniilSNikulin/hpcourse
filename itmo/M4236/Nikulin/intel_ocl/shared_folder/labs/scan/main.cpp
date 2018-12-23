
#include <utils.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>


template <class T>
std::vector<T> cpu_scan(std::vector<T> const inp)
{
  std::vector<T> outp = inp;
  for (size_t i = 1; i < inp.size(); i++) {
    outp[i] += outp[i-1];
  }
  return outp;
}


template <class T>
struct ScanCoreFunciton
{
  const std::vector<T> expected;
  ScanCoreFunciton(std::vector<T> const & inp) : expected(cpu_scan(inp))
  {}

  T operator()(std::vector<T> const & input, size_t global_index)
  {
    return expected[global_index];
  }
};



int computeUpperN( int N, int block_size )
{
  while ( N % block_size )
  {
    ++N;
  }
  return N;
}



template <class T>
void scan_recursivle_backend(cl_helper::Initializer & init_box,
                             cl::Kernel & kernel_scan,
                             cl::Kernel & kernel_add,
                             cl::Buffer & dev_data,
                             int block_size,
                             int N,
                             int shift)
{
  if (N == 1) {
    return;
  }

  int n_blocks = (N + (block_size - 1)) / block_size;

  cl_helper::setKernelParameters(kernel_scan,
                                 dev_data,
                                 static_cast<cl_int>(shift),
                                 static_cast<cl_int>(N),
                                 cl::Local(sizeof(T) * block_size));
  init_box.queue.enqueueNDRangeKernel(kernel_scan,
                                      cl::NullRange,
                                      cl::NDRange(computeUpperN(N, block_size)),
                                      cl::NDRange(block_size));
  init_box.queue.finish();

  scan_recursivle_backend<T>(init_box, kernel_scan, kernel_add, dev_data,
                             block_size, n_blocks, shift + N);

  cl_helper::setKernelParameters(kernel_add,
                                 dev_data,
                                 static_cast<cl_int>(shift),
                                 static_cast<cl_int>(N));
  init_box.queue.enqueueNDRangeKernel(kernel_add,
                                      cl::NullRange,
                                      cl::NDRange(computeUpperN(N, block_size)),
                                      cl::NDRange(block_size));
  init_box.queue.finish();
}


template <class T>
std::vector<T> scan(cl_helper::Initializer & init_box,
                    const size_t block_size,
                    std::vector<T> const & input)
{
  // allocate device buffer to hold message
  int half_size = computeUpperN(input.size(), block_size);
  int buffer_size = half_size * 2;
  cl::Buffer dev_data(init_box.context, CL_MEM_READ_WRITE, sizeof(T) * buffer_size);

  // copy from cpu to gpu
  std::vector<T> tmp = input;
  tmp.resize(buffer_size, 0);
  init_box.queue.enqueueWriteBuffer(dev_data, CL_TRUE,
                                    0, sizeof(T) * tmp.size(),
                                    &tmp[0]);

  // load named kernel from opencl source
  cl::Kernel kernel_scan(init_box.program, "scan_per_block");
  cl::Kernel kernel_add(init_box.program, "add_value_from_prev_block");

  scan_recursivle_backend<T>(init_box, kernel_scan, kernel_add, dev_data,
                             block_size, half_size, 0);

  std::vector<T> output(input.size(), 0);
  init_box.queue.enqueueReadBuffer(dev_data, CL_TRUE,
                                   0, sizeof(T) * input.size(),
                                   &output[0]);
  return output;
}


int main()
{
  cl_helper::Initializer init_box({"../scan.cl"}, false);
  Timer<> timer;

  using Type = int;
  // create a message to send to kernel
  double percentile = 0;
  for (size_t test_array_size = 1; test_array_size < 1048576; test_array_size = test_array_size * 1.01 + 1)
  {
    for (size_t block_size = 4; block_size < 1024; block_size *= 2)
    {
      if (test_array_size / block_size > 100) {
        // limitation of my computer
        continue;
      }

      std::vector<Type> input(test_array_size, 0);
      Type max_value = 10;
      for (size_t i = 0; i < test_array_size; ++i)
      {
        input[i] = (rand() % (int) max_value) + ((float)rand() / std::numeric_limits<int>::max());
      }

      std::vector<Type> output = scan(init_box, block_size, input);
      ScanCoreFunciton<Type> coreFunctor(input);
      if (!cl_helper::check(input, output, coreFunctor, test_array_size))
      {
        std::cout
          << "ERROR:"
          << "\n block_size = " << block_size
          << "\n test_array_size = " << test_array_size
          << std::endl;
        return 0;
      }

      double local_percentile = (double)test_array_size / (1048576) * 100;
      if (local_percentile - percentile > 1) {
        percentile = local_percentile;
        std::cout << percentile << "%\n";
      }
    }
  }

  double elapsed_time = timer.elapsed();

  std::cout << std::setprecision(2)
    << "Total time: " << elapsed_time << " sec"
    << std::endl;
}

