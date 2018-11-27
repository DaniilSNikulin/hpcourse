
#include <utils.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>


template <class T>
struct matrixConvCoreFunctor
{
  const int N, M;
  const T init_value;
  matrixConvCoreFunctor(size_t N, size_t M, T init_value) : N(N), M(M), init_value(init_value)
  {}

  bool is_index_valid(int row, int col)
  {
    return 0 <= row && row < N && 0 <= col && col < N;
  }

  T operator()(std::vector<T> const & input,
               std::vector<T> const & mask,
               size_t global_index)
  {
    int row = global_index / N;
    int col = global_index % N;
    T res = init_value;

    if ( is_index_valid(row, col) )
    {
      for (int mask_i = 0; mask_i < M; ++mask_i)
      {
        for (int mask_j = 0; mask_j < M; ++mask_j)
        {
          int input_row = (row + mask_i - M / 2);
          int input_col = (col + mask_j - M / 2);
          if ( is_index_valid(input_row, input_col) )
          {
            int input_idx = input_row * N + input_col;
            int mask_index = mask_i * M + mask_j;
            res += input.at(input_idx) * mask.at(mask_index);
          }
        }
      }
    }

    return res;
  }
};


template <class T>
bool test( std::vector<T> const & input,
           std::vector<T> const & mask,
           std::vector<T> & output,
           size_t N, size_t M)
{
  size_t const matrix_size = N * N;
  size_t const mask_size = M * M;
  size_t const block_size = 1;

  if (mask.size() < mask_size || input.size() < matrix_size || output.size() < matrix_size) {
    throw std::runtime_error("size of vectors a too small");
  }

  cl_helper::Initializer init_box({"../matrix_conv.cl"}, false);

  // allocate device buffer to hold message
  cl::Buffer dev_input(init_box.context,  CL_MEM_READ_ONLY,  sizeof(T) * matrix_size);
  cl::Buffer dev_output(init_box.context, CL_MEM_READ_ONLY,  sizeof(T) * matrix_size);
  cl::Buffer dev_mask(init_box.context,   CL_MEM_WRITE_ONLY, sizeof(T) * mask_size);

  init_box.queue.enqueueWriteBuffer(dev_input, CL_FALSE,
                                    0, sizeof(T) * matrix_size, &input[0]);
  init_box.queue.enqueueWriteBuffer(dev_mask, CL_FALSE,
                                    0, sizeof(T) * mask_size, &mask[0]);

  // load named kernel from opencl source
  cl::Kernel kernel(init_box.program, "matrix_conv");
  cl_helper::setKernelParameters(kernel,
                                 dev_input, dev_mask, dev_output,
                                 static_cast<cl_int>(N),
                                 static_cast<cl_int>(M));

  init_box.queue.enqueueNDRangeKernel(kernel,
                                      cl::NullRange,
                                      cl::NDRange(N, N),
                                      cl::NDRange(block_size, block_size));

  init_box.queue.enqueueReadBuffer(dev_output, CL_TRUE,
                                   0, sizeof(T) * matrix_size, &output[0]);
  init_box.queue.finish();

  matrixConvCoreFunctor<T> coreFunctor(N, M, 0);

  return cl_helper::check(input, mask, output, coreFunctor);
}


int main()
{
  std::vector<size_t> const N_array = {1024, 1024, 1, 31, 1023};
  std::vector<size_t> const M_array = {3, 9, 9, 9, 9};

  size_t const max_N = *std::max_element(N_array.begin(), N_array.end());
  size_t const max_M = *std::max_element(M_array.begin(), M_array.end());
  size_t const min_size = std::min(N_array.size(), M_array.size());

  std::vector<float> input(max_N * max_N);
  std::vector<float> output(max_N * max_N);
  std::vector<float> mask(max_M * max_M);

  for (size_t i = 0; i < min_size; ++i)
  {
    size_t const N = N_array[i];
    size_t const M = M_array[i];
    std::fill_n(input.begin(),  N, 1);
    std::fill_n(mask.begin(),   M, 1);
    std::fill(output.begin(), output.end(), 0);

    if ( !test(input, mask, output, N, M) )
    {
      std::cerr << "test " << i << " failed;"
        << "  N = " << N
        << "  M = " << M
        << "\n";
      return 0;
    }
  }


  std::cout << "all tests passed" << std::endl;



  return 0;
}


