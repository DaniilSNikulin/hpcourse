
bool is_index_valid(int row, int col, int N)
{
  return 0 <= row && row < N && 0 <= col && col < N;
}


kernel void matrix_conv(global float * input,
                        global float * mask,
                        global float * output,
                        int N, int M)
{
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= N || col >= N)
      return;

   int res = 0;

   for (int mask_i = 0; mask_i < M; ++mask_i)
   {
     for (int mask_j = 0; mask_j < M; ++mask_j)
     {
       int input_row = (row + mask_i - M / 2);
       int input_col = (col + mask_j - M / 2);
       if ( is_index_valid(input_row, input_col, N) )
       {
         int input_idx = input_row * N + input_col;
         int mask_index = mask_i * M + mask_j;
         res += input[input_idx] * mask[mask_index];
       }
     }
   }

   output[row * N + col] = res;
}

