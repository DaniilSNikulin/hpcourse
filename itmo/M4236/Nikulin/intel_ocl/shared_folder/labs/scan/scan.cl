#define TYPE int


kernel void scan_per_block(global TYPE * g_arr,
                           int arr_shift,
                           int N,
                           local TYPE * a)
{
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    int block_size = get_local_size(0);
    int bid = gid / block_size;

    global TYPE * arr = g_arr + arr_shift;
    global TYPE * sum_out = arr + N;

    a[tid] = arr[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = 1; shift < block_size; shift <<= 1)
    {
        TYPE cur = a[tid];
        TYPE prev = 0.0;
        if (tid - shift >= 0) {
            prev = a[tid - shift];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        a[tid] = cur + prev;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    arr[gid] = a[tid];

    if (tid == block_size - 1) {
       sum_out[bid] = a[tid];
    }
}


kernel void add_value_from_prev_block(global TYPE * g_arr, int arr_shift, int N)
{
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    int block_size = get_local_size(0);
    int bid = gid / block_size;

    if (gid >= N || bid == 0) {
        return;
    }

    global TYPE * arr = g_arr + arr_shift;
    global TYPE * sum_values = arr + N;
    arr[gid] += sum_values[bid-1];
}

