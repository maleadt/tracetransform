//
// Configuration
//

// CUDA
#include <math_constants.h>


//
// Auxiliary
//

__device__ inline void swap(float &a, float &b) {
    // Alternative swap doesn't use a temporary register:
    // a ^= b;
    // b ^= a;
    // a ^= b;

    float tmp = a;
    a = b;
    b = tmp;
}


//
// Kernels
//

// TODO: properly support non-pow2 row lengths
//       http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
__global__ void sortBitonic_kernel(float *input, const int rows_input) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    if (row < rows_input)
        temp[row] = input[row + col * rows_input];
    else
        temp[row] = CUDART_INF_F;
    __syncthreads();

    // Sort
    for (unsigned int major = 2; major <= rows; major *= 2) {
        // Merge
        for (unsigned int minor = major / 2; minor > 0; minor /= 2) {
            // Find sorting partner
            unsigned int row2 = row ^ minor;

            // The threads processing the lowest row sort the array
            if (row2 > row) {
        if ((row & major) == 0) {
            // Sort ascending
            if (temp[row] > temp[row2]) {
                swap(temp[row], temp[row2]);
            }
        } else {
            // Sort descending
            if (temp[row] < temp[row2]) {
                swap(temp[row], temp[row2]);
            }
        }
            }

            __syncthreads();
    }
    }

    // Write back
    if (row < rows_input)
        input[row + col * rows_input] = temp[row];
}
