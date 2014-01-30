//
// Configuration
//

enum scan_operation_t {
    SUM = 0,
    MIN,
    MAX
};

enum prescan_function_t {
    NONE = 0,
    SQRT
};


//
// Kernels
//

// TODO: replace with faster tree-based algorithm
//       http://stackoverflow.com/questions/11385475/scan-array-cuda
static __device__ void scan_array(float *temp, int index, int length,
                                  scan_operation_t operation) {
    int pout = 0, pin = 1;
    for (int offset = 1; offset < length; offset *= 2) {
        // Swap double buffer indices
        pout = 1 - pout;
        pin = 1 - pin;
        if (index >= offset) {
            switch (operation) {
            case SUM:
                temp[pout * length + index] =
                    temp[pin * length + index] +
                    temp[pin * length + index - offset];
                break;
            case MIN:
                temp[pout * length + index] =
                    fmin(temp[pin * length + index],
                         temp[pin * length + index - offset]);
                break;
            case MAX:
                temp[pout * length + index] =
                    fmax(temp[pin * length + index],
                         temp[pin * length + index - offset]);
                break;
            }
        } else {
            temp[pout * length + index] = temp[pin * length + index];
        }
        __syncthreads();
    }
    temp[pin * length + index] = temp[pout * length + index];
}

static __global__ void
prescan_kernel(const float *input, float *output,
               const prescan_function_t prescan_function) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    switch (prescan_function) {
    case SQRT:
        temp[row] = sqrt(input[row + col * rows]);
        break;
    case NONE:
    default:
        temp[row] = input[row + col * rows];
        break;
    }
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write back
    output[row + col * rows] = temp[rows + row];
}

static __global__ void findWeightedMedian_kernel(const float *input,
                                                 const float *prescan,
                                                 int *output) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = prescan[row + col * rows];
    __syncthreads();

    if (row > 0) {
        float threshold = temp[rows - 1] / 2;
        if (temp[row - 1] < threshold && temp[row] >= threshold)
            output[col] = row;
    }
}
