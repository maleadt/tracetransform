//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CUDAHELPER_CHRONO_
#define _TRACETRANSFORM_CUDAHELPER_CHRONO_

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// Local
#include "errorhandling.hpp"


//
// Module definitions
//

namespace CUDAHelper
{
        class Chrono
        {
        public:
                Chrono()
                {
                        checkError(cudaEventCreate(&_start));
                        checkError(cudaEventCreate(&_stop));
                }

                ~Chrono()
                {
                        checkError(cudaEventDestroy(_start));
                        checkError(cudaEventDestroy(_stop));
                }

                void start()
                {
                        checkError(cudaEventRecord(_start, 0));
                }

                void stop()
                {
                        checkError(cudaEventRecord(_stop, 0));
                        checkError(cudaEventSynchronize(_stop));
                }

                float elapsed() const
                {
                        float elapsedTime;
                        checkError(cudaEventElapsedTime(&elapsedTime, _start, _stop));
                        return elapsedTime;
                }

        private:
                cudaEvent_t _start, _stop;
        };
}

#endif
