//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CUDAHELPER_MEMORY_
#define _TRACETRANSFORM_CUDAHELPER_MEMORY_

// Standard library
#include <cstddef>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Local
#include "errorhandling.h"


//
// Module definitions
//

namespace CUDAHelper
{
    // Abstraction for Cuda memory management
    // Manages allocation and transfer of memory
    template<typename MemType>
    class Memory {
    public:
        Memory() :
            gpuPtr(0),
            size(0)
        {
        }

        explicit Memory(std::size_t size) :
            gpuPtr(allocate(size)),
            size(size)
        {
        }
        
        Memory(Memory& other) :
            gpuPtr(other.gpuPtr),
            size(other.size)
        {
            other.gpuPtr = 0;
            other.size = 0;
        }

        ~Memory() {
            free();
        }

        void reallocate(std::size_t newSize) {
            MemType *newPtr = allocate(newSize);
            free();
            gpuPtr = newPtr;
            size = newSize;
        }

        void transferTo(MemType* hostPtr) const {
            checkError(
                cudaMemcpy(hostPtr, gpuPtr, sizeInBytes(), cudaMemcpyDeviceToHost)
            );
        }

        void transferFrom(const MemType* hostPtr) {
            checkError(
                cudaMemcpy(gpuPtr, hostPtr, sizeInBytes(), cudaMemcpyHostToDevice)
            );
        }

        const MemType* get() const {
            return gpuPtr;
        }

        MemType* get() {
            return gpuPtr;
        }

    private:
        Memory(const Memory&) { }
        Memory& operator=(const Memory&) { }

        std::size_t sizeInBytes() const {
            return size * sizeof(MemType);
        }

        MemType* allocate(std::size_t size) {
            MemType* ptr;
            checkError(cudaMalloc(&ptr, size * sizeof(MemType)));
            return ptr;
        }

        void free() {
            if (gpuPtr != 0)
                cudaFree(gpuPtr);
        }

        MemType* gpuPtr;
        std::size_t size;
    };
}

#endif
