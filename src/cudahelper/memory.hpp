//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CUDAHELPER_MEMORY_
#define _TRACETRANSFORM_CUDAHELPER_MEMORY_

// Standard library
#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cassert>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Local
#include "../logger.hpp"
#include "errorhandling.hpp"


//
// Module definitions
//

namespace CUDAHelper
{
        // Abstraction for CUDA memory management
        // Manages allocation and transfer of memory

        template<typename MemType>
        class Memory
        {
        public:
                explicit Memory(std::size_t size)
                                : _size(size)
                {
                }

                std::size_t size() const
                {
                        return _size;
                }

        protected:
                std::size_t bytes() const
                {
                        return _size * sizeof(MemType);
                }

        private:
                Memory(const Memory&);
                Memory& operator=(const Memory&);

                std::size_t _size;
        };

        template<typename MemType>
        class HostMemory: public Memory<MemType>
        {
        public:
                HostMemory(std::size_t size)
                                : Memory<MemType>(size)
                {
                        checkError(cudaHostAlloc(&_hostPtr, this->bytes(), cudaHostAllocDefault));
                }

                HostMemory(const HostMemory<MemType>& other)
                {
                        assert(this->size() == other.size());
                        checkError(
                                        cudaMemcpy(_hostPtr, other._hostPtr,
                                                        this->bytes(),
                                                        cudaMemcpyHostToHost));
                }

                ~HostMemory()
                {
                        checkError(cudaFreeHost(_hostPtr));
                }

                operator MemType*()
                {
                        return _hostPtr;
                }

                operator const MemType*() const
                {
                        return _hostPtr;
                }

        private:
                MemType* _hostPtr;

        };

        template<typename MemType>
        class GlobalMemory: public Memory<MemType>
        {
        public:
                GlobalMemory(std::size_t size)
                                : Memory<MemType>(size)
                {
                        clog(trace) << "Allocating " << this->bytes() << " bytes in global memory." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                }

                GlobalMemory(std::size_t size, int value)
                                : Memory<MemType>(size)
                {
                        clog(trace) << "Allocating " << this->bytes() << " bytes in global memory and setting them to 0x" << std::hex << value << std::dec << "." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                        checkError(cudaMemset(_devicePtr, value, this->bytes()));
                }

                GlobalMemory(const GlobalMemory<MemType>& other)
                        : Memory<MemType>(other.size())
                {
                        clog(trace) << "Allocating and copying " << this->bytes() << " bytes in global memory." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                        checkError(
                                        cudaMemcpy(_devicePtr, other._devicePtr,
                                                        this->bytes(),
                                                        cudaMemcpyDeviceToDevice));
                }

                ~GlobalMemory()
                {
                        checkError(cudaFree(_devicePtr));
                }

                operator MemType*()
                {
                        return _devicePtr;
                }

                operator const MemType*() const
                {
                        return _devicePtr;
                }

                void download(MemType* hostPtr) const
                {
                        clog(trace) << "Downloading " << this->bytes() << " bytes from global memory." << std::endl;
                        checkError(
                                        cudaMemcpy(hostPtr, _devicePtr,
                                                        this->bytes(),
                                                        cudaMemcpyDeviceToHost));
                }

                void upload(const MemType* hostPtr)
                {
                        clog(trace) << "Uploading " << this->bytes() << " bytes to global memory." << std::endl;
                        checkError(
                                        cudaMemcpy(_devicePtr, hostPtr,
                                                        this->bytes(),
                                                        cudaMemcpyHostToDevice));
                }

        private:
                MemType* _devicePtr;
        };

        template<typename MemType>
        class ConstantMemory: public Memory<MemType>
        {
        public:
                ConstantMemory(const void* symbol, std::size_t size)
                                : _symbol(symbol), Memory<MemType>(size)
                {
                }

                ConstantMemory(const ConstantMemory<MemType>& other)
                {
                        assert(this->size() == other.size());
                        checkError(
                                        cudaMemcpyToSymbol(_symbol, other._symbol,
                                                        cudaMemcpyDeviceToDevice));
                }

                void upload(const MemType* hostPtr)
                {
                        checkError(
                                        cudaMemcpyToSymbol(_symbol, hostPtr,
                                                        this->bytes(), 0,
                                                        cudaMemcpyHostToDevice));
                }

        private:
                const void *_symbol;
        };
}

#endif
