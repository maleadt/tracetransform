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
#include <vector>

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
                explicit Memory(std::vector<std::size_t> sizes)
                                : _sizes(sizes)
                {
                        assert(sizes.size() > 0);
                }

                virtual ~Memory()
                {
                }

                std::vector<std::size_t> sizes() const
                {
                        return _sizes;
                }

                std::size_t size(unsigned int dim) const
                {
                        assert(dim < _sizes.size());
                        return _sizes[dim];
                }

                std::size_t size() const
                {
                        std::size_t product = 1;
                        for (unsigned int i = 0; i < _sizes.size(); i++)
                                product *= _sizes[i];
                        return product;
                }

                std::size_t bytes() const
                {
                        return size() * sizeof(MemType);
                }

                std::size_t rows() const
                {
                        return size(0);
                }

                std::size_t cols() const
                {
                        return size(1);
                }

        private:
                Memory(const Memory&);
                Memory& operator=(const Memory&);

                std::vector<std::size_t> _sizes;
        };

        __attribute__((unused))
        static std::vector<std::size_t> size_1d(std::size_t dim1)
        {
                std::vector<std::size_t> sizes(1);
                sizes[0] = dim1;
                return sizes;
        }

        __attribute__((unused))
        static std::vector<std::size_t> size_2d(std::size_t dim1, std::size_t dim2)
        {
                std::vector<std::size_t> sizes(2);
                sizes[0] = dim1;
                sizes[1] = dim2;
                return sizes;
        }

        __attribute__((unused))
        static std::vector<std::size_t> size_3d(std::size_t dim1, std::size_t dim2, std::size_t dim3)
        {
                std::vector<std::size_t> sizes(3);
                sizes[0] = dim1;
                sizes[1] = dim2;
                sizes[2] = dim3;
                return sizes;
        }

        template<typename MemType>
        class HostMemory: public Memory<MemType>
        {
        public:
                HostMemory(std::vector<std::size_t> sizes)
                                : Memory<MemType>(sizes)
                {
                        clog(trace) << "Allocating " << this->bytes()
                                        << " bytes of host memory."
                                        << std::endl;
                        checkError(
                                        cudaHostAlloc(&_hostPtr, this->bytes(),
                                                        cudaHostAllocDefault));
                }

                HostMemory(const HostMemory<MemType>& other)
                                : Memory<MemType>(other.sizes())
                {
                        clog(trace) << "Allocating " << this->bytes()
                                        << " bytes of host memory and setting contents."
                                        << std::endl;
                        checkError(
                                        cudaHostAlloc(&_hostPtr, this->bytes(),
                                                        cudaHostAllocDefault));
                        checkError(
                                        cudaMemcpy(_hostPtr, other._hostPtr,
                                                        this->bytes(),
                                                        cudaMemcpyHostToHost));
                }

                ~HostMemory()
                {
                        clog(trace) << "Freeing " << this->bytes() << " bytes of host memory." << std::endl;
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
                GlobalMemory(std::vector<std::size_t> sizes)
                                : Memory<MemType>(sizes)
                {
                        clog(trace) << "Allocating " << this->bytes() << " bytes of global memory." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                }

                GlobalMemory(std::vector<std::size_t> sizes, int value)
                                : Memory<MemType>(sizes)
                {
                        clog(trace) << "Allocating " << this->bytes() << " bytes of global memory and setting them to 0x" << std::hex << value << std::dec << "." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                        checkError(cudaMemset(_devicePtr, value, this->bytes()));
                }

                GlobalMemory(const GlobalMemory<MemType>& other)
                        : Memory<MemType>(other.sizes())
                {
                        clog(trace) << "Allocating " << this->bytes() << " bytes of global memory and setting contents." << std::endl;
                        checkError(cudaMalloc(&_devicePtr, this->bytes()));
                        checkError(
                                        cudaMemcpy(_devicePtr, other._devicePtr,
                                                        this->bytes(),
                                                        cudaMemcpyDeviceToDevice));
                }

                ~GlobalMemory()
                {
                        clog(trace) << "Freeing " << this->bytes() << " bytes of global memory." << std::endl;
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

                void downloadAsync(const HostMemory<MemType> hostMem) const
                {
                        clog(trace) << "Asynchronously downloading " << this->bytes() << " bytes from global memory." << std::endl;
                        checkError(
                                        cudaMemcpy(hostMem, _devicePtr,
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

                void uploadAsync(const HostMemory<MemType> hostMem)
                {
                        clog(trace) << "Asynchronously uploading " << this->bytes() << " bytes to global memory." << std::endl;
                        checkError(
                                        cudaMemcpyAsync(_devicePtr, hostMem,
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
                ConstantMemory(const void* symbol, std::vector<std::size_t> sizes)
                                : _symbol(symbol), Memory<MemType>(sizes)
                {
                        clog(trace) << "Configuring " << this->bytes() << " bytes of constant memory." << std::endl;
                }

                ConstantMemory(const ConstantMemory<MemType>& other)
                : Memory<MemType>(other.sizes())
                {
                        clog(trace) << "Configuring " << this->bytes() << " bytes of constant memory and setting contents." << std::endl;
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

                void uploadAsync(const HostMemory<MemType> hostMem)
                {
                        checkError(
                                        cudaMemcpyToSymbolAsync(_symbol, hostMem,
                                                        this->bytes(), 0,
                                                        cudaMemcpyHostToDevice));
                }

        private:
                const void *_symbol;
        };
}

#endif
