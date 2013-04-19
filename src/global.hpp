//
// Configuration
//

// NOTE: this header should adhere to C++98 since it is included by the CUDA
//       compiler (and nvcc, as of CUDAv5, does not support C++11 yet).

// Include guard
#ifndef _TRACETRANSFORM_GLOBAL_
#define _TRACETRANSFORM_GLOBAL_

// Eigen
#include <Eigen/Core>


//
// Structs
//

template<typename T>
struct Point
{
    typedef Eigen::Matrix<T, 1, 2> type;
};

template <typename T>
std::ostream& operator<<(std::ostream &stream, const typename Point<T>::type &point) {
        stream << point.x() << "x" << point.y();
        return stream;
}

#endif
