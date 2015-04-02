This repository hosts different implementations of the trace transform
algorithm, developed as part of a case study comparison.


Trace Transform
---------------

The trace transform is a radial projection transformation, and can be viewed as
a generalization of the Radon transform. The transformation allows to construct
image features that are invariant to a chosen group of image transformations.

Relevant papers are:

 * A. Kadyrov and M. Petrou, The trace transform and its applications
 * M. Petrou and A. Kadyrov, Affine invariant features from the trace transform
 * A. Frías-Velézquez et al., Object identification by using orthonormal circus
   functions from the trace transform
 * T. Besard and A. Frías-Velézquez, Case study of multiple trace transform
   implementations

The implementations in this repository only implement part of the trace
transform, namely the sinogram and circus function calculation, with some
implementations optionally supporting orthonormal circus functions. Also, the
circus functions are the raw signatures without the normalization suggested by
Petrou.


Implementations
---------------

 * matlab: original MATLAB implementation
 * matlab-mex: MATLAB version with most of the sinogram calculation ported over
   to MEX
 * octave: port of the MATLAB version to Octave (minimal differences)
 * scilab: port of the MATLAB version to Scilab (limited differences, but more
   than with Octave)
 * julia: version for Julia, tested on v0.3
 * c++: low-level version in C++, using Eigen for matrix calculations
 * openmp: C++ version with OpenMP annotations (minimal differences)
 * cuda: heavily-optimised version for NVIDIA GPUs, with most of the
   computations reimplemented in CUDA-C


Licensing
---------

All code is licensed under the GNU General Public License v3.
See the COPYING document for more information.


Contact
-------

For more information, you can contact:

 * Andrés Frías-Velézquez <Andres.FriasVelazquez@telin.ugent.be>: developer of
   the MATLAB implementations, and researcher behind the orthonormal circus
   functions
 * Tim Besard <tim.besard@elis.ugent.be>: developer of the other implementations
