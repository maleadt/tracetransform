//
// Configuration
//

// Include guard
#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H

// Local includes
#include "auxiliary.h"
#include "traceiterator.h"


//
// Class definition
//

template <typename IN, typename OUT>
class Functional
{
public:
	virtual cv::Mat *preprocess(const cv::Mat &sinogram)
	{
		return nullptr;
	}

	virtual OUT operator()(TraceIterator<IN>& iterator) = 0;
};


//
// Auxiliary
//

cv::Mat nearest_orthonormal_sinogram(
	const cv::Mat &sinogram,
	unsigned int& new_center)
{
	// Detect the offset of each column to the sinogram center
	assert(sinogram.rows > 0);
	assert(sinogram.cols >= 0);
	unsigned int sinogram_center = (unsigned int) std::floor((sinogram.rows - 1) / 2.0);
	std::vector<int> offset(sinogram.cols);
	for (int p = 0; p < sinogram.cols; p++) {
		// Determine the trace segment
		Segment trace = Segment{
			Point{(double)p, 0},
			Point{(double)p, (double)sinogram.rows-1}
		};

		// Set-up the trace iterator
		TraceIterator<double> iterator(sinogram, trace);
		assert(iterator.valid());

		// Get and compare the median
		Point median = iterator_weighedmedian(iterator);
		offset[p] = (median.y - sinogram_center);
	}

	// Align each column to the sinogram center
	int min = *(std::min_element(offset.begin(), offset.end()));
	int max = *(std::max_element(offset.begin(), offset.end()));
	unsigned int padding = max + std::abs(min);
	new_center = sinogram_center + max;
	cv::Mat aligned = cv::Mat::zeros(
		sinogram.rows + padding,
		sinogram.cols,
		sinogram.type()
	);
	for (int j = 0; j < sinogram.cols; j++) {
		for (int i = 0; i < sinogram.rows; i++) {
			aligned.at<double>(max+i-offset[j], j) = 
				sinogram.at<double>(i, j);
		}
	}

	// Compute the nearest orthonormal sinogram
	cv::SVD svd(aligned, cv::SVD::FULL_UV);
	cv::Mat diagonal = cv::Mat::eye(
		aligned.size(),	// (often) rectangular!
		aligned.type()
	);
	cv::Mat nos = svd.u * diagonal * svd.vt;

	return aligned;
}


//
// T functionals
//

template <typename IN, typename OUT>
class TFunctional : public Functional<IN, OUT>
{

};

// T-functional for the Radon transform.
template <typename IN>
class TFunctionalRadon : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] f(t)dt
	double operator()(TraceIterator<IN>& iterator)
	{
		double integral = 0;
		while (iterator.hasNext()) {
			integral += iterator.value();
			iterator.next();
		}
		return integral;		
	}
};

template <typename IN>
class TFunctional1 : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] r*f(r)dr
	double operator()(TraceIterator<IN>& iterator)
	{
		// Transform the domain from t to r
		Point r = iterator_weighedmedian(iterator);
		TraceIterator<IN> transformed = iterator.transformDomain(
			Segment{
				r,
				iterator.segment().end
			}
		);

		// Integrate
		double integral = 0;
		for (unsigned int t = 0; transformed.hasNext(); t++) {
			integral += transformed.value() * t;
			transformed.next();
		}
		return integral;		
	}
};

template <typename IN>
class TFunctional2 : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] r^2*f(r)dr
	double operator()(TraceIterator<IN>& iterator)
	{
		// Transform the domain from t to r
		Point r = iterator_weighedmedian(iterator);
		TraceIterator<IN> transformed = iterator.transformDomain(
			Segment{
				r,
				iterator.segment().end
			}
		);

		// Integrate
		double integral = 0;
		for (unsigned int t = 0; transformed.hasNext(); t++) {
			integral += transformed.value() * t*t;
			transformed.next();
		}
		return integral;		
	}
};

template <typename IN>
class TFunctional3 : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] exp(5i*log(r1))*r1*f(r1)dr1
	double operator()(TraceIterator<IN>& iterator)
	{
		// Transform the domain from t to r1
		Point r1 = iterator_weighedmedian_sqrt(iterator);
		TraceIterator<IN> transformed = iterator.transformDomain(
			Segment{
				r1,
				iterator.segment().end
			}
		);

		// Integrate
		std::complex<double> integral(0, 0);
		const std::complex<double> factor(0, 5);
		for (unsigned int t = 0; transformed.hasNext(); t++) {
			if (t > 0)	// since exp(i*log(0)) == 0
				integral += exp(factor*std::log(t))
					* (t*(double)transformed.value());
			transformed.next();
		}
		return std::abs(integral);		
	}
};

template <typename IN>
class TFunctional4 : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] exp(3i*log(r1))*f(r1)dr1
	double operator()(TraceIterator<IN>& iterator)
	{
		// Transform the domain from t to r1
		Point r1 = iterator_weighedmedian_sqrt(iterator);
		TraceIterator<IN> transformed = iterator.transformDomain(
			Segment{
				r1,
				iterator.segment().end
			}
		);

		// Integrate
		std::complex<double> integral(0, 0);
		const std::complex<double> factor(0, 3);
		for (unsigned int t = 0; transformed.hasNext(); t++) {
			if (t > 0)	// since exp(i*log(0)) == 0
				integral += exp(factor*std::log(t))
					* (double)transformed.value();
			transformed.next();
		}
		return std::abs(integral);	
	}
};

template <typename IN>
class TFunctional5 : public TFunctional<IN, double>
{
public:
	// T(f(t)) = Int[0-inf] exp(4i*log(r1))*sqrt(r1)*f(r1)dr1
	double operator()(TraceIterator<IN>& iterator)
	{
		// Transform the domain from t to r1
		Point r1 = iterator_weighedmedian_sqrt(iterator);
		TraceIterator<IN> transformed = iterator.transformDomain(
			Segment{
				r1,
				iterator.segment().end
			}
		);

		// Integrate
		std::complex<double> integral(0, 0);
		const std::complex<double> factor(0, 4);
		for (unsigned int t = 0; transformed.hasNext(); t++) {
			if (t > 0)	// since exp(i*log(0)) == 0
				integral += exp(factor*std::log(t))
					* (std::sqrt(t)*(double)transformed.value());
			transformed.next();
		}
		return std::abs(integral);
	}
};


//
// P-functionals
//

// TODO: P-functionals are column iterators, hence they should not contain all
//       the complex logic to billinearly interpolate points. Maybe use 
//       class inheritance to avoid this?

template <typename IN, typename OUT>
class PFunctional : public Functional<IN, OUT>
{
};

template <typename IN, typename OUT>
class PFunctionalOrthonormal : public PFunctional<IN, OUT>
{
public:
	cv::Mat *preprocess(const cv::Mat &sinogram)
	{
		cv::Mat *nos = new cv::Mat();
		*nos = nearest_orthonormal_sinogram(sinogram, m_center);
		return nos;
	}

protected:
	unsigned int m_center;
};

template <typename IN>
class PFunctional1 : public PFunctional<IN, double>
{
public:
	// P(g(p)) = Sum(k) abs(g(p+1) -g(p))
	double operator()(TraceIterator<IN>& iterator)
	{
		double sum = 0;
		double previous;
		if (iterator.hasNext()) {
			previous = iterator.value();
			iterator.next();
		}
		while (iterator.hasNext()) {
			double current = iterator.value();
			sum += std::abs(previous -current);
			previous = current;
			iterator.next();
		}
		return (double)sum;
	}
};

template <typename IN>
class PFunctional2 : public PFunctional<IN, double>
{
public:
	// P(g(p)) = median(g(p))
	double operator()(TraceIterator<IN>& iterator)
	{
		Point median = iterator_weighedmedian(iterator);
		return iterator.value(median);	// TODO: paper doesn't say g(median)?
	}
};

template <typename IN>
class PFunctional3 : public PFunctional<IN, double>
{
public:
	// P(g(p)) = Int |Fourier(g(p))|^4
	double operator()(TraceIterator<IN>& iterator)
	{
		// Dump the trace in a vector
		// TODO: don't do this explicitly?
		std::vector<std::complex<double>> trace;
		while (iterator.hasNext()) {
			trace.push_back(iterator.value());
			iterator.next();
		}

		// Calculate and post-process the Fourier transform
		std::vector<std::complex<double>> fourier = dft(trace);
		std::vector<double> trace_processed(fourier.size());
		for (size_t i = 0; i < fourier.size(); i++)
			trace_processed[i] = std::pow(std::abs(fourier[i]), 4);

		// Integrate
		// FIXME: these values are huge (read: overflow) since we use [0,255]
		double sum = 0;
		for (size_t i = 0; i < trace_processed.size(); i++)
			sum += trace_processed[i];
		return sum;
	}
};

template <typename IN>
class PFunctionalHermite : public PFunctionalOrthonormal<IN, double>
{
public:
	PFunctionalHermite(unsigned int order)
		: m_order(order)
	{
	}

	// Hn(g(p)) = Int psi(z)
	double operator()(TraceIterator<IN>& iterator)
	{
		// Discretize the [-10, 10] domain to fit the column iterator
		double z = -10;
		double stepsize_lower = 10.0 / this->m_center;
		double stepsize_upper = 10.0 / (iterator.samples() - 1 - this->m_center);
		// In case of 9 samples, and the center on the fifth (i=4), this results in
		// {-10, -7.5, -5.0, -2.5, 0 2.5, 5.0, 7.5, 10} (as per tutorial)
		// but matlab gives:
		//  -10.0000   -7.7778   -5.5556   -3.3333   -1.1111    1.1111    3.3333    5.5556    7.7778

		// Calculate the integral
		double integral = 0;
		while (iterator.hasNext()) {
			std::cout << hermite_function(m_order, z) << "\t";
			integral += iterator.value() * hermite_function(m_order, z);
			iterator.next();
			if (z < 0)
				z += stepsize_lower;
			else
				z += stepsize_upper;
		}
		std::cout << std::endl;
		return integral;
	}

private:
	const unsigned int m_order;
};

#endif
