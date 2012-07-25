//
// Configuration
//

// Include guard
#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H


//
// Class definition
//

template <typename IN, typename OUT>
class Functional
{
public:
	virtual OUT operator()(TraceIterator<IN>& iterator) = 0;
};


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

template <typename IN>
class PFunctional1 : public PFunctional<IN, double>
{
public:
	// P(g(p)) = Sum(k) abs(g(p+1) -g(p))
	double operator()(TraceIterator<IN>& iterator)
	{
		unsigned long sum = 0;
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

#endif
