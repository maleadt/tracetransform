//
// Configuration
//

// Include guard
#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H


//
// Class definition
//

typedef std::function<double (const double*, const size_t, const void*)> Functional;


//
// Auxiliary
//

inline
size_t findWeighedMedian(const double* data, const size_t length)
{		
	double sum = 0;
	for (size_t i = 0; i < length; i++)
		sum += data[i];

	double integral = 0;
	for (size_t i = 0; i < length; i++) {
		integral += data[i];
		if (2*integral >= sum)
			return i;
	}
	return length-1;
}

inline
size_t findWeighedMedianSquared(const double* data, const size_t length)
{		
	double sum = 0;
	for (size_t i = 0; i < length; i++)
		sum += std::sqrt(data[i]);

	double integral = 0;
	for (size_t i = 0; i < length; i++) {
		integral += std::sqrt(data[i]);
		if (2*integral >= sum)
			return i;
	}
}

// TODO: convert to plain C
inline
std::vector<std::complex<double>> dft(const std::vector<std::complex<double>> &sample)
{
	std::vector<std::complex<double>> output(sample.size());
	for(size_t i = 0; i < sample.size(); i++) 
	{
		output[i] = std::complex<double>(0, 0);
		double arg = -2.0 * M_PI * (double)i / (double)sample.size();
		for(size_t j = 0; j < sample.size(); j++) 
		{
			double cosarg = std::cos(j * arg);
			double sinarg = std::sin(j * arg);
			output[i] += std::complex<double>(
				((double)sample[j].real() * cosarg
					- (double)sample[j].imag() * sinarg),
				((double)sample[j].real() * sinarg
					+ (double)sample[j].imag() * cosarg)
			);
		}
	}
	return output;
}


//
// T functionals
//

// T-functional for the Radon transform.
double TFunctionalRadon(const double* data, const size_t length, const void*) {
	double integral = 0;
	for (size_t t = 0; t < length; t++)
		integral += data[t];
	return integral;		
}

double TFunctional1(const double* data, const size_t length, const void*) {
	// Transform the domain from t to r
	size_t median = findWeighedMedian(data, length);

	// Integrate
	double integral = 0;
	for (size_t r = 0; r < length-median; r++)
		integral += data[r+median] * r;
	return integral;		
}

double TFunctional2(const double* data, const size_t length, const void*) {
	// Transform the domain from t to r
	size_t median = findWeighedMedian(data, length);

	// Integrate
	double integral = 0;
	for (size_t r = 0; r < length-median; r++)
		integral += data[r+median] * r*r;
	return integral;		
}

double TFunctional3(const double* data, const size_t length, const void*) {
	// Transform the domain from t to r1
	size_t squaredmedian = findWeighedMedianSquared(data, length);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 5);
	for (size_t r1 = 0; r1 < length-squaredmedian; r1++) {
		if (r1 > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(r1))
				* (r1*data[r1+squaredmedian]);

	}
	return std::abs(integral);
}

double TFunctional4(const double* data, const size_t length, const void*) {
	// Transform the domain from t to r1
	size_t squaredmedian = findWeighedMedianSquared(data, length);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 3);
	for (size_t r1 = 0; r1 < length-squaredmedian; r1++) {
		if (r1 > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(r1))
				* data[r1+squaredmedian];

	}
	return std::abs(integral);
}

double TFunctional5(const double* data, const size_t length, const void*) {
	// Transform the domain from t to r1
	size_t squaredmedian = findWeighedMedianSquared(data, length);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 4);
	for (size_t r1 = 0; r1 < length-squaredmedian; r1++) {
		if (r1 > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(r1))
				* (std::sqrt(r1)*data[r1+squaredmedian]);

	}
	return std::abs(integral);
}


//
// P-functionals
//

double PFunctional1(const double* data, const size_t length, const void*) {
	double sum = 0;
	double previous = data[0];
	for (size_t t = 1; t < length; t++) {
		double current = data[t];
		sum += std::abs(previous - current);
		previous = current;
	}
	return sum;
}

double PFunctional2(const double* data, const size_t length, const void*) {
	size_t median = findWeighedMedian(data, length);
	return data[median];
}

double PFunctional3(const double* data, const size_t length, const void*) {
	// Dump the trace in a vector
	std::vector<std::complex<double>> trace;
	for (size_t i = 0; i < length; i++)
		trace.push_back(data[i]);

	// Calculate and post-process the Fourier transform
	std::vector<std::complex<double>> fourier = dft(trace);
	std::vector<double> trace_processed(fourier.size());
	for (size_t i = 0; i < fourier.size(); i++)
		trace_processed[i] = std::pow(std::abs(fourier[i]), 4);

	// Integrate
	double sum = 0;
	for (size_t i = 0; i < trace_processed.size(); i++)
		sum += trace_processed[i];
	return sum;
}

struct ArgumentsHermite {
	unsigned int order;
	unsigned int center;
};

double PFunctionalHermite(const double* data, const size_t length, const void* _arguments) {
	// Unpack the arguments
	ArgumentsHermite *arguments = (ArgumentsHermite*) _arguments;

	// Discretize the [-10, 10] domain to fit the column iterator
	double z = -10;
	double stepsize_lower = 10.0 / arguments->center;
	double stepsize_upper = 10.0 / (length - 1 - arguments->center);

	// Calculate the integral
	double integral = 0;
	for (size_t t = 1; t < length; t++) {
		integral += data[t] * hermite_function(arguments->order, z);
		if (z < 0)
			z += stepsize_lower;
		else
			z += stepsize_upper;
	}
	return integral;
}

#endif
