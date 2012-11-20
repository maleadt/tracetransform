//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_H
#define TRACETRANSFORM_H

// System includes
#include <limits>

// Library includes
#include <Eigen/Dense>


//
// Routines
//

Eigen::MatrixXd getTraceTransform(
	const Eigen::MatrixXd &input,
	const double a_stepsize,
	const double p_stepsize,
	Functional tfunctional,
	void *tfunctional_arguments)
{
	assert(a_stepsize > 0);
	assert(p_stepsize > 0);
	assert(input.rows() == input.cols());	// padded image!

	// Get the image origin to rotate around
	Point origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

	// Calculate and allocate the output matrix
	unsigned int a_steps = (unsigned int) std::floor(360 / a_stepsize);
	unsigned int p_steps = (unsigned int) std::floor(input.rows() / p_stepsize);
	Eigen::MatrixXd output((int) p_steps, (int) a_steps);

	// Process all angles
	#pragma omp parallel for
	for (unsigned int a_step = 0; a_step < a_steps; a_step++) {
		// Rotate the image
		double a = a_step * a_stepsize;
		Eigen::MatrixXd input_rotated = rotate(input, origin, deg2rad(a));

		// Process all projection bands
		for (unsigned int p_step = 0; p_step < p_steps; p_step++) {
			output(
				(signed) p_step,	// row
				(signed) a_step		// column
			) = tfunctional(
				input_rotated.data() + ((int) p_stepsize*p_step) * input.rows(),
				input.rows(),
				tfunctional_arguments);
		}
	}

	return output;
}

#endif
