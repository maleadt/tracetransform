//
// Configuration
//

// Include guard
#ifndef WRAPPER_H
#define WRAPPER_H

// Standard library
#include <functional>
#include <cassert>


//
// Module definitions
//

// Wrapper interface
class FunctionalWrapper {
public:
        virtual ~FunctionalWrapper()
        { };

        virtual double operator()(const double* data, const size_t length) const = 0;
};

// Simple wrapper without any additional arguments
class SimpleFunctionalWrapper : public FunctionalWrapper {
public:
        SimpleFunctionalWrapper(std::function<double(const double*, const size_t)> function)
                : _function(function)
        { }

        double operator()(const double* data, const size_t length) const
        {
                return _function(data, length);
        }

private:
        const std::function<double(const double*, const size_t)> _function;
};

// Generic wrapper implementation using variadic templates
template<typename... Parameters>
class GenericFunctionalWrapper : public FunctionalWrapper
{
public:
        GenericFunctionalWrapper(std::function<double(const double*, const size_t, Parameters...)> functional)
                : _functional(functional)
        { }

        // TODO: allow partial configuration
        void configure(Parameters... parameters)
        {
                _configured_functional = std::bind(_functional, std::placeholders::_1, std::placeholders::_2, parameters...);
        }

        double operator()(const double* data, const size_t length) const
        {
                assert(_configured_functional);
                return _configured_functional(data, length);
        }

private:
        const std::function<double(const double*, const size_t, Parameters...)> _functional;
        std::function<double(const double*, const size_t)> _configured_functional;
};

#endif
