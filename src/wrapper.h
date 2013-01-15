//
// Configuration
//

// Include guard
#ifndef WRAPPER_H
#define WRAPPER_H


//
// Module definitions
//

// Wrapper interface
class FunctionalWrapper {
public:
    FunctionalWrapper()
    { }
    virtual ~FunctionalWrapper()
    { };

    virtual double operator()(const double* data, const size_t length) const = 0;
};

// Simple wrapper without any additional arguments
class SimpleFunctionalWrapper : public FunctionalWrapper {
public:
    SimpleFunctionalWrapper(std::function<double(const double*, const size_t)> function)
        : FunctionalWrapper(), _function(function)
    { }

    double operator()(const double* data, const size_t length) const
    {
        return _function(data, length);
    }

private:
    std::function<double(const double*, const size_t)> _function;
};

// Generic wrapper implementation using variadic templates
template<typename... Parameters>
class GenericFunctionalWrapper : public FunctionalWrapper
{
public:
    GenericFunctionalWrapper(std::function<double(const double*, const size_t, Parameters...)> functional)
        : FunctionalWrapper(), _functional(functional)
    { }

    // TODO: allow partial configuration
    void configure(Parameters... parameters)
    {
        _reduced_functional = std::bind(_functional, std::placeholders::_1, std::placeholders::_2, parameters...);
    }

    double operator()(const double* data, const size_t length) const
    {
        return _reduced_functional(data, length);
    }

private:
    std::function<double(const double*, const size_t, Parameters...)> _functional;
    std::function<double(const double*, const size_t)> _reduced_functional;
};

#endif
