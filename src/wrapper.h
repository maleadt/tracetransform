//
// Configuration
//

// Include guard
#ifndef WRAPPER_H
#define WRAPPER_H

class FunctionalWrapper {
public:
    FunctionalWrapper()
    { }
    virtual ~FunctionalWrapper()
    { };

    virtual double operator()(const double* data, const size_t length) const = 0;
};

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

class HermiteFunctionalWrapper : public FunctionalWrapper {
public:
    HermiteFunctionalWrapper(std::function<double(const double*, const size_t, unsigned int, unsigned int)> function, unsigned int order)
        : FunctionalWrapper(), _function(function), _order(order)
    { }

    void center(unsigned int center)
    {
        _center = center;
    }

    double operator()(const double* data, const size_t length) const
    {
        return _function(data, length, _order, _center);
    }

private:
    std::function<double(const double*, const size_t, unsigned int, unsigned int)> _function;

    unsigned int _order;
    unsigned int _center;
};

#endif
