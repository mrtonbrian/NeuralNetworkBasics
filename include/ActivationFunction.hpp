#ifndef ACTIVATION_FUNCTION_HPP_
#define ACTIVATION_FUNCTION_HPP_

#include <vector>
#include "Config.hpp"

namespace NeuralNetworkBasics {

class ActivationFunction {
    virtual Scalar activate(std::vector<Scalar> input);
    virtual Scalar derivative(std::vector<Scalar> input);
};

}

#endif