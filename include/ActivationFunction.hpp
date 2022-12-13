#ifndef ACTIVATION_FUNCTION_HPP_
#define ACTIVATION_FUNCTION_HPP_

#include <vector>
#include "Config.hpp"

namespace NeuralNetworkBasics {

class ActivationFunction {
    virtual std::vector<Scalar> activate(std::vector<Scalar>& input);
    virtual std::vector<Scalar> get_gradient(std::vector<Scalar>& input);
};

}

#endif