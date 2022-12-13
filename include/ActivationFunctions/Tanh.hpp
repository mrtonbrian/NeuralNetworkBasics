#ifndef TANH_ACTIVATION_HPP_
#define TANH_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class Tanh : public ActivationFunction {

    Scalar activate(std::vector<Scalar> input, int index) {
        return std::tanh(input[index]);
    }

    // Derivative of Sigmoid: (tanh(x))' = 1-tanh^2(x)
    Scalar derivative(std::vector<Scalar> input, int index) {
        Scalar activationValue = activate(input, index);
        return 1 - (activationValue * activationValue);
    }
};

}

#endif