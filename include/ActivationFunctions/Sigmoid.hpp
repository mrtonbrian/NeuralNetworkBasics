#ifndef SIGMOID_ACTIVATION_HPP_
#define SIGMOID_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class Sigmoid : public ActivationFunction {

    // Sigmoid function: f(x) = 1/(1+e^(-x))
    Scalar activate(std::vector<Scalar> input, int index) {
        return ((Scalar) 1.0) / (1 + std::exp(-input[index]));
    }

    // Derivative of Sigmoid: f(x)*(1-f(x))
    Scalar derivative(std::vector<Scalar> input, int index) {
        Scalar activationValue = activate(input, index);
        return activationValue * (1 - activationValue);
    }
};

}

#endif