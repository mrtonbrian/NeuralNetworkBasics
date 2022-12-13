#ifndef RELU_ACTIVATION_HPP_
#define RELU_ACTIVATION_HPP_

#include <algorithm>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class ReLU : public ActivationFunction {

    Scalar activate(std::vector<Scalar> input, int index) {
        return std::max((Scalar) 0, input[index]);
    }

    // Derivative of Sigmoid: (tanh(x))' = 1-tanh^2(x)
    Scalar derivative(std::vector<Scalar> input, int index) {
        return (input[index] <= 0) ? 0 : 1; 
    }
};

}

#endif