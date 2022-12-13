#ifndef SOFTMAX_ACTIVATION_HPP_
#define SOFTMAX_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class Softmax : public ActivationFunction {

    Scalar activate(std::vector<Scalar> input, int index) {
        return std::max((Scalar) 0.01 * input[index], input[index]);
    }

    // Derivative of Sigmoid: (tanh(x))' = 1-tanh^2(x)
    Scalar derivative(std::vector<Scalar> input, int index) {
        return (input[index] < 0) ? 0.01 : (Scalar) 1; 
    }
};

}

#endif