#ifndef SIGMOID_ACTIVATION_HPP_
#define SIGMOID_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"

namespace NeuralNetworkBasics {

class Sigmoid {
    public:
    // Sigmoid function: f(x) = 1/(1+e^(-x))
    static std::vector<Scalar> activate(std::vector<Scalar> input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        for (const Scalar& s : input) {
            output.push_back(((Scalar) 1) / (1 + std::exp(-s)));
        }

        return output;
    }

    // Derivative of Sigmoid: f(x)*(1-f(x))
    static std::vector<Scalar> get_gradient(std::vector<Scalar>& input) {
        std::vector<Scalar> output = activate(input);

        for (unsigned int i = 0; i < output.size(); i++) {
            output[i] = output[i] * (1 - output[i]);
        }

        return output;
    }
};

}

#endif