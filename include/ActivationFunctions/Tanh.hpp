#ifndef TANH_ACTIVATION_HPP_
#define TANH_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class Tanh : public ActivationFunction {

    std::vector<Scalar> activate(std::vector<Scalar> input) {
        std::vector<Scalar> output;
        for (const Scalar& s : input) {
            output.push_back(std::tanh(s));
        }

        return output;
    }

    std::vector<Scalar> get_gradient(std::vector<Scalar>& input) {
        std::vector<Scalar> output = activate(input);

        // Derivative of Tanh: (tanh(x))' = 1-tanh^2(x)
        for (unsigned int i = 0; i < output.size(); i++) {
            output[i] = 1 - (output[i] * output[i]);
        }

        return output;
    }
};

}

#endif