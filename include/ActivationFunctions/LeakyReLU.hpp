#ifndef LEAKY_RELU_ACTIVATION_HPP_
#define LEAKY_RELU_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class LeakyReLU : public ActivationFunction {

    std::vector<Scalar> activate(std::vector<Scalar>& input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        for (const Scalar& s : input) {
            output.push_back(std::max((Scalar) 0.01 * s, s));
        }

        return output;
    }

    std::vector<Scalar> get_gradient(std::vector<Scalar>& input) {
        std::vector<Scalar> output;

        for (Scalar s : input) {
            output.push_back((s < 0) ? (Scalar) 0.01 : (Scalar) 1);
        }

        return output;
    }
};

}

#endif