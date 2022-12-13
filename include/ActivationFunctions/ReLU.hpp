#ifndef RELU_ACTIVATION_HPP_
#define RELU_ACTIVATION_HPP_

#include <algorithm>
#include <vector>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class ReLU : public ActivationFunction {

    std::vector<Scalar> activate(std::vector<Scalar>& input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        for (const Scalar& s : input) {
            output.push_back(std::max((Scalar) 0, s));
        }

        return output;
    }

    std::vector<Scalar> get_gradient(std::vector<Scalar>& input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        for (Scalar s : input) {
            output.push_back((s <= 0) ? (Scalar) 0 : (Scalar) 1);
        }

        return output;
    }
};

}

#endif