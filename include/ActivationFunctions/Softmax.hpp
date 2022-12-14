#ifndef SOFTMAX_ACTIVATION_HPP_
#define SOFTMAX_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include <algorithm>
#include "../Config.hpp"

namespace NeuralNetworkBasics {

class Softmax {

    static std::vector<Scalar> activate(std::vector<Scalar> input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        Scalar sum = (Scalar) 0;

        for (const Scalar& s : input) {
            sum += std::exp(s);
        }

        for (const Scalar& s : input) {
            output.push_back(std::exp(s) / sum);
        }

        return output;
    }

    // O(N^2) :( - Wonder if there's a better way to do this; Probably by using a Linear Algebra Library?
    static std::vector<Scalar> get_gradient(std::vector<Scalar> input) {
        std::vector<Scalar> activations = activate(input);
        std::vector<Scalar> output(input.size(), (Scalar) 0.);
        for (unsigned int i = 0; i < activations.size(); i++) {
            for (unsigned int j = 0; j < activations.size(); j++) {
                if (i == j) {
                    output[i] += activations[i] * (1 - activations[j]);
                } else {
                    output[i] += -activations[i] * activations[j];
                }
            }
        }

        return output;
    }
};

}

#endif