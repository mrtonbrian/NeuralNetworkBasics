#ifndef SOFTMAX_STABLE_ACTIVATION_HPP_
#define SOFTMAX_STABLE_ACTIVATION_HPP_

#include <cmath>
#include <vector>
#include <algorithm>
#include "../Config.hpp"
#include "../ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class SoftmaxStable : public ActivationFunction {

    // Numerically stable Softmax from here:
    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // Potential optimization: Precalculating softmax using offset (needs testing); Detailed below
    // offset = max + log(sum)
    // output[i] = exp(input[i] - offset)
    std::vector<Scalar> activate(std::vector<Scalar> input) {
        std::vector<Scalar> output;
        output.reserve(input.size());

        Scalar sum = (Scalar) 0;
        Scalar maxValue = input[0];

        for (const Scalar& s : input) {
            maxValue = std::max(maxValue, s);
        }

        for (const Scalar& s : input) {
            sum += std::exp(s - maxValue);
        }

        for (const Scalar& s : input) {
            output.push_back(std::exp(s - maxValue) / sum);
        }

        return output;
    }

    // O(N^2) :( - Wonder if there's a better way to do this; Probably by using a Linear Algebra Library?
    std::vector<Scalar> get_gradient(std::vector<Scalar> input) {
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