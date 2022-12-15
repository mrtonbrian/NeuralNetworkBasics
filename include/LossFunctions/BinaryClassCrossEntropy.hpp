#ifndef BINARY_CROSS_ENTROPY_LOSS_HPP_
#define BINARY_CROSS_ENTROPY_LOSS_HPP_

#include <stdexcept>
#include <cmath>
#include <vector>
#include <limits>
#include "../Config.hpp"
#include "../Helper.hpp"
#include "../LossFunction.hpp"

#include <assert.h>

namespace NeuralNetworkBasics {

class BinaryClassCrossEntropy : public LossFunction {

    // References: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    // https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
    Scalar loss(std::vector<Scalar>& predicted, std::vector<Scalar>& expected) {
        // Only run this check in debug mode (NDEBUG defined in Config.hpp)
        assert(predicted.size() == expected.size());

        Scalar output = 0;

        for (unsigned int i = 0; i < predicted.size(); i++) {
            output += (expected[i]) * safe_log(predicted[i]);
        }

        return -output;
    }

    std::vector<Scalar> get_gradient(std::vector<Scalar>& predicted, std::vector<Scalar>& expected) {
        assert(predicted.size() == expected.size());

        std::vector<Scalar> output(predicted.size());

        for (unsigned int i = 0; i < predicted.size(); i++) {
            // Formula: entropy for a class = -(expected_prob * log(predicted_prob) + (1-expected_prob) * log(1-predicted_prob))
            output[i] = -((expected[i] * safe_log(predicted[i]) + ((1 - expected[i]) * safe_log(1-predicted[i]))));
        }

        return output;
    }
};

}

#endif