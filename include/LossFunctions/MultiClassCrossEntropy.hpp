#ifndef CROSS_ENTROPY_LOSS_HPP_
#define CROSS_ENTROPY_LOSS_HPP_

#include <stdexcept>
#include <cmath>
#include <vector>
#include <limits>
#include "../Config.hpp"
#include "../Helper.hpp"
#include "../LossFunction.hpp"

#include <assert.h>

namespace NeuralNetworkBasics {

class MultiClassCrossEntropy : public LossFunction {

    // Reference: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
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
            output[i] = -(expected[i] / predicted[i]);
        }

        return output;
    }
};

}

#endif