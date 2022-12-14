#ifndef MSE_LOSS_HPP_
#define MSE_LOSS_HPP_

#include <stdexcept>
#include <cmath>
#include <vector>
#include "../Config.hpp"
#include "../LossFunction.hpp"

#include <assert.h>

namespace NeuralNetworkBasics {

class MSE {
    public:
    Scalar loss(std::vector<Scalar>& predicted, std::vector<Scalar>& expected) {
        // Only run this check in debug mode (defined in Config.hpp)
        assert(predicted.size() == expected.size());

        Scalar sum = 0;

        for (unsigned int i = 0; i < predicted.size(); i++) {
            Scalar error = predicted[i] - expected[i];

            sum += (error * error);
        }

        return ((Scalar) 0.5) * sum / predicted.size();
    }

    static std::vector<Scalar> get_gradient(std::vector<Scalar>& predicted, std::vector<Scalar>& expected) {
        assert(predicted.size() == expected.size());

        std::vector<Scalar> output(predicted.size());

        for (unsigned int i = 0; i < predicted.size(); i++) {
            output[i] = predicted[i] - expected[i];
        }

        return output;
    }
};

}

#endif