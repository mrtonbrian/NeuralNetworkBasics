#ifndef LOSS_FUNCTION_HPP_
#define LOSS_FUNCTION_HPP_

#include <vector>
#include "Config.hpp"

namespace NeuralNetworkBasics {

class LossFunction {
    virtual Scalar loss(std::vector<Scalar> input, int index);
    virtual std::vector<Scalar> get_gradient(std::vector<Scalar> input);
};

}

#endif