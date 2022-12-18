#ifndef LOSS_FUNCTION_HPP_
#define LOSS_FUNCTION_HPP_

#include <vector>
#include "Config.hpp"
#include "Helper.hpp"

namespace NeuralNetworkBasics {

class LossFunction {
    virtual Scalar loss(std::vector<Scalar>& predicted, std::vector<Scalar>& expected);
    virtual std::vector<Scalar> get_gradient(std::vector<Scalar>& predicted, std::vector<Scalar>& expected);
};


}

#endif