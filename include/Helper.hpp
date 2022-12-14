#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <type_traits>
#include <cmath>
#include <limits>

#include "Config.hpp"

#include "ActivationFunctions/LeakyReLU.hpp"
#include "ActivationFunctions/ReLU.hpp"
#include "ActivationFunctions/Sigmoid.hpp"
#include "ActivationFunctions/Softmax.hpp"
#include "ActivationFunctions/Tanh.hpp"

namespace NeuralNetworkBasics {

// Used in Layer.hpp to compile-time assert that activation function is valid
#define is_activation(T) (std::is_same<T, LeakyReLU>::value || \
                            std::is_same<T, ReLU>::value || \
                            std::is_same<T, Sigmoid>::value || \
                            std::is_same<T, Softmax>::value || \
                            std::is_same<T, Tanh>::value)

// Defines log function that is safe to run on 0 inputs
inline Scalar safe_log(Scalar x) {
    // Note: Numeric limits are there to prevent overflow errors 
    // (if smaller than epsilon, i.e. 0, then evaluate really small #)
    return std::log(
        std::max(std::numeric_limits<Scalar>::epsilon(), x)
    );
}

}

#endif