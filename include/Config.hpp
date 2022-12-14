#ifndef CONFIG_HPP_
#define CONFIG_HPP_

namespace NeuralNetworkBasics {
    #ifndef NNB_SCALAR
    typedef float Scalar;
    #else
    typedef NNB_SCALAR Scalar;
    #endif

    #define DEBUG_MODE
}

#endif