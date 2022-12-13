#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>
#include "Config.hpp"
#include "ActivationFunction.hpp"

namespace NeuralNetworkBasics {

class Layer {
    protected:
    const int numInputNodes;
    const int numOutputNodes;

    std::vector<Scalar> weights; // Weights for incoming connections
    std::vector<Scalar> biases; // For each node

    ActivationFunction activationFunction; // Our activation function

    public:

    // Constructor
    Layer(int inputSize, int outputSize, ActivationFunction activation) : 
        numInputNodes(inputSize), numOutputNodes(outputSize), activationFunction(activation) {}
};


}

#endif