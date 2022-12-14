#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>
#include "Config.hpp"
#include "Helper.hpp"

namespace NeuralNetworkBasics {

class ILayer {

};

template<class ActivationFunction>
class Layer : public ILayer {
    static_assert(is_activation(ActivationFunction), "Layer ActivationFunction parameter not valid!");
    protected:
    const int numInputNodes;
    const int numOutputNodes;

    std::vector<Scalar> weights; // Weights for incoming connections
    std::vector<Scalar> biases; // For each node

    public:

    // Constructor
    Layer(int inputSize, int outputSize) : numInputNodes(inputSize), numOutputNodes(outputSize) {}
};


}

#endif