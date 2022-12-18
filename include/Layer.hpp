#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <vector>
#include "Config.hpp"
#include "Helper.hpp"
#include <random>

namespace NeuralNetworkBasics {

class ILayer {
    public:
    virtual std::vector<Scalar> forward(std::vector<Scalar>& input) { return std::vector<Scalar>(); };
};

template<class ActivationFunction>
class Layer : public ILayer {
    static_assert(is_activation(ActivationFunction), "Layer ActivationFunction parameter not valid!");
    protected:
    const int numInputNodes;
    const int numOutputNodes;

    std::vector<Scalar> weights; // Weights for incoming connections
    std::vector<Scalar> biases; // For each node

    std::vector<Scalar> weightGradients; // Gradients of weights
    std::vector<Scalar> biasGradients; // Gradients of each bias

    private:
    // Fills vector with random scalar scalar values from -1 to 1, uniform
    void fillRandom(std::vector<Scalar>& vec) {
        std::random_device rd;
        std::mt19937 mersenne(rd());
        std::uniform_real_distribution<Scalar> dist(-1, 1);
        auto generator = [&dist, &mersenne]() {
            return dist(mersenne);
        };

        std::generate(vec.begin(), vec.end(), generator);
    }

    Scalar getWeightIndex(int inpNodeIndex, int outNodeIndex) {
        return (outNodeIndex * numOutputNodes) + inpNodeIndex;
    }

    public:

    // Constructor
    Layer(int inputSize, int outputSize) : numInputNodes(inputSize), numOutputNodes(outputSize) {
        // Each node in the layer (output) connects to all nodes of previous layers,
        // So there are # input * # output weights
        // Mapping: input node i to output node j will have weight in slot (j * # output nodes) + i
        this->weights = std::vector<Scalar>(inputSize * outputSize);
        this->weightGradients = std::vector<Scalar>(inputSize * outputSize);

        // Each node in the layer (output node) only has 1 bias
        this->biases = std::vector<Scalar>(outputSize); 
        this->biasGradients = std::vector<Scalar>(outputSize);

        fillRandom(this->weights);
        fillRandom(this->biases);
    }
    
    std::vector<Scalar> forward(std::vector<Scalar>& input) {
        std::vector<Scalar> output(numOutputNodes, 0);

        // Apply each bias and weight
        for (int out = 0; out < numOutputNodes; out++) {
            for (int inp = 0; inp < numInputNodes; inp++) {
                output[out] += weights[getWeightIndex(inp, out)] * input[inp];
            }

            output[out] += biases[out];
        }

        return ActivationFunction::activate(output);
    }
};


}

#endif