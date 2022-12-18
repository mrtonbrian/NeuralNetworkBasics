#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <memory>
#include <vector>
#include <cassert>
#include <stdexcept>
#include "Config.hpp"
#include "Layer.hpp"
#include "LossFunction.hpp"

namespace NeuralNetworkBasics {

class Network {
    std::vector<std::unique_ptr<ILayer>> layers;

    public:
    Network() {};

    Network(std::initializer_list<std::unique_ptr<ILayer>> layerList) {
        for (const auto& ptr : layerList) {
            layers.push_back(ptr);
        }
    }

    Network(std::initializer_list<ILayer*> layerList) {
        for (const auto& ptr : layerList) {
            layers.push_back(std::unique_ptr<ILayer>(ptr));
        }
    }

    /** 
     * Calculates the output of the function
     * 
     * Is more complicated than it should be. Used intermediate variable
     * prevLayerOutput to prevent expensive copy operation for input since
     * we don't want to overwrite input.
     */
    std::vector<Scalar> calculateOutput(std::vector<Scalar>& input) {
        assert(layers.size() > 0);

        std::vector<Scalar> prevLayerOutput = layers[0]->forward(input);
        for (unsigned int i = 1; i < prevLayerOutput.size(); i++) {
            prevLayerOutput = layers[i]->forward(prevLayerOutput);
        }

        return prevLayerOutput;
    }

    std::vector<Scalar> calculateGradients(std::vector<std::vector<Scalar>> batchInputs, std::vector<std::vector<Scalar>> batchOutputs) {
        
    }
};

}

#endif
