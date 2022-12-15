#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <memory>
#include <vector>
#include <stdexcept>
#include "Config.hpp"
#include "Layer.hpp"
#include "ActivationFunctions/ReLU.hpp"

namespace NeuralNetworkBasics {

class Network {
    std::vector<std::unique_ptr<ILayer>> layers;
};

}

#endif
