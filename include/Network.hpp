#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include <stdexcept>
#include "Config.hpp"
#include "Layer.hpp"

namespace NeuralNetworkBasics {

class Network {
    protected:
    std::vector<Layer> layers;    
};

}

#endif
