#include <iostream>
#include <memory>
#include "../../include/NeuralNetworkBasics.hpp"

using namespace NeuralNetworkBasics;

int main() {
    // Should fail to compile since we have a bad parameter to layer
    auto layer = std::unique_ptr<Layer<ReLU>> (new Layer<ReLU>(100, 100));
}