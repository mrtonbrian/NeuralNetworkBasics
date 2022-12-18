#include <iostream>
#include "../../include/NeuralNetworkBasics.hpp"

using namespace NeuralNetworkBasics;

int main() {
    // Should compile
    auto layer1 = Layer<ReLU>(700, 700);

    // Should fail to compile since we have a bad parameter to layer
    // auto layer = Layer<int> (100, 100);
}