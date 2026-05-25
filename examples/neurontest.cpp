#include <iostream>

#include "layer.hpp"

int main(void) {
    // Neuron n(2);
    // std::vector x = {1.0, 2.0};
    // Value fwd = n(x);
    // std::cout << "Computing backward ..." << std::endl;
    // fwd.grad(1.0);
    // fwd.backward();
    // std::cout << fwd;

    Layer layer = Layer(2, 3);
    std::vector<double> x = {1.0, 2.0};
    std::cout << layer(x);
}
