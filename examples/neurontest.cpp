#include <iostream>

#include "neuron.hpp"

int main(void) {
    Neuron n(2);
    std::vector x = {1.0, 2.0};
    std::cout << n(x);
}
