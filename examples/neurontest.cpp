#include <iostream>
#include <vector>

#include "mlp.hpp"

int main(void) {
    // Neuron n(2);
    // std::vector x = {1.0, 2.0};
    // Value fwd = n(x);
    // std::cout << "Computing backward ..." << std::endl;
    // fwd.grad(1.0);
    // fwd.backward();
    // fwd.printGraph(std::cout);

    // Layer layer = Layer(2, 3);
    // std::vector<double> x = {1.0, 2.0};
    // std::cout << layer(x);

    std::vector x = {2.0, 3.0, -1.0};
    MLP mlp = MLP(3, std::vector{4, 4, 1});
    std::vector<Value> out = mlp(x);

    std::cout << out << std::endl;
}
