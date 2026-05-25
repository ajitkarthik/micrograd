#include <vector>

#include "value.hpp"

class Neuron {
   private:
    // inputs of the neuron
    std::vector<Value> weights;
    Value bias;
    int inputs;

   public:
    Neuron(int inputs);

    // Operator overloads
    Value operator()(std::vector<double>);
};
