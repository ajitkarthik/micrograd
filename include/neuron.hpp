#pragma once
#include <vector>

#include "value.hpp"

class Neuron {
   private:
    std::vector<Value> weights;
    Value bias;

   public:
    Neuron(int inputs);

    // Operator overloads
    Value operator()(const std::vector<Value>&) const;
    // return a vector of parameters (weights and bias for this neuron)
    std::vector<Value> parameters() const;
};
