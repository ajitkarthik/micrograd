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
    Value operator()(const std::vector<Value>&);
};
