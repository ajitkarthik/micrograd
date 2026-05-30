#pragma once

#include <vector>

#include "neuron.hpp"
#include "value.hpp"

class Layer {
    std::vector<Neuron> neurons;

   public:
    // nin = number of inputs to this layer
    // nout = number of neurons in this layer = number of outputs from this layer
    Layer(int nin, int nout);
    std::vector<Value> operator()(const std::vector<Value>&) const;
    std::vector<Value> parameters() const;
};