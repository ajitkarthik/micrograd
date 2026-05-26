#include "layer.hpp"

#include <vector>

Layer::Layer(int nin, int nout) {
    for (int i = 0; i < nout; i++) {
        neurons.push_back(Neuron(nin));
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value>& x) {
    std::vector<Value> outs;
    for (auto& neuron : neurons) {
        outs.push_back(neuron(x));
    }
    return outs;
}
