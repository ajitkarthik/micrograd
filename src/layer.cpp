#include "layer.hpp"

#include <vector>

Layer::Layer(int nin, int nout) : nin{nin}, nout{nout} {
    // Initialize the neurons with random weights
    for (int i = 0; i < nout; i++) {
        neurons.push_back(Neuron(nin));
    }
}

std::vector<Value> Layer::operator()(std::vector<double> x) {
    std::vector<Value> outs;
    for (auto& neuron : neurons) {
        outs.push_back(neuron(x));
    }
    return outs;
}
