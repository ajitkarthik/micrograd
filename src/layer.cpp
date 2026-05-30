#include "layer.hpp"

#include <vector>

Layer::Layer(int nin, int nout) {
    for (int i = 0; i < nout; i++) {
        neurons.push_back(Neuron(nin));
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value>& x) const {
    std::vector<Value> outs;
    for (const auto& neuron : neurons) {
        outs.push_back(neuron(x));
    }
    return outs;
}

std::vector<Value> Layer::parameters() const {
    std::vector<Value> out;
    for (const auto& neuron : neurons) {
        for (auto& param : neuron.parameters()) {
            out.push_back(param);
        }
    }
    return out;
}
