#include "neuron.hpp"

#include <random>
#include <ranges>
#include <vector>

#include "value.hpp"

Neuron::Neuron(int inputs) {
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    weights.reserve(inputs);
    for (int i = 0; i < inputs; ++i) {
        weights.push_back(Value(dist(rng)));
    }
    bias = Value(dist(rng));
}

Value Neuron::operator()(const std::vector<Value>& x) {
    Value out = bias;

    for (auto&& [xi, wi] : std::views::zip(x, weights)) {
        out = out + (xi * wi);
    }
    return out.tanh();
}
