#include "neuron.hpp"

#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "value.hpp"

Neuron::Neuron(int inputs) : inputs(inputs) {
    // initialize the vector to random weights
    std::random_device rd;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    weights.reserve(inputs);
    for (int i = 0; i < inputs; ++i) {
        weights.push_back(Value(dist(rng)));
    }
    bias = Value(dist(rng));
}

Value Neuron::operator()(std::vector<double> x) {
    Value out = bias;

    for (auto&& [x, weight] : std::views::zip(x, weights)) {
        out = out + (x * weight);
    }
    return out.tanh();
}