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

    for (auto element : weights) {
        std::cout << element << " " << std::endl;
    }

    std::cout << "Bias = " << bias << std::endl;
}

Value Neuron::operator()(std::vector<double> x) {
    Value output = bias;

    for (auto&& [x, weight] : std::views::zip(x, weights)) {
        output = output + (x * weight);
    }
    return output.tanh();
}