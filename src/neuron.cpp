#include "neuron.hpp"

#include <random>
#include <ranges>
#include <vector>

#include "value.hpp"

Neuron::Neuron(int inputs) {
    // One RNG per thread, seeded once. Reusing it across neurons avoids the
    // per-construction std::random_device + mt19937 setup cost
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    weights.reserve(inputs);
    for (int i = 0; i < inputs; ++i) {
        weights.push_back(Value(dist(rng)));
    }
    bias = Value(dist(rng));
}

Value Neuron::operator()(const std::vector<Value>& x) const {
    Value out = bias;

    for (auto&& [xi, wi] : std::views::zip(x, weights)) {
        out = out + (xi * wi);
    }
    return out.tanh(); /* output is pinned from -1.0 to 1.0 */
}

std::vector<Value> Neuron::parameters() const {
    std::vector<Value> out;
    out = weights;
    out.push_back(bias);
    return out;
}
