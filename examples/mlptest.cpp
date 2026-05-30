#include <iostream>
#include <print>
#include <ranges>
#include <vector>

#include "mlp.hpp"

int main() {
    MLP mlp(3, {4, 4, 1});
    std::vector<double> ys = {1.0, 0.2, 0.5, 1.0};
    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    std::cout << "Target:";
    std::println("{}", ys);

    const double LEARNING_RATE = 0.05;

    for (int i = 0; i < 100; i++) {
        std::vector<Value> preds;
        Value loss = 0.0;

        for (auto& x : xs) {
            auto pred = mlp(x);
            preds.push_back(pred[0]);
        }

        std::cout << "[";
        for (auto& v : preds) {
            std::cout << v.data() << ",";
        }
        std::cout << "] ";

        for (auto&& [pi, yi] : std::views::zip(preds, ys)) {
            loss = loss + (pi - yi).pow(2.0);
        }

        std::cout << "loss = " << loss << "\n";

        // gradient descent: step against the gradient to reduce loss
        loss.grad(1.0);
        loss.backward();
        for (auto& param : mlp.parameters()) {
            param.data(param.data() - LEARNING_RATE * param.grad());
            param.grad(0.0);
        }
    }
}