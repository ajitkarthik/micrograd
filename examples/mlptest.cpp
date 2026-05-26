#include <iostream>
#include <ranges>
#include <vector>

#include "mlp.hpp"

int main(void) {
    MLP mlp = MLP(3, std::vector{4, 4, 1});
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};
    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    // vector to hold predictions
    std::vector<Value> preds;

    for (auto& x : xs) {
        auto pred = mlp(x);
        preds.push_back(pred[0]);
    }

    // std::cout << preds[0];
    // preds[0].printGraph(std::cout);

    Value mse = 0.0;
    for (auto&& [pi, yi] : std::views::zip(preds, ys)) {
        std::cout << "pi" << pi << std::endl;
        std::cout << "yi" << yi << std::endl;
        mse = mse + (pi - Value(yi).pow(2.0));
    }

    std::cout << mse;
}