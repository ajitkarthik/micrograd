#pragma once

#include <vector>

#include "layer.hpp"

class MLP {
    std::vector<Layer> layers;

   public:
    MLP(int nin, std::vector<int> nouts);
    std::vector<Value> operator()(const std::vector<double>& x);
};