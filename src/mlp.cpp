#include "mlp.hpp"

#include <vector>

#include "value.hpp"

MLP::MLP(int nin, std::vector<int> nouts) {
    std::vector<int> mlpsize = nouts;
    mlpsize.insert(mlpsize.begin(), nin);
    for (size_t i = 0; i + 1 < mlpsize.size(); ++i) {
        layers.push_back(Layer(mlpsize[i], mlpsize[i + 1]));
    }
}

std::vector<Value> MLP::operator()(const std::vector<double>& x) const {
    std::vector<Value> out(x.begin(), x.end());
    for (const auto& layer : layers) {
        out = layer(out);
    }
    return out;
}

std::vector<Value> MLP::parameters() const {
    std::vector<Value> out;
    for (const auto& layer : layers) {
        for (auto& param : layer.parameters()) {
            out.push_back(param);
        }
    }
    return out;
}
