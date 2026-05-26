#include "mlp.hpp"

MLP::MLP(int nin, std::vector<int> nouts) {
    std::vector<int> mlpsize = nouts;
    mlpsize.insert(mlpsize.begin(), nin);
    for (size_t i = 0; i + 1 < mlpsize.size(); ++i) {
        layers.push_back(Layer(mlpsize[i], mlpsize[i + 1]));
    }
}

std::vector<Value> MLP::operator()(const std::vector<double>& x) {
    std::vector<Value> out(x.begin(), x.end());
    for (auto& layer : layers) {
        out = layer(out);
    }
    return out;
}
