#include "PolicyModel.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<float> readFloatVector(const cv::FileNode& node)
{
    std::vector<float> out;
    if (node.empty()) {
        return out;
    }

    out.reserve(node.size());
    for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
        out.push_back(static_cast<float>((double)(*it)));
    }
    return out;
}

std::vector<std::vector<float>> readFloatMatrix(const cv::FileNode& node)
{
    std::vector<std::vector<float>> out;
    if (node.empty()) {
        return out;
    }

    out.reserve(node.size());
    for (cv::FileNodeIterator rowIt = node.begin(); rowIt != node.end(); ++rowIt) {
        out.push_back(readFloatVector(*rowIt));
    }
    return out;
}

std::vector<std::string> readStringVector(const cv::FileNode& node)
{
    std::vector<std::string> out;
    if (node.empty()) {
        return out;
    }

    out.reserve(node.size());
    for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
        out.emplace_back((std::string)(*it));
    }
    return out;
}

std::string joinStrings(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    return oss.str();
}

} // namespace

PolicyModel::PolicyModel(const std::string& jsonPath)
{
    std::string error;
    if (!loadFromJson(jsonPath, &error)) {
        throw std::runtime_error(error);
    }
}

bool PolicyModel::loadFromJson(const std::string& jsonPath, std::string* errorMessage)
{
    auto fail = [&](const std::string& message) {
        loaded_ = false;
        layers_.clear();
        feature_columns_.clear();
        feature_mean_.clear();
        feature_std_.clear();
        target_mean_ = 0.0f;
        target_std_ = 1.0f;
        if (errorMessage != nullptr) {
            *errorMessage = message;
        }
        return false;
    };

    cv::FileStorage fs(jsonPath, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        return fail("[PolicyModel] Cannot open JSON file: " + jsonPath);
    }

    feature_columns_ = readStringVector(fs["feature_columns"]);
    if (feature_columns_.size() != kFeatureDim) {
        return fail("[PolicyModel] Expected 8 feature columns, got " + std::to_string(feature_columns_.size()));
    }

    static const std::vector<std::string> kExpectedFeatureOrder = {
        "lateral_deviation",
        "yaw_angle",
        "curvature_0",
        "curvature_1",
        "curvature_2",
        "curvature_3",
        "velocity",
        "prev_steering",
    };

    if (feature_columns_ != kExpectedFeatureOrder) {
        return fail("[PolicyModel] Feature order mismatch. JSON has [" + joinStrings(feature_columns_)
                    + "] but runtime expects [" + joinStrings(kExpectedFeatureOrder) + "]");
    }

    activation_ = (std::string)fs["activation"];
    if (activation_ != "tanh" && activation_ != "relu") {
        return fail("[PolicyModel] Unsupported activation: " + activation_);
    }

    cv::FileNode normNode = fs["normalization"];
    if (normNode.empty()) {
        return fail("[PolicyModel] Missing normalization section");
    }

    feature_mean_ = readFloatVector(normNode["feature_mean"]);
    feature_std_  = readFloatVector(normNode["feature_std"]);
    if (feature_mean_.size() != kFeatureDim || feature_std_.size() != kFeatureDim) {
        return fail("[PolicyModel] Invalid normalization vector size");
    }

    target_mean_ = static_cast<float>((double)normNode["target_mean"]);
    target_std_  = static_cast<float>((double)normNode["target_std"]);
    if (std::abs(target_std_) < 1e-8f) {
        target_std_ = 1.0f;
    }

    cv::FileNode layersNode = fs["layers"];
    if (layersNode.empty() || !layersNode.isSeq()) {
        return fail("[PolicyModel] Missing or invalid layers sequence");
    }

    layers_.clear();
    for (cv::FileNodeIterator it = layersNode.begin(); it != layersNode.end(); ++it) {
        Layer layer;
        layer.weight = readFloatMatrix((*it)["weight"]);
        layer.bias   = readFloatVector((*it)["bias"]);

        if (layer.weight.empty() || layer.bias.empty()) {
            return fail("[PolicyModel] Empty weight or bias in a layer");
        }

        const std::size_t outDim = layer.weight.size();
        const std::size_t inDim = layer.weight.front().size();
        if (outDim != layer.bias.size()) {
            return fail("[PolicyModel] Bias size mismatch in a layer");
        }
        for (const auto& row : layer.weight) {
            if (row.size() != inDim) {
                return fail("[PolicyModel] Ragged weight matrix in a layer");
            }
        }

        layers_.push_back(std::move(layer));
    }

    if (layers_.empty()) {
        return fail("[PolicyModel] No layers found");
    }

    if (layers_.front().weight.front().size() != kFeatureDim) {
        return fail("[PolicyModel] First layer input dim does not match feature dim");
    }

    if (layers_.back().weight.size() != 1u || layers_.back().bias.size() != 1u) {
        return fail("[PolicyModel] Final layer must output exactly one value");
    }

    loaded_ = true;
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }
    return true;
}

float PolicyModel::activate(float x) const
{
    if (activation_ == "relu") {
        return std::max(0.0f, x);
    }
    return std::tanh(x);
}

std::vector<float> PolicyModel::normalize(const FeatureVector& features) const
{
    std::vector<float> out(kFeatureDim, 0.0f);
    for (int i = 0; i < kFeatureDim; ++i) {
        const float stdv = std::abs(feature_std_[i]) < 1e-8f ? 1.0f : feature_std_[i];
        out[i] = (features[static_cast<std::size_t>(i)] - feature_mean_[static_cast<std::size_t>(i)]) / stdv;
    }
    return out;
}

std::vector<float> PolicyModel::applyLayer(const Layer& layer,
                                           const std::vector<float>& input,
                                           bool applyActivation) const
{
    std::vector<float> output(layer.bias.size(), 0.0f);
    for (std::size_t row = 0; row < layer.weight.size(); ++row) {
        float acc = layer.bias[row];
        for (std::size_t col = 0; col < layer.weight[row].size(); ++col) {
            acc += layer.weight[row][col] * input[col];
        }
        output[row] = applyActivation ? activate(acc) : acc;
    }
    return output;
}

float PolicyModel::infer(const FeatureVector& features) const
{
    if (!loaded_) {
        throw std::runtime_error("[PolicyModel] infer() called before loadFromJson()");
    }

    std::vector<float> x = normalize(features);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const bool applyAct = (i + 1u != layers_.size());
        x = applyLayer(layers_[i], x, applyAct);
    }

    if (x.empty() || !std::isfinite(x[0])) {
        throw std::runtime_error("[PolicyModel] Invalid model output");
    }

    return x[0] * target_std_ + target_mean_;
}

PolicyModel::FeatureVector PolicyModel::buildFeatures(const MpcState& state,
                                                      float velocity,
                                                      float prev_raw_steering)
{
    FeatureVector f{};
    f[0] = static_cast<float>(state.lateral_deviation);
    f[1] = static_cast<float>(state.yaw_angle);
    f[2] = state.curvature.size() > 0 ? state.curvature[0] : 0.0f;
    f[3] = state.curvature.size() > 1 ? state.curvature[1] : 0.0f;
    f[4] = state.curvature.size() > 2 ? state.curvature[2] : 0.0f;
    f[5] = state.curvature.size() > 3 ? state.curvature[3] : 0.0f;
    f[6] = velocity;
    f[7] = prev_raw_steering;
    return f;
}
