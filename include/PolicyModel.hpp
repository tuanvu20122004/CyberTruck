#ifndef POLICY_MODEL_HPP
#define POLICY_MODEL_HPP

#include "MpcController.hpp"
#include <array>
#include <string>
#include <vector>

class PolicyModel {
public:
    static constexpr int kFeatureDim = 8;
    using FeatureVector = std::array<float, kFeatureDim>;

    PolicyModel() = default;
    explicit PolicyModel(const std::string& jsonPath);

    bool loadFromJson(const std::string& jsonPath, std::string* errorMessage = nullptr);
    bool isLoaded() const { return loaded_; }

    float infer(const FeatureVector& features) const;

    static FeatureVector buildFeatures(const MpcState& state,
                                       float velocity,
                                       float prev_raw_steering);

    const std::vector<std::string>& featureColumns() const { return feature_columns_; }
    const std::string& activation() const { return activation_; }

private:
    struct Layer {
        std::vector<std::vector<float>> weight; // [out_dim][in_dim]
        std::vector<float> bias;                // [out_dim]
    };

    std::vector<float> normalize(const FeatureVector& features) const;
    std::vector<float> applyLayer(const Layer& layer,
                                  const std::vector<float>& input,
                                  bool applyActivation) const;
    float activate(float x) const;

    bool loaded_ = false;
    std::string activation_ = "tanh";
    std::vector<std::string> feature_columns_;
    std::vector<float> feature_mean_;
    std::vector<float> feature_std_;
    float target_mean_ = 0.0f;
    float target_std_ = 1.0f;
    std::vector<Layer> layers_;
};

#endif // POLICY_MODEL_HPP
