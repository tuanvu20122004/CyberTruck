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

    // Output raw của MLP sau khi de-normalize về độ.
    float inferRaw(const FeatureVector& features) const;

    // Output dùng để điều khiển, đã áp dụng action limiter nếu JSON yêu cầu.
    float inferAction(const FeatureVector& features) const;

    // Giữ alias cũ nếu code cũ còn gọi infer().
    float infer(const FeatureVector& features) const { return inferAction(features); }

    static FeatureVector buildFeatures(const MpcState& state,
                                       float velocity,
                                       float prev_applied_steering_deg);

    const std::vector<std::string>& featureColumns() const { return feature_columns_; }
    const std::string& activation() const { return activation_; }

    float actionLimitDeg() const { return action_limit_deg_; }
    const std::string& actionLimitMode() const { return action_limit_mode_; }

private:
    struct Layer {
        std::vector<std::vector<float>> weight;
        std::vector<float> bias;
    };

    std::vector<float> normalize(const FeatureVector& features) const;
    std::vector<float> applyLayer(const Layer& layer,
                                  const std::vector<float>& input,
                                  bool applyActivation) const;
    float activate(float x) const;
    float applyActionLimit(float u_raw_deg) const;

    bool loaded_ = false;
    std::string activation_ = "tanh";
    std::vector<std::string> feature_columns_;
    std::vector<float> feature_mean_;
    std::vector<float> feature_std_;
    float target_mean_ = 0.0f;
    float target_std_ = 1.0f;
    std::vector<Layer> layers_;

    float action_limit_deg_ = 28.0f;
    std::string action_limit_mode_ = "none";
};

#endif // POLICY_MODEL_HPP