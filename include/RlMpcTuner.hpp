#pragma once

#include <array>
#include <cstddef>
#include <deque>

struct RlMpcState
{
    float lateral_error;
    float yaw_error;
    float velocity;
    float curvature;
    float prev_steering;
};

struct RlMpcWeights
{
    float Q1;
    float Q2;
    float R;
};

class RlMpcTuner
{
public:
    RlMpcTuner();

    // Infer online MPC weights from the current state.
    // The learner only predicts a bounded correction around a safe expert prior.
    RlMpcWeights infer(const RlMpcState& s);

    // Hand-crafted expert prior used as imitation anchor.
    RlMpcWeights getExpertWeights(const RlMpcState& s) const;

    // MPC-like stage cost surrogate (Q-loss proxy).
    float computeStageCost(const RlMpcState& s,
                           float steering,
                           float prev_steering) const;

    // Kept for compatibility with old logging / reward style.
    float computeReward(const RlMpcState& s,
                        float steering,
                        float prev_steering) const;

    // Hybrid online update on learner-induced states (DAgger-lite replay).
    void update(const RlMpcState& current_state,
                float steering,
                float prev_steering);

private:
    static constexpr std::size_t kFeatureDim = 6;
    using FeatureVector = std::array<float, kFeatureDim>;
    using ParameterVector = std::array<float, kFeatureDim>;

    struct TrainingSample
    {
        FeatureVector features{};
        RlMpcWeights expert{};
        RlMpcWeights target{};
    };

    float clamp(float v, float min_v, float max_v) const;
    FeatureVector buildFeatures(const RlMpcState& s) const;
    float dot(const ParameterVector& w, const FeatureVector& x) const;

    RlMpcWeights actorForward(const FeatureVector& features,
                              const RlMpcWeights& expert) const;
    RlMpcWeights buildHybridTarget(const RlMpcState& s,
                                   float steering,
                                   float prev_steering,
                                   const RlMpcWeights& expert) const;

    void pushSample(const TrainingSample& sample);
    void trainFromBuffer();
    void sgdStep(ParameterVector& params,
                 float raw,
                 float target_delta,
                 const FeatureVector& features,
                 float delta_scale);

private:
    // Three small linear heads that predict bounded corrections around the expert prior.
    ParameterVector actor_q1_{};
    ParameterVector actor_q2_{};
    ParameterVector actor_r_{};

    std::deque<TrainingSample> replay_buffer_;

    std::size_t replay_capacity_;
    std::size_t batch_size_;

    float lr_actor_;
    float weight_decay_;

    // Hybrid loss weights.
    float alpha_imitation_;
    float beta_q_;

    // Maximum safe correction around the expert prior.
    float q1_delta_max_;
    float q2_delta_max_;
    float r_delta_max_;

    bool has_last_action_;
    RlMpcState last_state_{};
    RlMpcWeights last_action_{};
};
