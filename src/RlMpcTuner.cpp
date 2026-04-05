#include "RlMpcTuner.hpp"

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kQ1Min = 150.0f;
constexpr float kQ1Max = 2000.0f;
constexpr float kQ2Min = 10.0f;
constexpr float kQ2Max = 300.0f;
constexpr float kRMin  = 1.0f;
constexpr float kRMax  = 20.0f;
}

RlMpcTuner::RlMpcTuner()
    : replay_capacity_(256),
      batch_size_(16),
      lr_actor_(0.0035f),
      weight_decay_(1e-4f),
      alpha_imitation_(0.7f),
      beta_q_(0.3f),
      q1_delta_max_(220.0f),
      q2_delta_max_(60.0f),
      r_delta_max_(3.0f),
      has_last_action_(false)
{
    // Small non-zero initialization so the learner can start adapting immediately.
    actor_q1_ = {0.00f, 0.08f, 0.02f, 0.00f, 0.00f, -0.01f};
    actor_q2_ = {0.00f, 0.01f, 0.10f, 0.02f, 0.00f, -0.01f};
    actor_r_  = {0.00f, -0.03f, -0.02f, -0.05f, 0.00f, 0.06f};
}

float RlMpcTuner::clamp(float v, float min_v, float max_v) const
{
    return std::max(min_v, std::min(v, max_v));
}

RlMpcTuner::FeatureVector RlMpcTuner::buildFeatures(const RlMpcState& s) const
{
    const float lat  = clamp(std::fabs(s.lateral_error), 0.0f, 0.75f);
    const float yaw  = clamp(std::fabs(s.yaw_error),     0.0f, 0.75f);
    const float curv = clamp(std::fabs(s.curvature),     0.0f, 4.00f);
    const float vel  = clamp(std::fabs(s.velocity),      0.0f, 0.20f);
    const float prev = clamp(std::fabs(s.prev_steering), 0.0f, 25.0f);

    FeatureVector phi{};
    phi[0] = 1.0f;
    phi[1] = lat  / 0.25f;
    phi[2] = yaw  / 0.25f;
    phi[3] = curv / 2.00f;
    phi[4] = vel  / 0.10f;
    phi[5] = prev / 15.0f;
    return phi;
}

float RlMpcTuner::dot(const ParameterVector& w, const FeatureVector& x) const
{
    float acc = 0.0f;
    for (std::size_t i = 0; i < kFeatureDim; ++i) {
        acc += w[i] * x[i];
    }
    return acc;
}

RlMpcWeights RlMpcTuner::getExpertWeights(const RlMpcState& s) const
{
    const float lat  = clamp(std::fabs(s.lateral_error), 0.0f, 0.60f);
    const float yaw  = clamp(std::fabs(s.yaw_error),     0.0f, 0.50f);
    const float curv = clamp(std::fabs(s.curvature),     0.0f, 3.00f);
    const float prev = clamp(std::fabs(s.prev_steering), 0.0f, 25.0f);

    // Safe expert prior used as the imitation anchor.
    float Q1 = 900.0f + 1400.0f * lat + 8.0f * prev;
    float Q2 =  70.0f + 180.0f * yaw + 20.0f * curv;
    float R  =   6.0f + 0.08f * prev - 1.0f * curv - 2.0f * lat - 1.0f * yaw;

    Q1 = clamp(Q1, 400.0f, kQ1Max);
    Q2 = clamp(Q2,  30.0f, kQ2Max);
    R  = clamp(R,    1.5f, 12.0f);

    return {Q1, Q2, R};
}

RlMpcWeights RlMpcTuner::actorForward(const FeatureVector& features,
                                      const RlMpcWeights& expert) const
{
    const float raw_q1 = dot(actor_q1_, features);
    const float raw_q2 = dot(actor_q2_, features);
    const float raw_r  = dot(actor_r_,  features);

    const float delta_q1 = q1_delta_max_ * std::tanh(raw_q1);
    const float delta_q2 = q2_delta_max_ * std::tanh(raw_q2);
    const float delta_r  = r_delta_max_  * std::tanh(raw_r);

    RlMpcWeights out{};
    out.Q1 = clamp(expert.Q1 + delta_q1, kQ1Min, kQ1Max);
    out.Q2 = clamp(expert.Q2 + delta_q2, kQ2Min, kQ2Max);
    out.R  = clamp(expert.R  + delta_r,  kRMin,  kRMax);
    return out;
}

RlMpcWeights RlMpcTuner::infer(const RlMpcState& s)
{
    const FeatureVector features = buildFeatures(s);
    const RlMpcWeights expert = getExpertWeights(s);

    RlMpcWeights candidate = actorForward(features, expert);

    // Mild smoothing to avoid abrupt online retuning of the QP.
    if (has_last_action_) {
        candidate.Q1 = 0.75f * last_action_.Q1 + 0.25f * candidate.Q1;
        candidate.Q2 = 0.75f * last_action_.Q2 + 0.25f * candidate.Q2;
        candidate.R  = 0.75f * last_action_.R  + 0.25f * candidate.R;
    }

    candidate.Q1 = clamp(candidate.Q1, kQ1Min, kQ1Max);
    candidate.Q2 = clamp(candidate.Q2, kQ2Min, kQ2Max);
    candidate.R  = clamp(candidate.R,  kRMin,  kRMax);

    last_state_ = s;
    last_action_ = candidate;
    has_last_action_ = true;

    return last_action_;
}

float RlMpcTuner::computeStageCost(const RlMpcState& s,
                                   float steering,
                                   float prev_steering) const
{
    const float lat   = clamp(std::fabs(s.lateral_error), 0.0f, 1.50f);
    const float yaw   = clamp(std::fabs(s.yaw_error),     0.0f, 1.20f);
    const float curv  = clamp(std::fabs(s.curvature),     0.0f, 5.00f);
    const float steer = clamp(std::fabs(steering),        0.0f, 25.0f);
    const float delta = clamp(std::fabs(steering - prev_steering), 0.0f, 25.0f);

    // Surrogate of the MPC stage cost in degree units.
    return 2.5f * lat * lat
         + 1.2f * yaw * yaw
         + 0.0025f * steer * steer
         + 0.0040f * delta * delta
         + 0.0300f * curv;
}

float RlMpcTuner::computeReward(const RlMpcState& s,
                                float steering,
                                float prev_steering) const
{
    return -computeStageCost(s, steering, prev_steering);
}

RlMpcWeights RlMpcTuner::buildHybridTarget(const RlMpcState& s,
                                           float steering,
                                           float prev_steering,
                                           const RlMpcWeights& expert) const
{
    const float lat   = clamp(std::fabs(s.lateral_error), 0.0f, 0.60f);
    const float yaw   = clamp(std::fabs(s.yaw_error),     0.0f, 0.50f);
    const float curv  = clamp(std::fabs(s.curvature),     0.0f, 3.00f);
    const float steer = clamp(std::fabs(steering),        0.0f, 25.0f);
    const float delta = clamp(std::fabs(steering - prev_steering), 0.0f, 25.0f);

    const float stage_cost = computeStageCost(s, steering, prev_steering);
    const float severity = clamp(stage_cost / 0.35f, 0.0f, 3.0f);

    // Task-aware target (Q-loss proxy):
    // - increase Q1/Q2 when tracking errors are large
    // - increase R when steering is too aggressive or oscillatory
    // - reduce R a bit on tighter curvature so the controller can still turn
    RlMpcWeights q_target{};
    q_target.Q1 = clamp(expert.Q1 + 180.0f * lat * (1.0f + 0.40f * severity), kQ1Min, kQ1Max);
    q_target.Q2 = clamp(expert.Q2 + 120.0f * yaw * (1.0f + 0.40f * severity), kQ2Min, kQ2Max);
    q_target.R  = clamp(expert.R  + 0.10f * steer + 0.18f * delta - 0.70f * curv, kRMin, kRMax);

    const float norm = alpha_imitation_ + beta_q_;
    RlMpcWeights hybrid{};
    hybrid.Q1 = (alpha_imitation_ * expert.Q1 + beta_q_ * q_target.Q1) / norm;
    hybrid.Q2 = (alpha_imitation_ * expert.Q2 + beta_q_ * q_target.Q2) / norm;
    hybrid.R  = (alpha_imitation_ * expert.R  + beta_q_ * q_target.R ) / norm;

    hybrid.Q1 = clamp(hybrid.Q1, kQ1Min, kQ1Max);
    hybrid.Q2 = clamp(hybrid.Q2, kQ2Min, kQ2Max);
    hybrid.R  = clamp(hybrid.R,  kRMin,  kRMax);
    return hybrid;
}

void RlMpcTuner::pushSample(const TrainingSample& sample)
{
    replay_buffer_.push_back(sample);
    while (replay_buffer_.size() > replay_capacity_) {
        replay_buffer_.pop_front();
    }
}

void RlMpcTuner::sgdStep(ParameterVector& params,
                         float raw,
                         float target_delta,
                         const FeatureVector& features,
                         float delta_scale)
{
    const float pred_norm = std::tanh(raw);
    const float target_norm = clamp(target_delta / delta_scale, -1.0f, 1.0f);
    const float grad_common = (pred_norm - target_norm) * (1.0f - pred_norm * pred_norm);

    for (std::size_t i = 0; i < kFeatureDim; ++i) {
        const float grad = grad_common * features[i] + weight_decay_ * params[i];
        params[i] -= lr_actor_ * grad;
    }
}

void RlMpcTuner::trainFromBuffer()
{
    if (replay_buffer_.empty()) {
        return;
    }

    const std::size_t batch = std::min(batch_size_, replay_buffer_.size());
    const std::size_t start = replay_buffer_.size() - batch;

    for (std::size_t i = start; i < replay_buffer_.size(); ++i) {
        const TrainingSample& sample = replay_buffer_[i];

        const float raw_q1 = dot(actor_q1_, sample.features);
        const float raw_q2 = dot(actor_q2_, sample.features);
        const float raw_r  = dot(actor_r_,  sample.features);

        const float pred_delta_q1 = q1_delta_max_ * std::tanh(raw_q1);
        const float pred_delta_q2 = q2_delta_max_ * std::tanh(raw_q2);
        const float pred_delta_r  = r_delta_max_  * std::tanh(raw_r);

        const float target_delta_q1 = clamp(sample.target.Q1 - sample.expert.Q1,
                                            -q1_delta_max_, q1_delta_max_);
        const float target_delta_q2 = clamp(sample.target.Q2 - sample.expert.Q2,
                                            -q2_delta_max_, q2_delta_max_);
        const float target_delta_r  = clamp(sample.target.R  - sample.expert.R,
                                            -r_delta_max_,  r_delta_max_);

        sgdStep(actor_q1_, raw_q1, target_delta_q1, sample.features, q1_delta_max_);
        sgdStep(actor_q2_, raw_q2, target_delta_q2, sample.features, q2_delta_max_);
        sgdStep(actor_r_,  raw_r,  target_delta_r,  sample.features, r_delta_max_);
    }
}

void RlMpcTuner::update(const RlMpcState& current_state,
                        float steering,
                        float prev_steering)
{
    const FeatureVector features = buildFeatures(current_state);
    const RlMpcWeights expert = getExpertWeights(current_state);
    const RlMpcWeights target = buildHybridTarget(current_state,
                                                  steering,
                                                  prev_steering,
                                                  expert);

    pushSample({features, expert, target});
    trainFromBuffer();
}
