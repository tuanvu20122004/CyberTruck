#include "RlMpcTuner.hpp"
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

RlMpcTuner::RlMpcTuner()
{
    // Lightweight adaptive parameters
    actor_w_ = {1.0f, 1.0f, 1.0f, 1.0f};
    critic_w_ = {1.0f, 1.0f, 1.0f, 1.0f};

    gamma_ = 0.95f;
    lr_actor_ = 0.001f;
    lr_critic_ = 0.001f;
}

float RlMpcTuner::clamp(float v, float min_v, float max_v) const
{
    if (v < min_v) return min_v;
    if (v > max_v) return max_v;
    return v;
}

float RlMpcTuner::deg2rad(float deg) const
{
    return deg * kPi / 180.0f;
}

RlMpcWeights RlMpcTuner::infer(const RlMpcState& s)
{
    // Keep policy simple and deterministic: RL only adapts MPC cost weights.
    // Q1: lateral tracking importance
    // Q2: yaw tracking importance
    // R : steering effort / smoothness importance
    
    const float abs_lat  = std::fabs(s.lateral_error);
    const float abs_yaw  = std::fabs(s.yaw_error);
    const float abs_curv = std::fabs(s.curvature);

    float Q1 = 500.0f + 500.0f * abs_lat;
    float Q2 =  50.0f + 100.0f * abs_yaw;
    float R  =   5.0f +   2.0f * abs_curv;

    // Optional lightweight actor scaling.
    Q1 *= clamp(actor_w_[0], 0.5f, 2.0f);
    Q2 *= clamp(actor_w_[1], 0.5f, 2.0f);
    R  *= clamp(actor_w_[2], 0.5f, 2.0f);

    // Safety bounds.
    Q1 = clamp(Q1, 100.0f, 2000.0f);
    Q2 = clamp(Q2,  10.0f,  300.0f);
    R  = clamp(R,    1.0f,   20.0f);

    last_state_ = s;
    last_action_ = {Q1, Q2, R};

    return last_action_;
}

float RlMpcTuner::computeReward(const RlMpcState& s,
                                float steering_deg,
                                float prev_steering_deg) const
{
    // IMPORTANT:
    // steering inputs come from the controller in degrees.
    // Convert to radians here so reward terms are consistent with yaw_error [rad].
    const float steering_rad = deg2rad(steering_deg);
    const float prev_steering_rad = deg2rad(prev_steering_deg);
    const float delta_rad = steering_rad - prev_steering_rad;

    // Reward aligned with MPC-style objective:
    // - phạt lệch ngang
    // - phạt lệch yaw
    // - phạt góc lái lớn
    // - phạt việc thay đổi góc lái quá nhanh
    const float reward =
        - (2.0f  * s.lateral_error * s.lateral_error
        + 1.0f  * s.yaw_error      * s.yaw_error
        + 0.05f * steering_rad     * steering_rad
        + 0.2f  * delta_rad        * delta_rad);

    return reward;
}

void RlMpcTuner::update(float reward, const RlMpcState& next_state)
{
    // Simple critic using error magnitude features.
    const float V_current =
        critic_w_[0] * std::fabs(last_state_.lateral_error) +
        critic_w_[1] * std::fabs(last_state_.yaw_error);

    const float V_next =
        critic_w_[0] * std::fabs(next_state.lateral_error) +
        critic_w_[1] * std::fabs(next_state.yaw_error);

    // TD error
    const float td_error = reward + gamma_ * V_next - V_current;

    // Critic update.
    critic_w_[0] += lr_critic_ * td_error * std::fabs(last_state_.lateral_error);
    critic_w_[1] += lr_critic_ * td_error * std::fabs(last_state_.yaw_error);

    // Lightweight actor update.
    actor_w_[0] += lr_actor_ * td_error;
    actor_w_[1] += lr_actor_ * td_error;
    actor_w_[2] += 0.5f * lr_actor_ * td_error;

    // Keep actor gains bounded.
    actor_w_[0] = clamp(actor_w_[0], 0.5f, 2.0f);
    actor_w_[1] = clamp(actor_w_[1], 0.5f, 2.0f);
    actor_w_[2] = clamp(actor_w_[2], 0.5f, 2.0f);
}
