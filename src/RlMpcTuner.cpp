#include "RlMpcTuner.hpp"
#include <cmath>

RlMpcTuner::RlMpcTuner()
{
    // init simple weights
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

RlMpcWeights RlMpcTuner::infer(const RlMpcState& s)
{
    // very simple linear policy
    float Q1 = 500.0f + 500.0f * std::fabs(s.lateral_error);
    float Q2 = 50.0f  + 100.0f * std::fabs(s.yaw_error);
    float R  = 5.0f   + 2.0f * std::fabs(s.curvature);

    // safety bounds (VERY IMPORTANT)
    Q1 = clamp(Q1, 100.0f, 2000.0f);
    Q2 = clamp(Q2, 10.0f, 300.0f);
    R  = clamp(R, 1.0f, 20.0f);

    last_state_ = s;
    last_action_ = {Q1, Q2, R};

    return last_action_;
}

float RlMpcTuner::computeReward(const RlMpcState& s,
                                float steering,
                                float prev_steering) const
{
    float delta = steering - prev_steering;

    float reward =
        - (2.0f * s.lateral_error * s.lateral_error
        + 1.0f * s.yaw_error * s.yaw_error
        + 0.05f * steering * steering
        + 0.2f * delta * delta);

    return reward;
}

void RlMpcTuner::update(float reward, const RlMpcState& next_state)
{
    // simple critic (value approximation)
    float V_current =
        critic_w_[0] * std::fabs(last_state_.lateral_error) +
        critic_w_[1] * std::fabs(last_state_.yaw_error);

    float V_next =
        critic_w_[0] * std::fabs(next_state.lateral_error) +
        critic_w_[1] * std::fabs(next_state.yaw_error);

    float td_error = reward + gamma_ * V_next - V_current;

    // update critic
    critic_w_[0] += lr_critic_ * td_error * std::fabs(last_state_.lateral_error);
    critic_w_[1] += lr_critic_ * td_error * std::fabs(last_state_.yaw_error);

    // update actor (very simplified)
    actor_w_[0] += lr_actor_ * td_error;
    actor_w_[1] += lr_actor_ * td_error;
}