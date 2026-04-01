#pragma once
#include <vector>

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

    // infer weights from state
    RlMpcWeights infer(const RlMpcState& s);

    // compute reward
    float computeReward(const RlMpcState& s,
                        float steering,
                        float prev_steering) const;

    // update actor-critic
    void update(float reward, const RlMpcState& next_state);

private:
    // internal helpers
    float clamp(float v, float min_v, float max_v) const;

private:
    // actor parameters (simple linear model)
    std::vector<float> actor_w_;

    // critic parameters
    std::vector<float> critic_w_;

    // last state/action
    RlMpcState last_state_;
    RlMpcWeights last_action_;

    float gamma_;
    float lr_actor_;
    float lr_critic_;
};