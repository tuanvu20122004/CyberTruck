#pragma once
#include <vector>

struct RlMpcState
{
    float lateral_error;   // độ lệch ngang so với centerline [m]
    float yaw_error;       // sai số góc heading [rad]
    float velocity;        // vận tốc xe [m/s]
    float curvature;       // độ cong quỹ đạo [1/m]
    float prev_steering;   // góc lái trước đó [deg]
};

struct RlMpcWeights
{
    float Q1;   // trọng số phạt lệch ngang
    float Q2;   // phạt lỗi yaw
    float R;    // phạt effort/ góc lái, độ gắt của bộ điều khiển
};

class RlMpcTuner
{
public:
    RlMpcTuner();

    //dùng trang thái hiện tại để suy ra bộ trọng số mới cho MPC
    RlMpcWeights infer(const RlMpcState& s);

    // Compute reward.
    float computeReward(const RlMpcState& s,
                        float steering_deg,
                        float prev_steering_deg) const;

    // Cập nhật actor - critic thông qua reward và state kế tiếp
    void update(float reward, const RlMpcState& next_state);

private:
    float clamp(float v, float min_v, float max_v) const;
    float deg2rad(float deg) const;

    //  tham số của actor, phần sinh ra weight
    std::vector<float> actor_w_;    

    // tham số của critic, phần ước lượng value
    std::vector<float> critic_w_;

    // lưu state và action trước đó
    RlMpcState last_state_{};
    RlMpcWeights last_action_{};

    float gamma_;       // discount factor
    float lr_actor_;    //learning rate
    float lr_critic_;   // learning rate
};
