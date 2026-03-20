#include "LaneChangeDecision.hpp"

#include <algorithm>
#include <cmath>

LaneChangeDecision::LaneChangeDecision(const Params& params)
    : params_(params),
      state_(DecisionState::KEEP_LANE),
      last_preferred_direction_(DecisionDirection::NONE),
      committed_direction_(DecisionDirection::NONE),
      last_obstacle_distance_(-1.0f),
      has_last_distance_(false),
      persistence_count_(0),
      cooldown_count_(0)
{
}

void LaneChangeDecision::reset()
{
    state_ = DecisionState::KEEP_LANE;
    last_preferred_direction_ = DecisionDirection::NONE;
    committed_direction_ = DecisionDirection::NONE;
    last_obstacle_distance_ = -1.0f;
    has_last_distance_ = false;
    persistence_count_ = 0;
    cooldown_count_ = 0;
}

void LaneChangeDecision::notifyLaneChangeStarted()
{
    committed_direction_ = last_preferred_direction_;
    state_ = DecisionState::APPROVE_CHANGE;
    persistence_count_ = 0;
}

void LaneChangeDecision::notifyLaneChangeFinished()
{
    state_ = DecisionState::COOLDOWN;
    cooldown_count_ = params_.cooldown_frames;
    committed_direction_ = DecisionDirection::NONE;
}

float LaneChangeDecision::meanX(const std::vector<cv::Point>& line) const
{
    if (line.empty()) return 0.0f;

    double sum = 0.0;
    for (const auto& p : line)
        sum += p.x;

    return static_cast<float>(sum / static_cast<double>(line.size()));
}

std::vector<cv::Point> LaneChangeDecision::buildCenterlineFromBoundary(
    const cv::Vec3f& coeff,
    float offset_px,
    int img_width,
    int img_height
) const
{
    std::vector<cv::Point> line;
    line.reserve(img_height / 10 + 1);

    for (int y = 0; y < img_height; y += 10)
    {
        float x = coeff[0] * y * y + coeff[1] * y + coeff[2] + offset_px;
        int xi = std::clamp(static_cast<int>(std::lround(x)), 0, img_width - 1);
        line.emplace_back(xi, y);
    }

    return line;
}

float LaneChangeDecision::estimateClosingRate(float distance, float dt)
{
    if (distance <= 0.0f || dt <= 1e-4f)
    {
        has_last_distance_ = false;
        last_obstacle_distance_ = -1.0f;
        return 0.0f;
    }

    float closing_rate = 0.0f;

    if (has_last_distance_)
    {
        // distance giảm => closing_rate dương
        closing_rate = (last_obstacle_distance_ - distance) / dt;
    }

    last_obstacle_distance_ = distance;
    has_last_distance_ = true;

    // chỉ quan tâm xu hướng đang tiến gần
    return std::max(0.0f, closing_rate);
}

float LaneChangeDecision::computeTtcProxy(float distance, float closing_rate, float ego_speed) const
{
    if (distance <= 0.0f)
        return std::numeric_limits<float>::infinity();

    // Nếu chưa có closing rate rõ ràng, dùng ego speed làm proxy thận trọng
    float relative_speed = closing_rate;

    if (relative_speed < 0.05f)
        relative_speed = std::max(0.10f, 0.35f * ego_speed);

    return distance / relative_speed;
}

float LaneChangeDecision::computeUrgency(float distance, float ttc_proxy) const
{
    if (distance <= 0.0f)
        return 0.0f;

    float distance_term = 0.0f;
    if (distance < params_.caution_distance)
    {
        float x = (params_.caution_distance - distance) /
                  std::max(0.001f, params_.caution_distance - params_.critical_distance);
        distance_term = std::clamp(x, 0.0f, 1.0f);
    }

    float ttc_term = 0.0f;
    if (ttc_proxy < params_.ttc_threshold)
    {
        float x = (params_.ttc_threshold - ttc_proxy) / params_.ttc_threshold;
        ttc_term = std::clamp(x, 0.0f, 1.0f);
    }

    float urgency = 0.6f * distance_term + 0.4f * ttc_term;
    return std::clamp(urgency, 0.0f, 1.0f);
}

float LaneChangeDecision::scoreCandidate(
    const std::vector<cv::Point>& candidate,
    const std::vector<cv::Point>& base_centerline,
    bool lane_visible,
    LaneLineType lane_type,
    float urgency,
    DecisionDirection dir
) const
{
    if (!lane_visible) return -1.0f;
    if (lane_type != LaneLineType::DASHED) return -1.0f;
    if (candidate.size() < 3 || base_centerline.size() < 3) return -1.0f;

    const float base_x = meanX(base_centerline);
    const float cand_x = meanX(candidate);
    const float lateral_shift = std::fabs(cand_x - base_x);

    if (lateral_shift < 20.0f)
        return -1.0f;

    float score = 0.0f;

    // 1) lane hợp lệ
    score += 1.0f;

    // 2) độ lệch sang làn mới đủ rõ
    score += std::clamp(lateral_shift / 220.0f, 0.0f, 1.0f);

    // 3) khi urgency cao thì ưu tiên đổi làn mạnh hơn
    score += 0.8f * urgency;

    // 4) hysteresis: ưu tiên giữ hướng đã thiên về trước đó
    if (dir == last_preferred_direction_)
        score += params_.hysteresis_bonus;

    // 5) nếu đang cooldown thì cấm
    if (state_ == DecisionState::COOLDOWN)
        score = -1.0f;

    return score;
}

LaneChangeDecision::Output LaneChangeDecision::update(const Input& in)
{
    Output out;
    out.state = state_;

    if (cooldown_count_ > 0)
    {
        --cooldown_count_;
        state_ = DecisionState::COOLDOWN;
    }
    else if (state_ == DecisionState::COOLDOWN)
    {
        state_ = DecisionState::KEEP_LANE;
    }

    if (in.base_centerline.size() < 3)
    {
        out.state = state_;
        return out;
    }

    float lane_width = in.lane_width_px;
    if (lane_width < 250.0f || lane_width > 550.0f)
        lane_width = 400.0f;

    const float closing_rate = estimateClosingRate(in.obstacle_distance, in.dt);
    const float ttc_proxy = computeTtcProxy(in.obstacle_distance, closing_rate, in.ego_speed);
    const float urgency = computeUrgency(in.obstacle_distance, ttc_proxy);

    out.ttc_proxy = ttc_proxy;
    out.urgency = urgency;

    const bool front_blocked =
        (in.obstacle_distance > 0.0f) &&
        (in.obstacle_distance < params_.caution_distance || ttc_proxy < params_.ttc_threshold);

    std::vector<cv::Point> left_candidate;
    std::vector<cv::Point> right_candidate;

    if (in.has_left_lane && in.left_type == LaneLineType::DASHED)
    {
        left_candidate = buildCenterlineFromBoundary(
            in.left_coeff,
            -0.5f * lane_width,
            in.img_width,
            in.img_height
        );
    }

    if (in.has_right_lane && in.right_type == LaneLineType::DASHED)
    {
        right_candidate = buildCenterlineFromBoundary(
            in.right_coeff,
            -0.5f * lane_width,
            in.img_width,
            in.img_height
        );
    }

    const float left_score = scoreCandidate(
        left_candidate,
        in.base_centerline,
        in.has_left_lane,
        in.left_type,
        urgency,
        DecisionDirection::LEFT
    );

    const float right_score = scoreCandidate(
        right_candidate,
        in.base_centerline,
        in.has_right_lane,
        in.right_type,
        urgency,
        DecisionDirection::RIGHT
    );

    out.left_score = left_score;
    out.right_score = right_score;

    DecisionDirection preferred = DecisionDirection::NONE;
    float best_score = -1.0f;

    if (left_score > best_score)
    {
        best_score = left_score;
        preferred = DecisionDirection::LEFT;
    }

    if (right_score > best_score)
    {
        best_score = right_score;
        preferred = DecisionDirection::RIGHT;
    }

    // Nếu hai phía gần ngang nhau, giữ hướng cũ để tránh rung
    if (left_score > 0.0f && right_score > 0.0f &&
        std::fabs(left_score - right_score) < params_.keep_direction_bias &&
        last_preferred_direction_ != DecisionDirection::NONE)
    {
        preferred = last_preferred_direction_;
    }

    if (!front_blocked || preferred == DecisionDirection::NONE || state_ == DecisionState::COOLDOWN)
    {
        persistence_count_ = 0;
        last_preferred_direction_ = preferred;
        state_ = (state_ == DecisionState::COOLDOWN) ? DecisionState::COOLDOWN
                                                     : DecisionState::KEEP_LANE;

        out.state = state_;
        out.direction = DecisionDirection::NONE;
        out.persistence_count = persistence_count_;
        return out;
    }

    last_preferred_direction_ = preferred;
    state_ = DecisionState::PREPARE_CHANGE;
    ++persistence_count_;

    if (persistence_count_ >= params_.persistence_frames)
    {
        out.approve_lane_change = true;
        out.direction = preferred;
        out.state = DecisionState::APPROVE_CHANGE;
        out.persistence_count = persistence_count_;
        return out;
    }

    out.state = state_;
    out.direction = preferred;
    out.persistence_count = persistence_count_;
    return out;
}