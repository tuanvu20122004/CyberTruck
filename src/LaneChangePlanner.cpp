#include "LaneChangePlanner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

LaneChangePlanner::LaneChangePlanner()
    : state_(PlannerState::KEEP_LANE),
      last_direction_(DONT_CHANGE),
      hold_counter_(0),
      hold_frames_(8),
      current_lane_(0),
      target_lane_(0),//
      settle_counter_(0),// giữ ổn định sau khi đổi lane xong trong 8 frame (~0.25s)
      settle_frames_(8),
      trigger_distance_(1.1f),
      lane_width_m_(0.40f),
      vehicle_width_m_(0.21f),
      vehicle_length_m_(0.432f),
      obstacle_width_m_(0.20f),
      obstacle_length_m_(0.22f),
      safe_margin_m_(0.08f),
      vx_mps_(0.08f),
      meter_per_pixel_(0.001f),
      last_min_distance_m_(std::numeric_limits<float>::infinity()),
      last_min_ttc_s_(std::numeric_limits<float>::infinity()),
      last_cost_(std::numeric_limits<float>::infinity()),
      last_img_width_(640),
      last_img_height_(480)
{
}

void LaneChangePlanner::setLaneWidthMeters(float lane_width_m)
{
    if (lane_width_m > 0.05f)
        lane_width_m_ = lane_width_m;
}

void LaneChangePlanner::setVehicleSize(float width_m, float length_m)
{
    if (width_m > 0.05f) vehicle_width_m_ = width_m;
    if (length_m > 0.05f) vehicle_length_m_ = length_m;
}

void LaneChangePlanner::setObstacleSize(float width_m, float length_m)
{
    if (width_m > 0.05f) obstacle_width_m_ = width_m;
    if (length_m > 0.05f) obstacle_length_m_ = length_m;
}

void LaneChangePlanner::setSafeMargin(float safe_margin_m)
{
    if (safe_margin_m >= 0.0f)
        safe_margin_m_ = safe_margin_m;
}

void LaneChangePlanner::setSpeed(float vx_mps)
{
    if (vx_mps > 0.02f)
        vx_mps_ = vx_mps;
}

void LaneChangePlanner::setTriggerDistance(float trigger_distance_m)
{
    if (trigger_distance_m > 0.05f)
        trigger_distance_ = trigger_distance_m;
}

std::vector<cv::Point> LaneChangePlanner::update(
    const std::vector<cv::Point>& base_centerline,
    const cv::Vec3f& /*left_coeff*/,
    const cv::Vec3f& /*right_coeff*/,
    bool has_left_lane,
    bool has_right_lane,
    float lane_width_px,
    LaneLineType left_type,
    LaneLineType right_type,
    float obstacle_distance,
    int img_width,
    int img_height
)
{
    last_img_width_ = img_width;
    last_img_height_ = img_height;

    if (base_centerline.size() < 3)
    {
        state_ = PlannerState::KEEP_LANE;
        last_direction_ = DONT_CHANGE;
        target_lane_ = current_lane_;
        last_min_distance_m_ = std::numeric_limits<float>::infinity();
        last_min_ttc_s_ = std::numeric_limits<float>::infinity();
        last_cost_ = std::numeric_limits<float>::infinity();
        return base_centerline;
    }

    updateScaleFromLaneWidth(lane_width_px);

    std::vector<ReferencePoint> ref = buildReferencePath(base_centerline);
    if (ref.size() < 3)
    {
        state_ = PlannerState::KEEP_LANE;
        last_direction_ = DONT_CHANGE;
        target_lane_ = current_lane_;
        return base_centerline;
    }

    if (hold_counter_ > 0)
        --hold_counter_;

    if (settle_counter_ > 0)
        --settle_counter_;

    const bool allow_left = canChangeLeft(obstacle_distance, has_left_lane, left_type);
    const bool allow_right = canChangeRight(obstacle_distance, has_right_lane, right_type);

    bool allow_left_committed = allow_left;
    bool allow_right_committed = allow_right;

    if (state_ == PlannerState::CHANGE_USING_DASHED)
    {
        if (target_lane_ < current_lane_)
            allow_right_committed = false;
        else if (target_lane_ > current_lane_)
            allow_left_committed = false;
    }

    // vừa settle xong thì không cho flip ngược ngay
    if (settle_counter_ > 0)
    {
        if (current_lane_ > 0)
            allow_left_committed = false;   // đang ở làn phải, tạm không cho quay trái ngay
        else if (current_lane_ < 0)
            allow_right_committed = false;  // đang ở làn trái, tạm không cho quay phải ngay
    }

    std::vector<Candidate> candidates =
        generateCandidates(ref, allow_left_committed, allow_right_committed);

    StaticObstacle obstacle = buildStaticObstacle(obstacle_distance);

    for (auto& c : candidates)
    {
        evaluateCandidate(c, obstacle);
    }

    Candidate best = selectBestCandidate(candidates);

    last_min_distance_m_ = best.min_distance_m;
    last_min_ttc_s_ = best.min_ttc_s;
    last_cost_ = best.cost;

    updatePlannerState(best);

    if (best.polyline_px.size() >= 3)
        return best.polyline_px;

    return base_centerline;
}

void LaneChangePlanner::updateScaleFromLaneWidth(float lane_width_px)
{
    if (lane_width_px > 50.0f && lane_width_m_ > 0.05f)
    {
        meter_per_pixel_ = lane_width_m_ / lane_width_px;
    }
}

std::vector<LaneChangePlanner::ReferencePoint>
LaneChangePlanner::buildReferencePath(const std::vector<cv::Point>& base_centerline) const
{
    std::vector<ReferencePoint> ref;
    if (base_centerline.size() < 3 || meter_per_pixel_ <= 0.0f)
        return ref;

    std::vector<cv::Point2f> ordered;
    ordered.reserve(base_centerline.size());
    for (auto it = base_centerline.rbegin(); it != base_centerline.rend(); ++it)
    {
        ordered.emplace_back(static_cast<float>(it->x), static_cast<float>(it->y));
    }

    ref.reserve(ordered.size());

    float cumulative_s = 0.0f;

    for (size_t i = 0; i < ordered.size(); ++i)
    {
        if (i > 0)
        {
            cumulative_s += static_cast<float>(cv::norm(ordered[i] - ordered[i - 1])) * meter_per_pixel_;
        }

        cv::Point2f tangent(0.0f, -1.0f);

        if (i + 1 < ordered.size())
        {
            tangent = ordered[i + 1] - ordered[i];
        }
        else if (i > 0)
        {
            tangent = ordered[i] - ordered[i - 1];
        }

        float norm_t = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
        if (norm_t < 1e-6f)
            tangent = cv::Point2f(0.0f, -1.0f);
        else
            tangent *= (1.0f / norm_t);

        cv::Point2f n_right(-tangent.y, tangent.x);

        ref.push_back({ordered[i], cumulative_s, n_right});
    }

    return ref;
}

LaneChangePlanner::StaticObstacle
LaneChangePlanner::buildStaticObstacle(float obstacle_distance_m) const
{
    StaticObstacle obs;
    obs.s_m = obstacle_distance_m;
    obs.d_m = 0.0f;
    obs.width_m = obstacle_width_m_;
    obs.length_m = obstacle_length_m_;
    obs.valid = (obstacle_distance_m > 0.02f && vx_mps_ > 0.02f);
    return obs;
}

bool LaneChangePlanner::canChangeLeft(float distance, bool has_left_lane, LaneLineType left_type) const
{
    return (distance <= trigger_distance_) && has_left_lane && left_type == LaneLineType::DASHED;
}

bool LaneChangePlanner::canChangeRight(float distance, bool has_right_lane, LaneLineType right_type) const
{
    return (distance <= trigger_distance_) && has_right_lane && right_type == LaneLineType::DASHED;
}

std::vector<LaneChangePlanner::Candidate>
LaneChangePlanner::generateCandidates(
    const std::vector<ReferencePoint>& ref,
    bool allow_left,
    bool allow_right
) const
{
    std::vector<Candidate> candidates;

    candidates.push_back(makeKeepLaneCandidate(ref));

    static const std::array<float, 4> T_SET = {0.80f, 1.00f, 1.20f, 1.50f};

    if (allow_left)
    {
        for (float T : T_SET)
            candidates.push_back(makeLaneChangeCandidate(ref, current_lane_ - 1, T));
    }

    if (allow_right)
    {
        for (float T : T_SET)
            candidates.push_back(makeLaneChangeCandidate(ref, current_lane_ + 1, T));
    }

    return candidates;
}

LaneChangePlanner::Candidate
LaneChangePlanner::makeKeepLaneCandidate(const std::vector<ReferencePoint>& ref) const
{
    Candidate c;
    c.target_lane = current_lane_;
    c.maneuver_time_s = 0.0f;
    c.lane_change_distance_m = 0.0f;
    c.target_offset_m = laneOffsetFromIndex(current_lane_);
    c.polyline_px = offsetReferenceToPolyline(ref, c);
    c.feasible = (c.polyline_px.size() >= 3);
    return c;
}

LaneChangePlanner::Candidate
LaneChangePlanner::makeLaneChangeCandidate(
    const std::vector<ReferencePoint>& ref,
    int target_lane,
    float maneuver_time_s
) const
{
    Candidate c;
    c.target_lane = target_lane;
    c.maneuver_time_s = maneuver_time_s;
    c.lane_change_distance_m = std::max(0.15f, vx_mps_ * maneuver_time_s);
    c.target_offset_m = laneOffsetFromIndex(target_lane);
    c.polyline_px = offsetReferenceToPolyline(ref, c);
    c.feasible = (c.polyline_px.size() >= 3);
    return c;
}

void LaneChangePlanner::evaluateCandidate(
    Candidate& c,
    const StaticObstacle& obstacle
) const
{
    if (!c.feasible)
        return;

    const float lat_clear =
        0.5f * (vehicle_width_m_ + obstacle.width_m) + safe_margin_m_;

    const float long_clear =
        0.5f * (vehicle_length_m_ + obstacle.length_m) + safe_margin_m_;

    const float dt = 0.05f;

    const float t_end =
        obstacle.valid
            ? std::max(c.maneuver_time_s + 0.6f,
                       obstacle.s_m / std::max(vx_mps_, 0.05f) + 0.8f)
            : (c.maneuver_time_s + 0.8f);

    c.min_distance_m = std::numeric_limits<float>::infinity();
    c.min_ttc_s = std::numeric_limits<float>::infinity();
    c.max_curvature = 0.0f;
    c.max_jerk = 0.0f;
    c.collision = false;

    for (float t = 0.0f; t <= t_end; t += dt)
    {
        const float s = vx_mps_ * t;
        const float d = lateralOffsetAtS(s, c);

        const float ds = lateralDsAtS(s, c);
        const float dss = lateralDssAtS(s, c);
        const float jerk = std::fabs(lateralD3dt3AtS(s, c));

        const float curvature =
            std::fabs(dss) / std::pow(1.0f + ds * ds, 1.5f);

        c.max_curvature = std::max(c.max_curvature, curvature);
        c.max_jerk = std::max(c.max_jerk, jerk);

        if (obstacle.valid)
        {
            const float ds_obs = obstacle.s_m - s;
            const float dd_obs = obstacle.d_m - d;

            const float center_dist = std::sqrt(ds_obs * ds_obs + dd_obs * dd_obs);
            c.min_distance_m = std::min(c.min_distance_m, center_dist);

            if (std::fabs(ds_obs) <= long_clear && std::fabs(dd_obs) <= lat_clear)
            {
                c.collision = true;
            }

            if (ds_obs > 0.0f && std::fabs(dd_obs) <= lat_clear)
            {
                const float ttc = ds_obs / std::max(vx_mps_, 0.05f);
                c.min_ttc_s = std::min(c.min_ttc_s, ttc);
            }
        }
    }

    if (!obstacle.valid)
    {
        c.min_distance_m = 999.0f;
        c.min_ttc_s = 999.0f;
    }
    else
    {
        if (!std::isfinite(c.min_distance_m))
            c.min_distance_m = 999.0f;

        if (!std::isfinite(c.min_ttc_s))
            c.min_ttc_s = 999.0f;
    }

    const float desired_dist = std::max(lat_clear + 0.05f, 0.18f);
    const float ttc_threshold = 1.5f;

    const float dist_cost =
        (c.min_distance_m >= desired_dist)
            ? 0.0f
            : sqr((desired_dist - c.min_distance_m) / desired_dist);

    const float ttc_cost =
        (c.min_ttc_s >= ttc_threshold)
            ? 0.0f
            : sqr((ttc_threshold - c.min_ttc_s) / ttc_threshold);

    const float curvature_cost = c.max_curvature;
    const float jerk_cost = c.max_jerk;
    const float time_cost = c.maneuver_time_s;

    float keep_lane_penalty = 0.0f;
    if (obstacle.valid && c.target_lane == current_lane_ && obstacle.s_m < trigger_distance_)
    {
        keep_lane_penalty += 8.0f * dist_cost;
        keep_lane_penalty += 6.0f * ttc_cost;
        if (c.collision)
            keep_lane_penalty += 500.0f;
    }

    float direction_bias = 0.0f;
    if (hold_counter_ > 0)
    {
        if (target_lane_ < current_lane_)
        {
            if (c.target_lane == target_lane_)      direction_bias = -7.0f;
            //else if (c.target_lane == current_lane_) direction_bias = +15.0f;
            else if (c.target_lane > current_lane_)  direction_bias = +20.0f;
        }
        else if (target_lane_ > current_lane_)
        {
            if (c.target_lane == target_lane_)      direction_bias = -7.0f;
            //else if (c.target_lane == current_lane_) direction_bias = +15.0f;
            else if (c.target_lane < current_lane_)  direction_bias = +20.0f;
        }
    }

    if (c.collision)
    {
        c.cost = 1.0e6f;
    }
    else
    {
        c.cost =
            12.0f * dist_cost +
            10.0f * ttc_cost +
            1.00f * curvature_cost +
            0.03f * jerk_cost +
            0.25f * time_cost +
            keep_lane_penalty +
            direction_bias;
    }
}

LaneChangePlanner::Candidate
LaneChangePlanner::selectBestCandidate(const std::vector<Candidate>& candidates) const
{
    Candidate best;
    const Candidate* latched = nullptr;

    for (const auto& c : candidates)
    {
        if (!c.feasible)
            continue;

        if (c.cost < best.cost)
            best = c;

        if (hold_counter_ > 0)
        {
            if (c.target_lane == target_lane_)
                latched = &c;
        }
    }

    if (latched != nullptr && latched->cost <= best.cost + 7.0f)
        return *latched;

    return best;
}

void LaneChangePlanner::updatePlannerState(const Candidate& best)
{
    // giữ làn hiện tại
    if (best.target_lane == current_lane_)
    {
        if (current_lane_ == 0)
            state_ = PlannerState::KEEP_LANE;
        else
            state_ = PlannerState::FOLLOW_LANE;

        // nếu vừa hoàn tất đổi làn trước đó thì bắt đầu settle
        if (target_lane_ != current_lane_)
        {
            settle_counter_ = settle_frames_;
        }

        target_lane_ = current_lane_;
        last_direction_ = DONT_CHANGE;
        return;
    }

    // bắt đầu hoặc tiếp tục đổi làn
    state_ = PlannerState::CHANGE_USING_DASHED;
    target_lane_ = best.target_lane;

    if (target_lane_ < current_lane_)
        last_direction_ = CHANGE_LEFT;
    else
        last_direction_ = CHANGE_RIGHT;

    // khi best target khác current lane, coi như planner đang bám mục tiêu mới
    // current_lane_ chỉ cập nhật khi maneuver hoàn tất
    const bool change_completed = (std::fabs(best.target_offset_m - laneOffsetFromIndex(current_lane_)) < 1e-3f);
    (void)change_completed;

    // heuristic: nếu best đã quay về keep trên target mới ở vòng sau thì current_lane_ sẽ được cập nhật ở block đầu
    if (state_ == PlannerState::CHANGE_USING_DASHED)
    {
        // cập nhật "làn đã xác nhận" khi mục tiêu thay đổi và quỹ đạo đã đủ commit
        // ở đây dùng cách đơn giản: chốt current_lane_ ngay khi đã chọn target ổn định
        current_lane_ = target_lane_;
    }

    hold_counter_ = hold_frames_;
}

std::vector<cv::Point> LaneChangePlanner::offsetReferenceToPolyline(
    const std::vector<ReferencePoint>& ref,
    const Candidate& c
) const
{
    std::vector<cv::Point> polyline;
    polyline.reserve(ref.size());

    for (const auto& rp : ref)
    {
        const float d_m = lateralOffsetAtS(rp.s_m, c);
        const float d_px = d_m / std::max(meter_per_pixel_, 1e-6f);

        float x = rp.pos_px.x + d_px * rp.n_right.x;
        float y = rp.pos_px.y + d_px * rp.n_right.y;

        int xi = static_cast<int>(std::lround(x));
        int yi = static_cast<int>(std::lround(y));

        xi = std::clamp(xi, 0, std::max(0, last_img_width_ - 1));
        yi = std::clamp(yi, 0, std::max(0, last_img_height_ - 1));

        polyline.emplace_back(xi, yi);
    }

    std::reverse(polyline.begin(), polyline.end());
    return polyline;
}

float LaneChangePlanner::lateralOffsetAtS(float s_m, const Candidate& c) const
{
    const float start_offset = laneOffsetFromIndex(current_lane_);
    const float end_offset = c.target_offset_m;

    if (c.lane_change_distance_m <= 1e-6f)
        return end_offset;

    if (s_m <= 0.0f)
        return start_offset;

    if (s_m >= c.lane_change_distance_m)
        return end_offset;

    const float sigma = s_m / c.lane_change_distance_m;
    return start_offset + (end_offset - start_offset) * quinticBlend(sigma);
}

float LaneChangePlanner::lateralDsAtS(float s_m, const Candidate& c) const
{
    if (c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    const float sigma = s_m / c.lane_change_distance_m;
    const float start_offset = laneOffsetFromIndex(current_lane_);
    const float delta_offset = c.target_offset_m - start_offset;

    return delta_offset * quinticBlendD1(sigma) / c.lane_change_distance_m;
}

float LaneChangePlanner::lateralDssAtS(float s_m, const Candidate& c) const
{
    if (c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    const float sigma = s_m / c.lane_change_distance_m;
    const float start_offset = laneOffsetFromIndex(current_lane_);
    const float delta_offset = c.target_offset_m - start_offset;
    const float denom = c.lane_change_distance_m * c.lane_change_distance_m;

    return delta_offset * quinticBlendD2(sigma) / denom;
}

float LaneChangePlanner::lateralD3dt3AtS(float s_m, const Candidate& c) const
{
    if (c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    const float sigma = s_m / c.lane_change_distance_m;
    const float start_offset = laneOffsetFromIndex(current_lane_);
    const float delta_offset = c.target_offset_m - start_offset;
    const float scale = std::pow(vx_mps_ / c.lane_change_distance_m, 3.0f);

    return delta_offset * quinticBlendD3(sigma) * scale;
}

float LaneChangePlanner::laneOffsetFromIndex(int lane_idx) const
{
    return static_cast<float>(lane_idx) * lane_width_m_;
}

float LaneChangePlanner::quinticBlend(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 10.0f * sigma * sigma * sigma
         - 15.0f * sigma * sigma * sigma * sigma
         + 6.0f * sigma * sigma * sigma * sigma * sigma;
}

float LaneChangePlanner::quinticBlendD1(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 30.0f * sigma * sigma
         - 60.0f * sigma * sigma * sigma
         + 30.0f * sigma * sigma * sigma * sigma;
}

float LaneChangePlanner::quinticBlendD2(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 60.0f * sigma
         - 180.0f * sigma * sigma
         + 120.0f * sigma * sigma * sigma;
}

float LaneChangePlanner::quinticBlendD3(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 60.0f - 360.0f * sigma + 360.0f * sigma * sigma;
}