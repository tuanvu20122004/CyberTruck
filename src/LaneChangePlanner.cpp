#include "LaneChangePlanner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

LaneChangePlanner::LaneChangePlanner()
    : state_(PlannerState::KEEP_LANE),
      progress_(0.0f),
      trigger_distance_(1.1f),
      min_progress_step_(0.025f),
      max_progress_step_(0.12f),
      lane_change_requested_(false),
      lane_change_finished_(false),
      requested_direction_(DONT_CHANGE)
{
    State_change_line.type_change = DONT_CHANGE;
    State_change_line.first_access = 0;
}

void LaneChangePlanner::requestLaneChange(Type_Change_t dir)
{
    if (dir != CHANGE_LEFT && dir != CHANGE_RIGHT)
        return;

    if (state_ == PlannerState::CHANGE_USING_DASHED)
        return;

    lane_change_requested_ = true;
    requested_direction_ = dir;
    lane_change_finished_ = false;
}

bool LaneChangePlanner::isLaneChangeActive() const
{
    return state_ == PlannerState::CHANGE_USING_DASHED;
}

bool LaneChangePlanner::isLaneChangeFinished() const
{
    return lane_change_finished_;
}

void LaneChangePlanner::clearFinishedFlag()
{
    lane_change_finished_ = false;
}

float LaneChangePlanner::smoothStep(float x)
{
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

float LaneChangePlanner::meanX(const std::vector<cv::Point>& line)
{
    if (line.empty()) return 0.0f;

    double sum = 0.0;
    for (const auto& p : line)
        sum += p.x;

    return static_cast<float>(sum / static_cast<double>(line.size()));
}

std::vector<cv::Point> LaneChangePlanner::buildCenterlineFromBoundary(
    const cv::Vec3f& coeff,
    float offset_px,
    int img_width,
    int img_height
)
{
    std::vector<cv::Point> line;
    line.reserve(img_height / 10 + 1);

    for (int y = 0; y < img_height; y += 10)
    {
        float x = coeff[0] * y * y + coeff[1] * y + coeff[2] + offset_px;
        int xi = std::clamp(static_cast<int>(std::round(x)), 0, img_width - 1);
        line.emplace_back(xi, y);
    }

    return line;
}

float LaneChangePlanner::aggressiveBlend(float alpha)
{
    alpha = std::clamp(alpha, 0.0f, 1.0f);

    float s = smoothStep(alpha);

    if (alpha > 0.45f)
    {
        float t = (alpha - 0.45f) / 0.55f;
        t = std::clamp(t, 0.0f, 1.0f);
        s += 0.22f * t;
    }

    return std::clamp(s, 0.0f, 1.0f);
}

std::vector<cv::Point> LaneChangePlanner::blendCenterlines(
    const std::vector<cv::Point>& from_line,
    const std::vector<cv::Point>& to_line,
    float alpha
)
{
    if (from_line.size() < 3) return from_line;
    if (to_line.size() < 3)   return from_line;

    const size_t n = std::min(from_line.size(), to_line.size());
    std::vector<cv::Point> out;
    out.reserve(n);

    const float s = aggressiveBlend(alpha);

    for (size_t i = 0; i < n; ++i)
    {
        int x = static_cast<int>(
            std::round((1.0f - s) * from_line[i].x + s * to_line[i].x)
        );
        int y = from_line[i].y;
        out.emplace_back(x, y);
    }

    return out;
}

float LaneChangePlanner::computeProgressStep(float obstacle_distance) const
{
    float distance_factor = 0.0f;

    if (obstacle_distance > 0.0f)
    {
        float d = std::clamp(obstacle_distance, 0.0f, trigger_distance_);
        float urgency = (trigger_distance_ - d) / trigger_distance_;
        urgency = std::clamp(urgency, 0.0f, 1.0f);

        distance_factor = std::sqrt(urgency);
    }

    float base_step =
        min_progress_step_ +
        distance_factor * (max_progress_step_ - min_progress_step_);

    float phase = smoothStep(progress_);
    float phase_factor = 1.20f - 0.10f * phase;

    float step = base_step * phase_factor;

    return std::clamp(step, min_progress_step_, max_progress_step_);
}

std::vector<cv::Point> LaneChangePlanner::update(
    const std::vector<cv::Point>& base_centerline,
    const cv::Vec3f& left_coeff,
    const cv::Vec3f& right_coeff,
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
    if (base_centerline.size() < 3)
        return base_centerline;

    float lane_width = lane_width_px;
    if (lane_width < 300.0f || lane_width > 500.0f)
        lane_width = 400.0f;

    std::vector<cv::Point> left_target;
    std::vector<cv::Point> right_target;

    // Lưu ý:
    // offset dấu cộng/trừ cần khớp với hệ tọa độ thực tế của bird-eye view.
    // Bản này dùng quy ước:
    // - left_coeff  -> dịch sang trái để ra centerline làn bên trái
    // - right_coeff -> dịch sang phải để ra centerline làn bên phải
    if (has_left_lane && left_type == LaneLineType::DASHED)
    {
        left_target = buildCenterlineFromBoundary(
            left_coeff,
            -0.5f * lane_width,
            img_width,
            img_height
        );
    }

    if (has_right_lane && right_type == LaneLineType::DASHED)
    {
        right_target = buildCenterlineFromBoundary(
            right_coeff,
            +0.5f * lane_width,
            img_width,
            img_height
        );
    }

    const bool both_lanes_visible = has_left_lane && has_right_lane;

    switch (state_)
    {
    case PlannerState::KEEP_LANE:
    {
        // Planner không tự quyết định nữa.
        // Chỉ bắt đầu khi Decision đã gửi request.
        if (lane_change_requested_)
        {
            lane_change_requested_ = false;
            lane_change_finished_ = false;
            progress_ = 0.0f;

            State_change_line.type_change = requested_direction_;
            State_change_line.first_access = 1;

            if (requested_direction_ == CHANGE_LEFT)
            {
                if (left_target.size() < 3)
                {
                    State_change_line.type_change = DONT_CHANGE;
                    State_change_line.first_access = 0;
                    requested_direction_ = DONT_CHANGE;
                    return base_centerline;
                }

                last_target_line_ = left_target;
            }
            else if (requested_direction_ == CHANGE_RIGHT)
            {
                if (right_target.size() < 3)
                {
                    State_change_line.type_change = DONT_CHANGE;
                    State_change_line.first_access = 0;
                    requested_direction_ = DONT_CHANGE;
                    return base_centerline;
                }

                last_target_line_ = right_target;
            }
            else
            {
                State_change_line.type_change = DONT_CHANGE;
                State_change_line.first_access = 0;
                return base_centerline;
            }

            state_ = PlannerState::CHANGE_USING_DASHED;

            std::cout << "[PLANNER] CHANGE_USING_DASHED start by request, dir="
                      << State_change_line.type_change << "\n";
        }

        return base_centerline;
    }

    case PlannerState::CHANGE_USING_DASHED:
    {
        std::vector<cv::Point> dynamic_target;

        if (State_change_line.type_change == CHANGE_LEFT)
        {
            dynamic_target = left_target;
        }
        else if (State_change_line.type_change == CHANGE_RIGHT)
        {
            dynamic_target = right_target;
        }

        // Nếu perception còn thấy target lane thì cập nhật target mới.
        // Nếu mất target tạm thời thì giữ target cũ.
        if (dynamic_target.size() >= 3)
        {
            last_target_line_ = dynamic_target;
        }
        else if (last_target_line_.size() >= 3)
        {
            dynamic_target = last_target_line_;
        }
        else
        {
            return base_centerline;
        }

        float step = computeProgressStep(obstacle_distance);
        progress_ += step;
        progress_ = std::clamp(progress_, 0.0f, 1.0f);

        std::cout << "[PLANNER] dir=" << State_change_line.type_change
                  << " distance=" << obstacle_distance
                  << " step=" << step
                  << " progress=" << progress_
                  << "\n";

        // Điều kiện hoàn tất:
        // - progress gần hoàn tất
        // - và perception đã ổn định trở lại
        if ((progress_ >= 0.95f) ||
            (progress_ > 0.80f && both_lanes_visible))
        {
            state_ = PlannerState::FOLLOW_LANE;
            progress_ = 0.0f;
            last_target_line_.clear();

            lane_change_finished_ = true;
            requested_direction_ = DONT_CHANGE;
            State_change_line.type_change = DONT_CHANGE;
            State_change_line.first_access = 0;

            std::cout << "[PLANNER] FOLLOW_LANE\n";
            return base_centerline;
        }

        return blendCenterlines(base_centerline, dynamic_target, progress_);
    }

    case PlannerState::FOLLOW_LANE:
    {
        // FOLLOW_LANE chỉ là trạng thái ổn định sau khi vừa đổi làn xong.
        // Không tự restart decision ở đây nữa.
        state_ = PlannerState::KEEP_LANE;
        return base_centerline;
    }

    default:
        return base_centerline;
    }
}