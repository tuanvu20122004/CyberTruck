#include "LaneChangePlanner.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

LaneChangePlanner::LaneChangePlanner()
    : state_(PlannerState::KEEP_LANE),
      progress_(0.0f),
      trigger_distance_(0.3f),
      change_rate_(0.08f)
{
}

float LaneChangePlanner::smoothStep(float x)
{
    x = std::clamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
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
        int xi = std::clamp((int)std::round(x), 0, img_width - 1);
        line.emplace_back(xi, y);
    }

    return line;
}

std::vector<cv::Point> LaneChangePlanner::blendCenterlines(
    const std::vector<cv::Point>& from_line,
    const std::vector<cv::Point>& to_line,
    float alpha
)
{
    std::vector<cv::Point> out;
    size_t n = std::min(from_line.size(), to_line.size());
    if (n < 3) return from_line;

    out.reserve(n);
    float s = smoothStep(alpha);

    for (size_t i = 0; i < n; ++i)
    {
        int x = (int)std::round((1.0f - s) * from_line[i].x + s * to_line[i].x);
        int y = from_line[i].y;
        out.emplace_back(x, y);
    }

    return out;
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

    switch (state_)
    {
    case PlannerState::KEEP_LANE:
    {
        if (obstacle_distance > 0.0f && obstacle_distance < trigger_distance_)
        {
            if (!left_target.empty())
            {
                state_ = PlannerState::CHANGE_LEFT;
                progress_ = 0.0f;
                std::cout << "[PLANNER] CHANGE_LEFT start\n";
            }
            else if (!right_target.empty())
            {
                state_ = PlannerState::CHANGE_RIGHT;
                progress_ = 0.0f;
                std::cout << "[PLANNER] CHANGE_RIGHT start\n";
            }
        }
        return base_centerline;
    }

    case PlannerState::CHANGE_LEFT:
    {
        if (left_target.empty())
        {
            state_ = PlannerState::KEEP_LANE;
            progress_ = 0.0f;
            return base_centerline;
        }

        progress_ += change_rate_;
        if (progress_ >= 100.0f)
        {
            progress_ = 1.0f;
            state_ = PlannerState::FOLLOW_LEFT_LANE;
            std::cout << "[PLANNER] FOLLOW_LEFT_LANE\n";
        }

        return blendCenterlines(base_centerline, left_target, progress_);
    }

    case PlannerState::CHANGE_RIGHT:
    {
        if (right_target.empty())
        {
            state_ = PlannerState::KEEP_LANE;
            progress_ = 0.0f;
            return base_centerline;
        }

        progress_ += change_rate_;
        if (progress_ >= 100.0f)
        {
            progress_ = 1.0f;
            state_ = PlannerState::FOLLOW_RIGHT_LANE;
            std::cout << "[PLANNER] FOLLOW_RIGHT_LANE\n";
        }

        return blendCenterlines(base_centerline, right_target, progress_);
    }

    case PlannerState::FOLLOW_LEFT_LANE:
        //return left_target.empty() ? base_centerline : left_target;
        return base_centerline;
    case PlannerState::FOLLOW_RIGHT_LANE:
        //return right_target.empty() ? base_centerline : right_target;
        return base_centerline;
    default:
        return base_centerline;
    }
}
