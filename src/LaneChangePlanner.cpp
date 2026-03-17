#include "LaneChangePlanner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

LaneChangePlanner::LaneChangePlanner()
    : state_(PlannerState::KEEP_LANE),
      progress_(0.0f),
      trigger_distance_(1.2f),
      min_progress_step_(0.025f),
      max_progress_step_(0.12f)
{
    State_change_line.type_change = DONT_CHANGE;
    State_change_line.first_access = 0;
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

std::vector<cv::Point> LaneChangePlanner::blendCenterlines(
    const std::vector<cv::Point>& from_line,
    const std::vector<cv::Point>& to_line,
    float alpha
)
{
    if (from_line.size() < 3) return from_line;
    if (to_line.size() < 3)   return from_line;

    size_t n = std::min(from_line.size(), to_line.size());
    std::vector<cv::Point> out;
    out.reserve(n);

    float s = smoothStep(alpha);

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

std::vector<cv::Point> LaneChangePlanner::chooseDashedTarget(
    const std::vector<cv::Point>& left_target,
    const std::vector<cv::Point>& right_target,
    const std::vector<cv::Point>& base_centerline
) const
{
    const bool has_left_target  = left_target.size()  >= 3;
    const bool has_right_target = right_target.size() >= 3;

    if (has_left_target && !has_right_target)
        return left_target;

    if (!has_left_target && has_right_target)
        return right_target;
    

    if (!has_left_target && !has_right_target)
        return {};


    // Nếu cả 2 đều là dashed, chọn target gần với target trước đó hơn
    // để tránh nhảy trái/phải liên tục.
    if (!last_target_line_.empty())
    {
        float last_x  = meanX(last_target_line_);
        float left_x  = meanX(left_target);
        float right_x = meanX(right_target);

        float d_left  = std::fabs(left_x - last_x);
        float d_right = std::fabs(right_x - last_x);

        return (d_left <= d_right) ? left_target : right_target;
    }

    // Chưa có target trước đó:
    // chọn target lệch xa hơn so với base_centerline để tạo lane-change rõ ràng.
    float base_x  = meanX(base_centerline);
    float left_dx = std::fabs(meanX(left_target) - base_x);
    float right_dx = std::fabs(meanX(right_target) - base_x);

    return (left_dx >= right_dx) ? left_target : right_target;
}

float LaneChangePlanner::computeProgressStep(float obstacle_distance) const
{
    // 1) Thành phần theo distance
    float distance_factor = 0.0f;

    if (obstacle_distance > 0.0f)
    {
        float d = std::clamp(obstacle_distance, 0.0f, trigger_distance_);
        float urgency = (trigger_distance_ - d) / trigger_distance_;
        urgency = std::clamp(urgency, 0.0f, 1.0f);

        // Làm phản ứng đầu pha nhạy hơn một chút
        distance_factor = std::sqrt(urgency);
    }

    float base_step =
        min_progress_step_ +
        distance_factor * (max_progress_step_ - min_progress_step_);

    // 2) Thành phần theo phase của lane-change
    // progress_=0  -> nhanh hơn
    // progress_=1  -> chậm hơn
    float phase = smoothStep(progress_);
    float phase_factor = 1.35f - 0.55f * phase;

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

    // Target centerline tạo từ lane nét đứt hiện tại
    std::vector<cv::Point> left_target;
    std::vector<cv::Point> right_target;

    // Nếu lane bên trái là nét đứt, target centerline sẽ nằm về phía trái của biên trái
    if (has_left_lane && left_type == LaneLineType::DASHED)
    {
        if (State_change_line.type_change != CHANGE_RIGHT)
        {
            left_target = buildCenterlineFromBoundary(
                left_coeff,
                -0.5f * lane_width,
                img_width,
                img_height
            );            
        }
        else
        {
            left_target = buildCenterlineFromBoundary(
                left_coeff,
                +0.5f * lane_width,
                img_width,
                img_height
            );  
        }


    }

    // Nếu lane bên phải là nét đứt, target centerline sẽ nằm về phía phải của biên phải
    if (has_right_lane && right_type == LaneLineType::DASHED)
    {
        if (State_change_line.type_change != CHANGE_LEFT)
        {
            right_target = buildCenterlineFromBoundary(
                right_coeff,
                +0.5f * lane_width,
                img_width,
                img_height
            );         
        }

        else
        {
            right_target = buildCenterlineFromBoundary(
                right_coeff,
                -0.5f * lane_width,
                img_width,
                img_height
            );         
        }
    }

    const bool both_lanes_visible = has_left_lane && has_right_lane;
    const bool any_dashed_visible = !left_target.empty() || !right_target.empty();

    switch (state_)
    {
    case PlannerState::KEEP_LANE:
    {
        // Bình thường bám lane hiện tại
        // Chỉ bắt đầu lane-change khi obstacle đủ gần và có ít nhất 1 lane nét đứt
        if (obstacle_distance > 0.0f &&
            obstacle_distance < trigger_distance_ &&
            any_dashed_visible)
        {
            state_ = PlannerState::CHANGE_USING_DASHED;
            progress_ = 0.0f;
            last_target_line_ = chooseDashedTarget(left_target, right_target, base_centerline);
           
            if (State_change_line.first_access == 0)
            {
                State_change_line.first_access == 1;

                if (!left_target.empty() && right_target.empty())
                {
                    State_change_line.type_change = CHANGE_LEFT;
                }

                else if (left_target.empty() && !right_target.empty())
                {
                    State_change_line.type_change = CHANGE_RIGHT;
                }

                else if (!left_target.empty() && !right_target.empty())
                {
                    float dx_left  = std::fabs(meanX(last_target_line_) - meanX(left_target));
                    float dx_right = std::fabs(meanX(last_target_line_) - meanX(right_target));

                    State_change_line.type_change = (dx_left < dx_right) ? CHANGE_LEFT : CHANGE_RIGHT;
                }

                else
                {
                    State_change_line.type_change = DONT_CHANGE;
                }
            }
            std::cout << "[PLANNER] CHANGE_USING_DASHED start, distance="
                      << obstacle_distance << "\n";
        }
        std::cout << "[PLANNER] KEEP_LANE\n";
        return base_centerline;
    }

    case PlannerState::CHANGE_USING_DASHED:
    {
        // Khi đã nhìn thấy đủ 2 lane thì kết thúc chuyển làn
        // và quay về bám lane chuẩn.
        if ((obstacle_distance < 0.0f ||
            obstacle_distance > trigger_distance_) &&
            both_lanes_visible &&
            progress_ > 0.6f)
        {
            state_ = PlannerState::FOLLOW_LANE;
            progress_ = 0.0f;
            last_target_line_.clear();
            State_change_line.type_change = DONT_CHANGE;
            State_change_line.first_access = 0;
            std::cout << "[PLANNER] FOLLOW_LANE\n";
            return base_centerline;
        }

        // Trong lúc chuyển làn:
        // dashed có thể đổi từ trái sang phải hoặc ngược lại,
        // nên mỗi frame chọn lại target theo dashed hiện tại.

        std::vector<cv::Point> dynamic_target =
            chooseDashedTarget(left_target, right_target, base_centerline);

        // Nếu mất dashed tạm thời thì giữ target cũ để xe không bị hụt lái
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
            // Không có gì để bám thì cứ giữ base_centerline
            return base_centerline;
        }

        float step = computeProgressStep(obstacle_distance);
        progress_ += step;
        progress_ = std::clamp(progress_, 0.0f, 1.0f);

        std::cout << "[PLANNER] distance=" << obstacle_distance
                  << " step=" << step
                  << " progress=" << progress_
                  << "\n";

        return blendCenterlines(base_centerline, dynamic_target, progress_);
    }

    case PlannerState::FOLLOW_LANE:
    {
        // Khi đã thấy 2 lane -> bám lane bình thường bằng base_centerline
        // Nếu sau đó obstacle lại xuất hiện gần và có dashed,
        // cho phép bắt đầu một lần lane-change mới.
        if (obstacle_distance > 0.8f &&
            obstacle_distance < trigger_distance_ &&
            any_dashed_visible)
        {
            state_ = PlannerState::CHANGE_USING_DASHED;
            progress_ = 0.0f;
            last_target_line_ = chooseDashedTarget(left_target, right_target, base_centerline);

            std::cout << "[PLANNER] CHANGE_USING_DASHED restart\n";
            return blendCenterlines(base_centerline, last_target_line_, progress_);
        }

        return base_centerline;
    }

    default:
        return base_centerline;
    }
}