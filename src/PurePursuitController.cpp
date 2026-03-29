#include "PurePursuitController.hpp"
#include <cmath>
#include <algorithm>

PurePursuitController::PurePursuitController()
    : wheelbase_m_(0.2515f),
      lookahead_m_(0.35f),
      pixel_per_meter_(250.0f),
      rear_axle_offset_px_(40.0f),
      max_steering_deg_(25.0f)
{
}

void PurePursuitController::setWheelbase(float wheelbase_m)
{
    wheelbase_m_ = wheelbase_m;
}

void PurePursuitController::setLookahead(float lookahead_m)
{
    lookahead_m_ = lookahead_m;
}

void PurePursuitController::setPixelPerMeter(float pixel_per_meter)
{
    if (pixel_per_meter > 1e-3f)
        pixel_per_meter_ = pixel_per_meter;
}

void PurePursuitController::setRearAxleOffsetPx(float offset_px)
{
    rear_axle_offset_px_ = offset_px;
}

void PurePursuitController::setMaxSteeringDeg(float max_deg)
{
    max_steering_deg_ = max_deg;
}

bool PurePursuitController::findTargetPoint(
    const std::vector<cv::Point>& path,
    const cv::Point2f& rear_axle,
    float lookahead_px,
    cv::Point2f& target
) const
{
    if (path.size() < 2)
        return false;

    for (size_t i = 0; i < path.size(); ++i)
    {
        cv::Point2f p(path[i].x, path[i].y);
        float d = cv::norm(p - rear_axle);

        if (d >= lookahead_px)
        {
            target = p;
            return true;
        }
    }

    target = cv::Point2f(path.back().x, path.back().y);
    return true;
}

float PurePursuitController::computeSteeringAngle(
    const std::vector<cv::Point>& path,
    const cv::Size& bev_size,
    float vehicle_speed_mps
)
{
    if (path.size() < 2 || pixel_per_meter_ < 1e-3f)
        return 0.0f;

    // Giả sử xe nằm giữa đáy ảnh BEV và hướng lên trên
    cv::Point2f rear_axle(
        bev_size.width * 0.5f,
        bev_size.height - rear_axle_offset_px_
    );

    // Có thể đổi sang lookahead động nếu muốn:
    // float ld_m = std::max(0.25f, 0.25f + 1.5f * vehicle_speed_mps);
    float ld_m = lookahead_m_;
    float ld_px = ld_m * pixel_per_meter_;

    cv::Point2f target;
    if (!findTargetPoint(path, rear_axle, ld_px, target))
        return 0.0f;

    // Vì ảnh OpenCV trục y hướng xuống, còn xe hướng lên trên,
    // dùng dx = target.x - rear.x
    // dy_forward = rear.y - target.y
    float dx = target.x - rear_axle.x;
    float dy = rear_axle.y - target.y;

    float alpha = std::atan2(dx, dy);

    float delta_rad = std::atan2(2.0f * wheelbase_m_ * std::sin(alpha), ld_m);
    float delta_deg = delta_rad * 180.0f / static_cast<float>(M_PI);

    delta_deg = std::clamp(delta_deg, -max_steering_deg_, max_steering_deg_);
    return delta_deg;
}