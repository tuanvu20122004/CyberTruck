#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PurePursuitController
{
public:
    PurePursuitController();

    void setWheelbase(float wheelbase_m);
    void setLookahead(float lookahead_m);
    void setPixelPerMeter(float pixel_per_meter);
    void setRearAxleOffsetPx(float offset_px);
    void setMaxSteeringDeg(float max_deg);

    float computeSteeringAngle(
        const std::vector<cv::Point>& path,
        const cv::Size& bev_size,
        float vehicle_speed_mps
    );

private:
    bool findTargetPoint(
        const std::vector<cv::Point>& path,
        const cv::Point2f& rear_axle,
        float lookahead_px,
        cv::Point2f& target
    ) const;

private:
    float wheelbase_m_;
    float lookahead_m_;
    float pixel_per_meter_;
    float rear_axle_offset_px_;
    float max_steering_deg_;
};