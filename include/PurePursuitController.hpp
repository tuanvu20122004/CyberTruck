#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PurePursuitController
{
public:
    PurePursuitController();

    void setWheelbase(float wheelbase_m);           // chiều dài cơ sở
    void setLookahead(float lookahead_m);           // khoảng lookahead
    void setPixelPerMeter(float pixel_per_meter);   // chuyển đổi pixel <-> mét
    void setRearAxleOffsetPx(float offset_px);      // vị trí trục sau trong ảnh BEV
    void setMaxSteeringDeg(float max_deg);          // giới hạn góc lái

    // output góc lái
    float computeSteeringAngle(
        const std::vector<cv::Point>& path,         // vector các điểm 
        const cv::Size& bev_size,                   //kích thước ảnh bird-eye-view
        float vehicle_speed_mps                     // tốc độ
    );

private:

    // tìm target point theo lookahead distance
    // mục đích tính ld bằng cách duyệt từng điểm trong path
    // chọn điểm đầu tiên có khoảng cách >= ld
    bool findTargetPoint(
        const std::vector<cv::Point>& path,
        const cv::Point2f& rear_axle,
        float lookahead_px,
        cv::Point2f& target
    ) const;

private:
    float wheelbase_m_;             // L: wheelbase
    float lookahead_m_;             // ld: lookahead distance
    float pixel_per_meter_;         // scale: pixel -> meter
    float rear_axle_offset_px_;     // vị trí trục sau
    float max_steering_deg_;        // giới hạn góc lái
};