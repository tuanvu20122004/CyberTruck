#ifndef LOGIC_HPP
#define LOGIC_HPP

#include "LaneDetector.hpp"
#include "MpcController.hpp"
#include "PurePursuitController.hpp"
#include "communication.hpp"
#include "Trans_UDP.hpp"
#include "LaneChangePlanner.hpp"
#include "PurePursuitController.hpp"

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

enum class ControlMode
{
    MPC,
    PURE_PURSUIT
};

class Logic
{
public:
    explicit Logic(const std::string& videoPath);
    void run();
    void setControlMode(ControlMode mode);
private:
    float computeSteering(
        const std::vector<cv::Point>& base_centerline,
        const std::vector<cv::Point>& target_centerline,
        const cv::Mat& birdEyeView,
        float distance,
        float& lateral_error_out,
        float& yaw_out
    );

private:
    LaneDetector            detector;
    MpcController           mpc;
    PurePursuitController   pure_pursuit;
    Communication           comm;
    Trans_UDP               udp_yolo;
    Trans_UDP               udp_debug;
    LaneChangePlanner       planner;

    ControlMode control_mode = ControlMode::PURE_PURSUIT;

    // Đồng bộ với planner: vx = 0.08 m/s
    const float desired_velocity = 0.08f;

    // Thời gian giữ obstacle cũ nếu YOLO mất detection tạm thời
    const int obstacle_timeout_ms = 2000;

    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
};

#endif // LOGIC_HPP