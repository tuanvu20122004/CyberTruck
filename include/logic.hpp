#ifndef LOGIC_HPP
#define LOGIC_HPP

#include "LaneDetector.hpp"
#include "MpcController.hpp"
#include "communication.hpp"
#include "Trans_UDP.hpp"
#include "LaneChangePlanner.hpp"
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <string>

class Logic {
public:
    explicit Logic(const std::string& videoPath);
    void run();

private:
    LaneDetector      detector;
    MpcController     mpc;
    Communication     comm;
    Trans_UDP         udp_yolo;
    Trans_UDP         udp_debug;
    LaneChangePlanner planner;

    // Đồng bộ với planner: vx = 0.08 m/s
    const float desired_velocity = 0.08f;

    // Thời gian giữ obstacle cũ nếu YOLO mất detection tạm thời
    const int obstacle_timeout_ms = 2000;

    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
};

#endif // LOGIC_HPP