#ifndef LOGIC_HPP
#define LOGIC_HPP

#include "LaneDetector.hpp"
#include "MpcController.hpp"
#include "communication.hpp"
#include "logger.hpp"
#include "Trans_UDP.hpp"
#include "RlMpcTuner.hpp"

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <string>

class Logic {
public:
    explicit Logic(const std::string& videoPath);
    void run();

private:
    // Core modules
    LaneDetector   detector;
    MpcController  mpc;
    Communication  comm;
    Trans_UDP      udp_send;
    Logger         logger;
    RlMpcTuner     rl_tuner;

    // Runtime
    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;

    // Vehicle config
    const float desired_velocity = 0.04f;

private:
    void cameraLoop();
    void controlLoop();
};

#endif // LOGIC_HPP