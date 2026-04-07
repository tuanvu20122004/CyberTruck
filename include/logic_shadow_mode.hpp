#ifndef LOGIC_SHADOW_MODE_HPP
#define LOGIC_SHADOW_MODE_HPP

#include "LaneDetector.hpp"
#include "MpcController.hpp"
#include "communication.hpp"
#include "logger.hpp"
#include "Trans_UDP.hpp"
#include "PolicyModel.hpp"

#include <opencv2/opencv.hpp>
#include <atomic>
#include <fstream>
#include <mutex>
#include <string>

class Logic {
public:
    explicit Logic(const std::string& videoPath,
                   const std::string& policyPath = "policy_export.json");
    void run();

private:
    void cameraLoop();
    void controlLoop();
    void openShadowCsv(const std::string& filename);
    void logShadowRow(long long timestamp_ms,
                      int frame_id,
                      const MpcState& state,
                      float prev_raw_steering,
                      float raw_steering_mpc,
                      float raw_steering_pred,
                      float steering_sent_mpc,
                      int servo_command,
                      float lane_width_px);

    LaneDetector  detector;
    MpcController mpc;
    Communication comm;
    Trans_UDP     udp_send;
    PolicyModel   policy_model;

    Logger        logger;
    std::ofstream shadow_csv;

    const float desired_velocity = 0.04f;

    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
    int               frame_id_ = 0;
};

#endif // LOGIC_SHADOW_MODE_HPP
