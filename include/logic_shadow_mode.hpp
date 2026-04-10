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
    void logCsvRow(long long timestamp_ms,
                   int frame_id,
                   bool is_valid,
                   bool fallback_to_mpc,
                   const std::string& fallback_reason,
                   const MpcState& state,
                   float velocity,
                   float prev_raw_steering,
                   float raw_steering_policy,
                   float raw_steering_expert,
                   float raw_steering_executed,
                   float steering_sent_executed,
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
    const float raw_abs_error_fallback_deg_ = 6.0f;
    const float max_raw_steering_deg_ = 28.0f;
    const float max_delta_policy_deg_ = 8.0f;
    const int command_period_ms_ = 50;
    const bool hold_last_on_invalid_state_ = true;

    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
    int               frame_id_ = 0;
};

#endif // LOGIC_SHADOW_MODE_HPP
