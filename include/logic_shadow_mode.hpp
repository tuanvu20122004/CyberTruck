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

enum class RunMode {
    Shadow,        // chỉ log policy, ưu tiên MPC
    PolicyRollout  // ưu tiên policy, MPC chỉ fallback khi unsafe
};

enum class ControlSource {
    Policy = 0,
    MpcFallback = 1,
    HoldLast = 2,
};

class Logic {
public:
    Logic(const std::string& videoPath,
          const std::string& policyPath,
          const std::string& modelName,
          const std::string& csvPath,
          RunMode runMode,
          bool enableExactQ);

    void run();

private:
    void cameraLoop();
    void controlLoop();
    void keyboardLoop();

    void openShadowCsv(const std::string& filename);

    void logShadowRow(long long timestamp_ms,
                      int frame_id,
                      const MpcState& state,
                      float prev_raw_steering,
                      float raw_steering_policy,
                      float raw_steering_expert,
                      float raw_steering_cmd,
                      float raw_abs_error,
                      double q_policy,
                      double q_mpc,
                      double q_gap,
                      float steering_sent,
                      int servo_command,
                      bool fallback_to_mpc,
                      ControlSource control_source,
                      const std::string& fallback_reason,
                      float lane_width_px);

    LaneDetector  detector;
    MpcController mpc;
    Communication comm;
    Trans_UDP     udp_send;
    PolicyModel   policy_model;
    Logger        logger;

    std::ofstream shadow_csv;

    std::string model_name_;
    std::string csv_path_;
    RunMode run_mode_;
    bool enable_exact_q_;

    float desired_velocity_ = 0.15f;
    const float max_raw_steering_deg_ = 28.0f;
    const int command_period_ms_ = 50;

    float raw_abs_error_fallback_deg_ = 6.0f;
    float max_delta_policy_deg_ = 8.0f;
    float delta_mismatch_fallback_deg_ = 3.0f;
    float max_lateral_deviation_fallback_m_ = 0.12f;
    float max_yaw_error_fallback_rad_ = 0.18f;
    float max_curvature_fallback_ = 0.80f;
    double q_gap_fallback_threshold_ = 50.0;
    bool hold_last_on_invalid_state_ = true;

    std::atomic<bool> drive_enabled{false};
    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
    int               frame_id_ = 0;
};

#endif