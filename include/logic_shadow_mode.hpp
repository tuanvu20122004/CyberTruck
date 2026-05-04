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
    // runMode:
    //   "dagger" : policy controls when safe; MPC is queried as expert and used as safety fallback.
    //   "shadow" : MPC controls; policy is only logged for evaluation.
    //   "expert" : MPC controls and the log can be used as an expert dataset D0.
    Logic(const std::string& videoPath,
          const std::string& policyPath,
          const std::string& logPrefix = "dagger_run",
          const std::string& runMode = "dagger");

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
                      float delta_policy,
                      float delta_expert,
                      float delta_mismatch,
                      double q_policy,
                      double q_mpc,
                      double q_gap,
                      float steering_sent,
                      int servo_command,
                      bool fallback_to_mpc,
                      const std::string& fallback_reason,
                      float lane_width_px);

    LaneDetector  detector;
    MpcController mpc;
    Communication comm;
    Trans_UDP     udp_send;
    PolicyModel   policy_model;
    Logger        logger;

    std::ofstream shadow_csv;
    std::string   log_prefix_;
    std::string   run_mode_;

    float desired_velocity_ = 0.08f;
    const float max_raw_steering_deg_ = 28.0f;
    const int command_period_ms_ = 50;

    // Safety fallback thresholds for DAgger collection.
    float raw_abs_error_fallback_deg_ = 6.0f;
    float max_delta_policy_deg_ = 8.0f;
    float delta_mismatch_fallback_deg_ = 3.0f;
    float max_lateral_deviation_fallback_m_ = 0.12f;
    float max_yaw_error_fallback_rad_ = 0.18f;
    float max_curvature_fallback_ = 0.80f;
    double q_gap_fallback_threshold_ = 50.0;
    bool hold_last_on_invalid_state_ = true;
    bool log_only_when_drive_enabled_ = true;

    float prev_raw_steering_policy_ = 0.0f;
    float prev_raw_steering_expert_ = 0.0f;
    int   last_servo_command_ = 80;

    std::atomic<bool> drive_enabled{false};
    std::atomic<bool> running{true};
    std::mutex        frame_mutex;
    cv::Mat           latest_frame;
    int               frame_id_ = 0;
};

#endif
