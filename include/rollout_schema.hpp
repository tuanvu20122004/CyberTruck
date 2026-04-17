#pragma once

#include <string>
#include <fstream>

// Forward declaration
struct MpcState;

// ================= Metadata =================
struct RolloutMetadata {
    std::string run_id;
    std::string method_name;
    std::string model_tag;
    std::string scenario_id;
    int lap_id = 0;
};

// ================= One CSV row =================
struct RolloutRow {
    // Metadata
    std::string run_id;
    std::string method_name;
    std::string model_tag;
    std::string scenario_id;
    int lap_id = 0;

    // Time
    long long timestamp_ms = 0;
    double time_s = 0.0;
    int frame_id = 0;
    int control_tick_id = 0;

    // Perception / validity
    int is_valid = 0;
    float lane_width_px = 0.0f;
    int left_lane_detected = 0;
    int right_lane_detected = 0;

    // State (features)
    float lateral_deviation = 0.0f;
    float yaw_angle = 0.0f;
    float curvature[4] = {0};
    float velocity = 0.0f;
    float prev_steering_raw = 0.0f;

    // Actions
    float raw_steering_policy = 0.0f;
    float raw_steering_expert = 0.0f;
    float raw_steering_cmd = 0.0f;
    float raw_abs_error = 0.0f;

    // Smoothness
    float delta_policy = 0.0f;
    float delta_expert = 0.0f;
    float delta_cmd = 0.0f;
    float delta_mismatch = 0.0f;

    // Safety indicators
    float ey_abs = 0.0f;
    float yaw_abs = 0.0f;
    float kappa_max = 0.0f;

    // Actuator
    float steering_sent = 0.0f;
    int servo_command = 0;
    float speed_cmd = 0.0f;
    float speed_measured = 0.0f;

    // Events
    int fallback_to_mpc = 0;
    std::string fallback_reason;
    int monitor_warning = 0;
    std::string monitor_reason;
    int stop_event_flag = 0;
    int human_intervention_flag = 0;
    int run_abort_flag = 0;

    // Reference tracking
    float lateral_error_ref = 0.0f;
    float yaw_error_ref = 0.0f;

    // Trajectory (optional)
    float x_vehicle = 0.0f;
    float y_vehicle = 0.0f;
    float yaw_vehicle = 0.0f;
    float x_ref = 0.0f;
    float y_ref = 0.0f;
    float yaw_ref = 0.0f;

    std::string segment_type;

    // Q values
    float exact_q_policy = 0.0f;
    float exact_q_expert = 0.0f;
    float q_gap = 0.0f;
};

// ================= Logger =================
class RolloutCsvLogger {
public:
    RolloutCsvLogger(const RolloutMetadata& meta, const std::string& filename);
    ~RolloutCsvLogger();

    void logRow(const RolloutRow& row);

    static RolloutRow fromState(
        const RolloutMetadata& meta,
        long long timestamp_ms,
        double time_s,
        int frame_id,
        int control_tick_id,
        const MpcState& state,
        float lane_width_px,
        float prev_steering,
        float raw_policy,
        float raw_expert,
        float raw_cmd,
        float delta_policy,
        float delta_expert,
        float delta_cmd,
        float steering_sent,
        int servo_cmd,
        float speed_cmd,
        int fallback_flag,
        const std::string& fallback_reason,
        int monitor_flag,
        const std::string& monitor_reason,
        int stop_flag,
        int human_flag,
        int abort_flag
    );

private:
    std::ofstream file_;
    RolloutMetadata meta_;

    void writeHeader();
    std::string escape(const std::string& s);
};