#include "logic_shadow_mode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <sstream>
#include <thread>

namespace {

void bindToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    const int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "[LOGIC] Error setting thread affinity to core " << core_id << std::endl;
    } else {
        std::cout << "[LOGIC] Thread bound to core " << core_id << std::endl;
    }
}

float remapSteeringForActuator(float raw_steering)
{
    float sent = 0.02f * std::pow(raw_steering, 3.0f) + 1.15f * raw_steering;
    sent = std::clamp(sent, -25.0f, 25.0f);
    return sent;
}

float clampRawSteering(float raw_steering, float limit_deg)
{
    if (!std::isfinite(raw_steering)) {
        return 0.0f;
    }
    return std::clamp(raw_steering, -limit_deg, limit_deg);
}

float getCurvatureAt(const MpcState& state, std::size_t idx)
{
    return (state.curvature.size() > idx) ? state.curvature[idx] : 0.0f;
}

} // namespace

Logic::Logic(const std::string& videoPath, const std::string& policyPath)
    : detector(videoPath, 640, 480),
      mpc(),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.100", 9996),
      policy_model(policyPath),
      logger("policy_fallback_log.txt")
{
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        throw std::runtime_error("[LOGIC] LaneDetector not opened");
    }
    if (!policy_model.isLoaded()) {
        throw std::runtime_error("[LOGIC] Policy model not loaded");
    }

    openShadowCsv("policy_fallback_log.csv");
    std::cout << "[LOGIC] Policy control + MPC fallback started with policy JSON: "
              << policyPath << std::endl;
}

void Logic::openShadowCsv(const std::string& filename)
{
    shadow_csv.open(filename, std::ios::out | std::ios::trunc);
    if (!shadow_csv.is_open()) {
        throw std::runtime_error("[LOGIC] Cannot open CSV: " + filename);
    }

    shadow_csv
        << "timestamp_ms,frame_id,is_valid,fallback_to_mpc,fallback_reason,"
        << "lateral_deviation,yaw_angle,"
        << "curvature_0,curvature_1,curvature_2,curvature_3,curvature_4,curvature_5,curvature_6,curvature_7,curvature_8,curvature_9,"
        << "velocity,prev_steering_raw,"
        << "raw_steering_policy,raw_steering_expert,raw_steering_executed,raw_abs_error,"
        << "steering_sent_executed,servo_command,lane_width_px\n";
}

void Logic::logCsvRow(long long timestamp_ms,
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
                      float lane_width_px)
{
    if (!shadow_csv.is_open()) {
        return;
    }

    const float raw_abs_error = (std::isfinite(raw_steering_policy) && std::isfinite(raw_steering_expert))
        ? std::abs(raw_steering_expert - raw_steering_policy)
        : 0.0f;

    shadow_csv
        << timestamp_ms << ','
        << frame_id << ','
        << (is_valid ? 1 : 0) << ','
        << (fallback_to_mpc ? 1 : 0) << ','
        << fallback_reason << ','
        << state.lateral_deviation << ','
        << state.yaw_angle << ','
        << getCurvatureAt(state, 0) << ','
        << getCurvatureAt(state, 1) << ','
        << getCurvatureAt(state, 2) << ','
        << getCurvatureAt(state, 3) << ','
        << getCurvatureAt(state, 4) << ','
        << getCurvatureAt(state, 5) << ','
        << getCurvatureAt(state, 6) << ','
        << getCurvatureAt(state, 7) << ','
        << getCurvatureAt(state, 8) << ','
        << getCurvatureAt(state, 9) << ','
        << velocity << ','
        << prev_raw_steering << ','
        << raw_steering_policy << ','
        << raw_steering_expert << ','
        << raw_steering_executed << ','
        << raw_abs_error << ','
        << steering_sent_executed << ','
        << servo_command << ','
        << lane_width_px
        << '\n';
}

void Logic::cameraLoop()
{
    bindToCore(0);

    cv::Mat frame;
    while (running.load()) {
        if (!detector.getFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = detector.getFrameResize().clone();
        }

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            running.store(false);
            break;
        }
    }
}

void Logic::controlLoop()
{
    bindToCore(1);

    cv::Mat frame_local;
    auto last_send = std::chrono::steady_clock::now();

    float prev_feature_steering = 0.0f;
    float last_executed_raw_steering = 0.0f;

    while (running.load()) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!latest_frame.empty()) {
                frame_local = latest_frame.clone();
            } else {
                frame_local.release();
            }
        }

        if (frame_local.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        detector.processFrame(frame_local);

        const std::vector<cv::Point> centerline = detector.getCenterline();
        const cv::Mat birdEyeView = detector.getBirdEyeView();
        const MpcState state = mpc.computeMpcParameters(centerline, birdEyeView);

        const cv::Mat bev = detector.getBirdEyeView();
        if (!bev.empty()) {
            udp_send.sendFrame(bev, 60);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() < command_period_ms_) {
            continue;
        }
        last_send = now;

        const long long timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const int current_frame_id = frame_id_++;
        const float lane_width_px = detector.getLaneWidthPx();

        if (!state.is_valid) {
            const float raw_executed = hold_last_on_invalid_state_ ? last_executed_raw_steering : 0.0f;
            const float steering_sent = remapSteeringForActuator(raw_executed);
            const int servo_command = static_cast<int>(std::lround(97.0f + steering_sent));
            comm.sendCommands(desired_velocity, servo_command);

            logCsvRow(timestamp_ms,
                      current_frame_id,
                      false,
                      true,
                      "invalid_state_hold",
                      state,
                      desired_velocity,
                      prev_feature_steering,
                      0.0f,
                      0.0f,
                      raw_executed,
                      steering_sent,
                      servo_command,
                      lane_width_px);

            logger.log("invalid_state_hold", 0.0);
            std::cout << "[FALLBACK] frame=" << current_frame_id
                      << " reason=invalid_state_hold"
                      << " raw_executed=" << raw_executed << std::endl;
            continue;
        }

        const PolicyModel::FeatureVector features =
            PolicyModel::buildFeatures(state, desired_velocity, prev_feature_steering);

        float raw_steering_policy = clampRawSteering(policy_model.infer(features), max_raw_steering_deg_);
        float raw_steering_expert = clampRawSteering(mpc.computeSteeringAngle(state, desired_velocity), max_raw_steering_deg_);

        bool fallback_to_mpc = false;
        std::string fallback_reason = "none";

        if (!std::isfinite(raw_steering_policy)) {
            fallback_to_mpc = true;
            fallback_reason = "policy_nan";
            raw_steering_policy = 0.0f;
        }

        const float raw_abs_error = std::abs(raw_steering_expert - raw_steering_policy);
        if (!fallback_to_mpc && raw_abs_error > raw_abs_error_fallback_deg_) {
            fallback_to_mpc = true;
            fallback_reason = "abs_error";
        }

        const float policy_delta = std::abs(raw_steering_policy - prev_feature_steering);
        if (!fallback_to_mpc && policy_delta > max_delta_policy_deg_) {
            fallback_to_mpc = true;
            fallback_reason = "delta_policy";
        }

        const float raw_steering_executed = fallback_to_mpc ? raw_steering_expert : raw_steering_policy;
        const float steering_sent_executed = remapSteeringForActuator(raw_steering_executed);
        const int servo_command = static_cast<int>(std::lround(97.0f + steering_sent_executed));
        comm.sendCommands(desired_velocity, servo_command);

        logCsvRow(timestamp_ms,
                  current_frame_id,
                  true,
                  fallback_to_mpc,
                  fallback_reason,
                  state,
                  desired_velocity,
                  prev_feature_steering,
                  raw_steering_policy,
                  raw_steering_expert,
                  raw_steering_executed,
                  steering_sent_executed,
                  servo_command,
                  lane_width_px);

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "policy=" << raw_steering_policy
           << ", expert=" << raw_steering_expert
           << ", exec=" << raw_steering_executed
           << ", abs_err=" << raw_abs_error
           << ", fallback=" << (fallback_to_mpc ? 1 : 0)
           << ", reason=" << fallback_reason
           << ", ey=" << state.lateral_deviation
           << ", yaw=" << state.yaw_angle;
        logger.log(ss.str(), 0.0);

        std::cout << "[CONTROL] frame=" << current_frame_id
                  << " policy=" << raw_steering_policy
                  << " expert=" << raw_steering_expert
                  << " exec=" << raw_steering_executed
                  << " abs_err=" << raw_abs_error
                  << " fallback=" << (fallback_to_mpc ? 1 : 0)
                  << " reason=" << fallback_reason
                  << std::endl;

        prev_feature_steering = raw_steering_executed;
        last_executed_raw_steering = raw_steering_executed;
    }
}

void Logic::run()
{
    std::thread camera_thread(&Logic::cameraLoop, this);
    std::thread control_thread(&Logic::controlLoop, this);

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (camera_thread.joinable()) {
        camera_thread.join();
    }
    if (control_thread.joinable()) {
        control_thread.join();
    }

    if (shadow_csv.is_open()) {
        shadow_csv.close();
    }

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}
