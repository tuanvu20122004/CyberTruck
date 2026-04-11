#include "logic_shadow_mode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
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
    float sent = 0.00f * std::pow(raw_steering, 3.0f) + 3.0f * raw_steering;
    sent = std::clamp(sent, -25.0f, 25.0f);
    return sent;
}

float maxAbsCurvature4(const MpcState& state)
{
    float kmax = 0.0f;
    for (size_t i = 0; i < state.curvature.size() && i < 4; ++i) {
        kmax = std::max(kmax, std::abs(state.curvature[i]));
    }
    return kmax;
}

} // namespace

Logic::Logic(const std::string& videoPath, const std::string& policyPath)
    : detector(videoPath, 640, 480),
      mpc(),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.106", 9996),
      policy_model(policyPath),
      logger("policy_dagger_log.txt")
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

    openShadowCsv("policy_dagger_log.csv");
    std::cout << "[LOGIC] Policy-control + MPC-fallback started with policy JSON: "
              << policyPath << std::endl;
}

void Logic::openShadowCsv(const std::string& filename)
{
    shadow_csv.open(filename, std::ios::out | std::ios::trunc);
    if (!shadow_csv.is_open()) {
        throw std::runtime_error("[LOGIC] Cannot open DAgger CSV: " + filename);
    }

    shadow_csv
        << "timestamp_ms,frame_id,is_valid,"
        << "lateral_deviation,yaw_angle,"
        << "curvature_0,curvature_1,curvature_2,curvature_3,"
        << "velocity,prev_steering_raw,"
        << "raw_steering_policy,raw_steering_expert,raw_steering_cmd,raw_abs_error,"
        << "steering_sent,servo_command,"
        << "fallback_to_mpc,fallback_reason,"
        << "lane_width_px\n";
}

void Logic::logShadowRow(long long timestamp_ms,
                         int frame_id,
                         const MpcState& state,
                         float prev_raw_steering,
                         float raw_steering_policy,
                         float raw_steering_expert,
                         float raw_steering_cmd,
                         float steering_sent,
                         int servo_command,
                         bool fallback_to_mpc,
                         const std::string& fallback_reason,
                         float lane_width_px)
{
    if (!shadow_csv.is_open()) {
        return;
    }

    shadow_csv << timestamp_ms << ','
               << frame_id << ','
               << (state.is_valid ? 1 : 0) << ','
               << state.lateral_deviation << ','
               << state.yaw_angle << ','
               << (state.curvature.size() > 0 ? state.curvature[0] : 0.0f) << ','
               << (state.curvature.size() > 1 ? state.curvature[1] : 0.0f) << ','
               << (state.curvature.size() > 2 ? state.curvature[2] : 0.0f) << ','
               << (state.curvature.size() > 3 ? state.curvature[3] : 0.0f) << ','
               << desired_velocity << ','
               << prev_raw_steering << ','
               << raw_steering_policy << ','
               << raw_steering_expert << ','
               << raw_steering_cmd << ','
               << std::abs(raw_steering_expert - raw_steering_policy) << ','
               << steering_sent << ','
               << servo_command << ','
               << (fallback_to_mpc ? 1 : 0) << ','
               << '"' << fallback_reason << '"' << ','
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

    float prev_raw_steering = 0.0f;
    float prev_raw_steering_policy = 0.0f;
    float prev_raw_steering_expert = 0.0f;

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
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() < 20) {
            continue;
        }
        last_send = now;

        if (!state.is_valid) {
            continue;
        }

        const PolicyModel::FeatureVector features =
            PolicyModel::buildFeatures(state, desired_velocity, prev_raw_steering);

        float raw_steering_policy = 0.0f;
        try {
            raw_steering_policy = policy_model.infer(features);
        } catch (const std::exception& e) {
            std::cerr << "[POLICY] infer failed: " << e.what() << std::endl;
            continue;
        }

        const float raw_steering_expert = mpc.computeSteeringAngle(state, desired_velocity);

        float raw_steering_cmd = raw_steering_policy;
        bool fallback_to_mpc = false;
        std::string fallback_reason = "policy";

        const float abs_policy_expert_err = std::abs(raw_steering_policy - raw_steering_expert);
        const float delta_policy = raw_steering_policy - prev_raw_steering_policy;
        const float delta_expert = raw_steering_expert - prev_raw_steering_expert;
        const float delta_mismatch = std::abs(delta_policy - delta_expert);

        const float ey_abs = std::abs(state.lateral_deviation);
        const float yaw_abs = std::abs(state.yaw_angle);
        const float kappa_max = maxAbsCurvature4(state);
        # if 1
        // ===== MPC fallback rules =====
        if (abs_policy_expert_err > 2.0f) {
            raw_steering_cmd = raw_steering_expert;
            fallback_to_mpc = true;
            fallback_reason = "abs_policy_expert_err";
        }

        if (!fallback_to_mpc && delta_mismatch > 3.0f) {
            raw_steering_cmd = raw_steering_expert;
            fallback_to_mpc = true;
            fallback_reason = "delta_mismatch";
        }

        if (!fallback_to_mpc && ey_abs > 0.12f) {
            raw_steering_cmd = raw_steering_expert;
            fallback_to_mpc = true;
            fallback_reason = "large_lateral_deviation";
        }

        if (!fallback_to_mpc && yaw_abs > 0.18f) {
            raw_steering_cmd = raw_steering_expert;
            fallback_to_mpc = true;
            fallback_reason = "large_yaw_error";
        }

        if (!fallback_to_mpc && kappa_max > 0.80f) {
            raw_steering_cmd = raw_steering_expert;
            fallback_to_mpc = true;
            fallback_reason = "high_curvature";
        }
        # endif
        const float steering_sent = remapSteeringForActuator(raw_steering_cmd);
        const int servo_command = static_cast<int>(std::lround(80.0f + steering_sent));
        std:: cout << "SERVO_COMMAND: " << servo_command << std::endl;
        comm.sendCommands(desired_velocity, servo_command);

        const long long timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const int current_frame_id = frame_id_++;

        logShadowRow(timestamp_ms,
                     current_frame_id,
                     state,
                     prev_raw_steering,
                     raw_steering_policy,
                     raw_steering_expert,
                     raw_steering_cmd,
                     steering_sent,
                     servo_command,
                     fallback_to_mpc,
                     fallback_reason,
                     detector.getLaneWidthPx());

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "mode=" << (fallback_to_mpc ? "mpc_fallback" : "policy")
           << ", raw_policy=" << raw_steering_policy
           << ", raw_expert=" << raw_steering_expert
           << ", raw_cmd=" << raw_steering_cmd
           << ", abs_err=" << abs_policy_expert_err
           << ", d_err=" << delta_mismatch
           << ", ey=" << state.lateral_deviation
           << ", yaw=" << state.yaw_angle
           << ", kappa_max=" << kappa_max
           << ", reason=" << fallback_reason;
        logger.log(ss.str(), 0.0);

        // std::cout << "[DAGGER] frame=" << current_frame_id
        //           << " mode=" << (fallback_to_mpc ? "MPC" : "POLICY")
        //           << " raw_policy=" << raw_steering_policy
        //           << " raw_expert=" << raw_steering_expert
        //           << " raw_cmd=" << raw_steering_cmd
        //           << " abs_err=" << abs_policy_expert_err
        //           << " reason=" << fallback_reason
        //           << std::endl;

        // Quan trọng: dùng lệnh THỰC THI thật cho feature frame sau
        prev_raw_steering = raw_steering_cmd;
        prev_raw_steering_policy = raw_steering_policy;
        prev_raw_steering_expert = raw_steering_expert;
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
