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

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

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
    float sent = 0.02f * std::pow(raw_steering, 3.0f) + 1.2f * raw_steering;
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

struct TerminalModeGuard {
    termios oldt{};
    bool ok = false;

    TerminalModeGuard() {
        if (tcgetattr(STDIN_FILENO, &oldt) == 0) {
            termios newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);
            ok = true;
        }
    }

    ~TerminalModeGuard() {
        if (ok) {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        }
    }
};

int readKeyNonBlocking()
{
    unsigned char ch;
    int n = ::read(STDIN_FILENO, &ch, 1);
    if (n == 1) return static_cast<int>(ch);
    return -1;
}

Logic::Logic(const std::string& videoPath,
             const std::string& policyPath,
             const std::string& modelName,
             const std::string& csvPath,
             RunMode runMode,
             bool enableExactQ)
    : detector(videoPath, 640, 480),
      mpc(),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.103", 9996),
      policy_model(policyPath),
      logger(modelName + "_runtime_log.txt"),
      model_name_(modelName),
      csv_path_(csvPath),
      run_mode_(runMode),
      enable_exact_q_(enableExactQ)
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

    openShadowCsv(csv_path_);

    std::cout << "[LOGIC] model=" << model_name_
              << " policy=" << policyPath
              << " csv=" << csv_path_
              << " run_mode=" << (run_mode_ == RunMode::PolicyRollout ? "policy_rollout" : "shadow")
              << " exact_q=" << (enable_exact_q_ ? "on" : "off")
              << std::endl;
}

void Logic::openShadowCsv(const std::string& filename)
{
    shadow_csv.open(filename, std::ios::out | std::ios::trunc);
    if (!shadow_csv.is_open()) {
        throw std::runtime_error("[LOGIC] Cannot open CSV: " + filename);
    }

    shadow_csv
        << "model_name,run_mode,"
        << "timestamp_ms,frame_id,is_valid,"
        << "lateral_deviation,yaw_angle,"
        << "curvature_0,curvature_1,curvature_2,curvature_3,"
        << "velocity,prev_steering_raw,"
        << "raw_steering_policy,raw_steering_expert,raw_steering_cmd,"
        << "raw_abs_error,"
        << "q_policy,q_mpc,q_gap,"
        << "steering_sent,servo_command,"
        << "fallback_to_mpc,control_source,fallback_reason,"
        << "lane_width_px\n";
}

void Logic::logShadowRow(long long timestamp_ms,
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
                         float lane_width_px)
{
    if (!shadow_csv.is_open()) {
        return;
    }

    shadow_csv << '"' << model_name_ << '"' << ','
               << '"' << (run_mode_ == RunMode::PolicyRollout ? "policy_rollout" : "shadow") << '"' << ','
               << timestamp_ms << ','
               << frame_id << ','
               << (state.is_valid ? 1 : 0) << ','
               << state.lateral_deviation << ','
               << state.yaw_angle << ','
               << (state.curvature.size() > 0 ? state.curvature[0] : 0.0f) << ','
               << (state.curvature.size() > 1 ? state.curvature[1] : 0.0f) << ','
               << (state.curvature.size() > 2 ? state.curvature[2] : 0.0f) << ','
               << (state.curvature.size() > 3 ? state.curvature[3] : 0.0f) << ','
               << (drive_enabled.load() ? desired_velocity_ : 0.0f) << ','
               << prev_raw_steering << ','
               << raw_steering_policy << ','
               << raw_steering_expert << ','
               << raw_steering_cmd << ','
               << raw_abs_error << ','
               << q_policy << ','
               << q_mpc << ','
               << q_gap << ','
               << steering_sent << ','
               << servo_command << ','
               << (fallback_to_mpc ? 1 : 0) << ','
               << static_cast<int>(control_source) << ','
               << '"' << fallback_reason << '"' << ','
               << lane_width_px
               << '\n';
}

void Logic::keyboardLoop()
{
    TerminalModeGuard term_guard;

    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    std::cout << "[KEYBOARD] Press r=run, s=stop, q=quit" << std::endl;

    while (running.load()) {
        const int key = readKeyNonBlocking();

        if (key == 'r' || key == 'R') {
            drive_enabled.store(true);
            std::cout << "\n[KEYBOARD] DRIVE ENABLED" << std::endl;
        }
        else if (key == 's' || key == 'S') {
            drive_enabled.store(false);
            std::cout << "\n[KEYBOARD] DRIVE DISABLED" << std::endl;
        }
        else if (key == 'q' || key == 'Q') {
            drive_enabled.store(false);
            running.store(false);
            std::cout << "\n[KEYBOARD] QUIT" << std::endl;
            break;
        }

        else if (key == 'n') {
            std::cout << "[KEY] Next model (stop current run)\n";
            running = false;   // 🔥 QUAN TRỌNG
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    fcntl(STDIN_FILENO, F_SETFL, flags);
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

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Logic::controlLoop()
{
    bindToCore(1);

    using clock = std::chrono::steady_clock;
    auto next_time = clock::now();

    float prev_raw_steering_policy = 0.0f;
    float prev_raw_steering_expert = 0.0f;
    float last_cmd_raw = 0.0f;

    while (running.load()) {
        next_time += std::chrono::milliseconds(command_period_ms_);

        cv::Mat frame_local;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (latest_frame.empty()) {
                std::this_thread::sleep_until(next_time);
                continue;
            }
            frame_local = latest_frame.clone();
        }

        detector.processFrame(frame_local);
        const MpcState state = detector.computeMpcState();

        float raw_steering_policy = prev_raw_steering_policy;
        float raw_steering_expert = prev_raw_steering_expert;
        float raw_steering_cmd    = prev_raw_steering_policy;

        float abs_policy_expert_err = 0.0f;
        float delta_policy = 0.0f;
        float delta_expert = 0.0f;
        float delta_mismatch = 0.0f;
        float kappa_max = 0.0f;

        bool fallback_to_mpc = false;
        std::string fallback_reason = "policy";
        ControlSource control_source = ControlSource::Policy;

        double q_policy = std::numeric_limits<double>::quiet_NaN();
        double q_mpc    = std::numeric_limits<double>::quiet_NaN();
        double q_gap    = std::numeric_limits<double>::quiet_NaN();

        if (!state.is_valid) {
            fallback_to_mpc = true;
            fallback_reason = "invalid_state";

            if (hold_last_on_invalid_state_) {
                raw_steering_cmd = last_cmd_raw;
                control_source = ControlSource::HoldLast;
            } else {
                raw_steering_cmd = prev_raw_steering_expert;
                control_source = ControlSource::MpcFallback;
            }
        } else {
            const PolicyModel::FeatureVector features =
                PolicyModel::buildFeatures(state, desired_velocity_, prev_raw_steering_policy);

            raw_steering_policy = policy_model.infer(features);
            raw_steering_expert = mpc.computeSteeringAngle(state, desired_velocity_);

            abs_policy_expert_err = std::abs(raw_steering_policy - raw_steering_expert);
            delta_policy = raw_steering_policy - prev_raw_steering_policy;
            delta_expert = raw_steering_expert - prev_raw_steering_expert;
            delta_mismatch = std::abs(delta_policy - delta_expert);

            const float ey_abs  = std::abs(state.lateral_deviation);
            const float yaw_abs = std::abs(state.yaw_angle);
            kappa_max = maxAbsCurvature4(state);

            if (enable_exact_q_) {
                q_policy = mpc.evaluateExactQ(state, desired_velocity_, raw_steering_policy);
                q_mpc    = mpc.evaluateExactQ(state, desired_velocity_, raw_steering_expert);

                if (std::isfinite(q_policy) && std::isfinite(q_mpc)) {
                    q_gap = q_policy - q_mpc;
                }
            }

            // --- default command selection ---
            if (run_mode_ == RunMode::PolicyRollout) {
                raw_steering_cmd = raw_steering_policy;
                control_source = ControlSource::Policy;
            } else {
                raw_steering_cmd = raw_steering_expert;
                control_source = ControlSource::MpcFallback;
                fallback_reason = "shadow_mode_mpc";
            }

            // --- safety arbitration ---
            if (run_mode_ == RunMode::PolicyRollout) {
                if (abs_policy_expert_err > raw_abs_error_fallback_deg_) {
                    fallback_to_mpc = true;
                    fallback_reason = "abs_policy_expert_err";
                }
                else if (std::abs(delta_policy) > max_delta_policy_deg_) {
                    fallback_to_mpc = true;
                    fallback_reason = "delta_policy";
                }
                else if (delta_mismatch > delta_mismatch_fallback_deg_) {
                    fallback_to_mpc = true;
                    fallback_reason = "delta_mismatch";
                }
                else if (ey_abs > max_lateral_deviation_fallback_m_) {
                    fallback_to_mpc = true;
                    fallback_reason = "large_lateral_deviation";
                }
                else if (yaw_abs > max_yaw_error_fallback_rad_) {
                    fallback_to_mpc = true;
                    fallback_reason = "large_yaw_error";
                }
                else if (kappa_max > max_curvature_fallback_) {
                    fallback_to_mpc = true;
                    fallback_reason = "high_curvature";
                }
                else if (enable_exact_q_ && std::isfinite(q_gap) && q_gap > q_gap_fallback_threshold_) {
                    fallback_to_mpc = true;
                    fallback_reason = "exact_q_gap";
                }

                if (fallback_to_mpc) {
                    raw_steering_cmd = raw_steering_expert;
                    control_source = ControlSource::MpcFallback;
                }
            }
        }

        raw_steering_cmd = std::clamp(raw_steering_cmd,
                                      -max_raw_steering_deg_,
                                      max_raw_steering_deg_);

        const float steering_sent = remapSteeringForActuator(raw_steering_cmd);
        const int servo_command = static_cast<int>(std::lround(93.5f + steering_sent));

        const float speed_cmd = drive_enabled.load() ? desired_velocity_ : 0.0f;
        const int servo_safe = drive_enabled.load() ? servo_command : 80;
        comm.sendCommands(speed_cmd, servo_safe);

        const long long timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const int current_frame_id = frame_id_++;

        logShadowRow(timestamp_ms,
                     current_frame_id,
                     state,
                     prev_raw_steering_policy,
                     raw_steering_policy,
                     raw_steering_expert,
                     raw_steering_cmd,
                     abs_policy_expert_err,
                     q_policy,
                     q_mpc,
                     q_gap,
                     steering_sent,
                     servo_command,
                     fallback_to_mpc,
                     control_source,
                     fallback_reason,
                     detector.getLaneWidthPx());

        std::cout << "[ROLLOUT] model=" << model_name_
                  << " frame=" << current_frame_id
                  << " mode=" << (run_mode_ == RunMode::PolicyRollout ? "policy_rollout" : "shadow")
                  << " src=" << static_cast<int>(control_source)
                  << " raw_policy=" << raw_steering_policy
                  << " raw_expert=" << raw_steering_expert
                  << " raw_cmd=" << raw_steering_cmd
                  << " q_gap=" << q_gap
                  << " reason=" << fallback_reason
                  << std::endl;

        prev_raw_steering_policy = raw_steering_policy;
        prev_raw_steering_expert = raw_steering_expert;
        last_cmd_raw = raw_steering_cmd;

        std::this_thread::sleep_until(next_time);
    }
}

void Logic::run()
{
    std::thread cam_thread(&Logic::cameraLoop, this);
    std::thread ctrl_thread(&Logic::controlLoop, this);
    std::thread key_thread(&Logic::keyboardLoop, this);

    cam_thread.join();
    ctrl_thread.join();
    key_thread.join();

    if (shadow_csv.is_open()) {
        shadow_csv.close();
    }
}
