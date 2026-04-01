#include "logic.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <pthread.h>

namespace {
void bindToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "[LOGIC] Error setting thread affinity to core " << core_id << std::endl;
    } else {
        std::cout << "[LOGIC] Thread bound to core " << core_id << std::endl;
    }
    }
}

Logic::Logic(const std::string& videoPath)
    : detector(videoPath, 640, 480),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.102", 9996),
      logger("MPC_RL_Log.txt")
{
    // Init MPC base weights
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        std::cerr << "[LOGIC] LaneDetector/Camera khong mo duoc." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    std::cout << "[LOGIC] System initialized." << std::endl;
}

void Logic::cameraLoop() {
    bindToCore(0);

    cv::Mat frame;
    while (running.load()) {
        if (!detector.getFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = LaneDetector.getBirdEyeView();
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            running.store(false);
            break;
        }
    }
}

void Logic::controlLoop() {
    bindToCore(1);

    cv::Mat frame_local;
    auto last_send = std::chrono::steady_clock::now();

    float prev_steering = 0.0f;
    bool has_prev_rl_state = false;
    RlMpcState prev_rl_state{};

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

        // Chỉ thread này mới xử lý detector để tránh race condition
        detector.processFrame(frame_local);

        std::vector<cv::Point> centerline = detector.getCenterline();
        cv::Mat birdEyeView = detector.getBirdEyeView();
        MpcState state = mpc.computeMpcParameters(centerline, birdEyeView);

        // Gửi BEV
        auto now = std::chrono::steady_clock::now();

        // =====================
        // 1. Gửi BEV (10 Hz)
        // =====================
        static auto last_udp = now;

        if (!birdEyeView.empty() &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_udp).count() >= 50)
        {
            udp_send.sendFrame(birdEyeView, 60);
            last_udp = now;
        }

        // =====================
        // 2. CONTROL (không bị block)
        // =====================
        if (!state.is_valid) {
            return; // hoặc continue
        }
        // =====================================================
        // 1) Tạo RL state hiện tại
        // =====================================================
        RlMpcState rl_state{};
        rl_state.lateral_error = state.lateral_deviation;
        rl_state.yaw_error = state.yaw_angle;
        rl_state.velocity = desired_velocity;
        rl_state.curvature = state.curvature.empty() ? 0.0f : state.curvature[0];
        rl_state.prev_steering = prev_steering;

        // =====================================================
        // 2) Nếu đã có state trước đó thì update RL bằng next_state hiện tại
        // =====================================================
        if (has_prev_rl_state) {
            float reward = rl_tuner.computeReward(prev_rl_state, prev_steering, prev_rl_state.prev_steering);
            rl_tuner.update(reward, rl_state);
        }

        // =====================================================
        // 3) RL suy ra trọng số mới cho MPC
        // =====================================================
        RlMpcWeights weights = rl_tuner.infer(rl_state);

        // Update weights online
        mpc.setWeights(weights.Q1, weights.Q2, weights.R);

        // =====================================================
        // 4) MPC tính steering
        // =====================================================
        float steering = mpc.computeSteeringAngle(state, desired_velocity);

        // Nội suy thực nghiệm
        steering = 0.02f * std::pow(steering, 3) + 1.15f * steering;

        // Clamp theo cơ cấu lái
        if (steering < -25.0f) steering = -25.0f;
        if (steering >  25.0f) steering =  25.0f;

        std::cout << "[RL] Q1=" << weights.Q1
                  << " Q2=" << weights.Q2
                  << " R="  << weights.R
                  << " | steer=" << steering << std::endl;

        int servo = static_cast<int>(std::lround(97.0f + steering));
        comm.sendCommands(desired_velocity, servo);

        // =====================================================
        // 5) Log tất cả vào 1 file
        // =====================================================
        std::stringstream ss;
        ss << "lat=" << rl_state.lateral_error
           << ", yaw=" << rl_state.yaw_error
           << ", vel=" << rl_state.velocity
           << ", curv=" << rl_state.curvature
           << ", prevSteer=" << rl_state.prev_steering
           << ", Q1=" << weights.Q1
           << ", Q2=" << weights.Q2
           << ", R=" << weights.R
           << ", steer=" << steering
           << ", servo=" << servo;

        logger.log(ss.str(), 0.0);

        // =====================================================
        // 6) Lưu state cho vòng sau
        // =====================================================
        prev_rl_state = rl_state;
        prev_rl_state.prev_steering = prev_steering;
        prev_steering = steering;
        has_prev_rl_state = true;
    }
}

void Logic::run() {
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

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}