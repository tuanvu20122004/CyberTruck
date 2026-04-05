#include "logic.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>
#include <pthread.h>

void bindToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "[LOGIC] Error setting thread affinity." << std::endl;
    } else {
        std::cout << "[LOGIC] Thread bound to core " << core_id << "." << std::endl;
    }
}

Logic::Logic(const std::string& videoPath)
    : detector(videoPath, 640, 480),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.102", 9996),
      logger("MPC_RL_Log.txt")
{
    // Khởi tạo MPC ban đầu
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        std::cerr << "[LOGIC] LaneDetector/Camera không mở được." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    std::cout << "[LOGIC] MPC initialized." << std::endl;
}

void Logic::run() {
    // ---- Camera thread ----
    std::thread camera_thread([&]() {
        bindToCore(0);
        cv::Mat frame;

        while (running.load()) {
            if (!detector.getFrame(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                latest_frame = frame.clone();
            }

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                running.store(false);
                break;
            }
        }
    });

    // ---- MPC thread ----
    std::thread mpc_thread([&]() {
        bindToCore(1);

        cv::Mat frame_local;
        auto last_send = std::chrono::steady_clock::now();
        float prev_steering = 0.0f;

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

            std::vector<cv::Point> centerline = detector.getCenterline();
            cv::Mat birdEyeView = detector.getBirdEyeView();
            MpcState state = mpc.computeMpcParameters(centerline, birdEyeView);

            // Gửi BEV lên server
            cv::Mat bev = detector.getBirdEyeView();
            if (!bev.empty()) {
                udp_send.sendFrame(bev, 60);
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 100) {
                last_send = now;

                if (state.is_valid) {
                    // =====================================================
                    // 1) Tạo hybrid tuning state từ MPC state
                    // =====================================================
                    RlMpcState rl_state{};
                    rl_state.lateral_error = state.lateral_deviation;
                    rl_state.yaw_error = state.yaw_angle;
                    rl_state.velocity = desired_velocity;
                    rl_state.curvature = state.curvature.empty() ? 0.0f : state.curvature[0];
                    rl_state.prev_steering = prev_steering;

                    // =====================================================
                    // 2) Learner đề xuất weight, nhưng vẫn bám theo expert prior
                    // =====================================================
                    const RlMpcWeights expert_weights = rl_tuner.getExpertWeights(rl_state);
                    const RlMpcWeights weights = rl_tuner.infer(rl_state);

                    // cập nhật MPC weights online
                    mpc.setWeights(weights.Q1, weights.Q2, weights.R);

                    std::cout << "[HYBRID] expert(Q1=" << expert_weights.Q1
                              << ", Q2=" << expert_weights.Q2
                              << ", R="  << expert_weights.R
                              << ") learner(Q1=" << weights.Q1
                              << ", Q2=" << weights.Q2
                              << ", R="  << weights.R
                              << ")" << std::endl;

                    // =====================================================
                    // 3) Tính steering bằng MPC với weights mới
                    // =====================================================
                    float steering = mpc.computeSteeringAngle(state, desired_velocity);

                    steering = 0.02f * std::pow(steering, 3) + 1.15f * steering;

                    // clamp
                    if (steering <= -25.0f)
                        steering = -25.0f;
                    else if (steering >= 25.0f)
                        steering = 25.0f;

                    const float stage_cost = rl_tuner.computeStageCost(rl_state, steering, prev_steering);

                    // =====================================================
                    // LOGGER (1 dòng duy nhất)
                    // =====================================================
                    std::stringstream ss;
                    ss << "lat=" << rl_state.lateral_error
                       << ", yaw=" << rl_state.yaw_error
                       << ", curv=" << rl_state.curvature
                       << ", expertQ1=" << expert_weights.Q1
                       << ", expertQ2=" << expert_weights.Q2
                       << ", expertR=" << expert_weights.R
                       << ", Q1=" << weights.Q1
                       << ", Q2=" << weights.Q2
                       << ", R=" << weights.R
                       << ", steer=" << steering
                       << ", cost=" << stage_cost;

                    logger.log(ss.str(), 0);

                    std::cout << "Gia tri goc lai qua noi suy: " << steering
                              << " | surrogate cost=" << stage_cost << std::endl;

                    int servo = static_cast<int>(std::lround(97.0f + steering));
                    comm.sendCommands(desired_velocity, servo);

                    // =====================================================
                    // 4) Hybrid update: learner-induced state + expert anchor + Q proxy
                    // =====================================================
                    rl_tuner.update(rl_state, steering, prev_steering);

                    prev_steering = steering;
                }
            }
        }
    });

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (camera_thread.joinable()) camera_thread.join();
    if (mpc_thread.joinable()) mpc_thread.join();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}
