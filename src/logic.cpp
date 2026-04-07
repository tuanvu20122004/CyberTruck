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
      logger("MPC_RL_Log.txt"),
      dataset_logger("mpc_expert_dataset.csv"),
      frame_id_(0)
{
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        std::cerr << "[LOGIC] LaneDetector/Camera khong mo duoc." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    if (!dataset_logger.isOpen()) {
        throw std::runtime_error("Dataset logger not opened");
    }

    std::cout << "[LOGIC] MPC dataset collection mode." << std::endl;
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
            latest_frame = detector.getFrameResize().clone();
        }

        int key = cv::waitKey(1);
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

        cv::Mat bev = detector.getBirdEyeView();
        if (!bev.empty()) {
            udp_send.sendFrame(bev, 60);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 50) {
            last_send = now;

            if (!state.is_valid) {
                continue;
            }

            float raw_steering = mpc.computeSteeringAngle(state, desired_velocity);

            float steering_sent = 0.02f * std::pow(raw_steering, 3) + 1.15f * raw_steering;

            if (steering_sent <= -25.0f)
                steering_sent = -25.0f;
            else if (steering_sent >= 25.0f)
                steering_sent = 25.0f;

            int servo = static_cast<int>(std::lround(97.0f + steering_sent));
            comm.sendCommands(desired_velocity, servo);

            // lấy data
            DatasetSample sample{};
            sample.timestamp_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            sample.frame_id = frame_id_++;
            sample.is_valid = 1;

            sample.lateral_deviation = static_cast<float>(state.lateral_deviation);
            sample.yaw_angle = static_cast<float>(state.yaw_angle);

            sample.curvature_0 = state.curvature.size() > 0 ? state.curvature[0] : 0.0f;
            sample.curvature_1 = state.curvature.size() > 1 ? state.curvature[1] : 0.0f;
            sample.curvature_2 = state.curvature.size() > 2 ? state.curvature[2] : 0.0f;
            sample.curvature_3 = state.curvature.size() > 3 ? state.curvature[3] : 0.0f;

            sample.velocity = desired_velocity;
            sample.prev_steering = prev_raw_steering;
            sample.expert_steering = raw_steering;
            sample.steering_sent = steering_sent;
            sample.servo_command = servo;
            sample.lane_width_px = detector.getLaneWidthPx();

            dataset_logger.logSample(sample);

            prev_raw_steering = raw_steering;

            std::cout << "[DATASET] frame=" << sample.frame_id
                      << " ey=" << sample.lateral_deviation
                      << " yaw=" << sample.yaw_angle
                      << " u*=" << sample.expert_steering
                      << std::endl;
        }
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