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
      udp_send("192.168.1.103", 9996),
      logger("MPC_Log.txt")
{
    // Init MPC base weights
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        std::cerr << "[LOGIC] LaneDetector/Camera khong mo duoc." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    std::cout << "[LOGIC] Pure MPC system initialized." << std::endl;
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

void Logic::controlLoop() {
    bindToCore(1);

    cv::Mat frame_local;

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

        auto now = std::chrono::steady_clock::now();
        static auto last_udp = now;

        // =====================
        // 1. Gửi BEV debug
        // =====================
        if (!birdEyeView.empty() &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_udp).count() >= 50)
        {
            udp_send.sendFrame(birdEyeView, 60);
            last_udp = now;
        }

        // =====================
        // 2. Pure MPC control
        // =====================
        if (!state.is_valid) {
            continue;
        }

        float steering = mpc.computeSteeringAngle(state, desired_velocity);

        // Nội suy thực nghiệm
        steering = 0.018f * std::pow(steering, 3) + 1.5f * steering;

        // Clamp theo cơ cấu lái
        if (steering < -25.0f) steering = -25.0f;
        if (steering >  25.0f) steering =  25.0f;

        std::cout << "[MPC] steer=" << steering << std::endl;

        int servo = static_cast<int>(std::lround(97.0f + steering));
        comm.sendCommands(desired_velocity, servo);

        // Log thuần MPC
        std::stringstream ss;
        ss << "lat=" << state.lateral_deviation
           << ", yaw=" << state.yaw_angle
           << ", vel=" << desired_velocity
           << ", curv=" << (state.curvature.empty() ? 0.0f : state.curvature[0])
           << ", steer=" << steering
           << ", servo=" << servo;

        logger.log(ss.str(), 0.0);
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